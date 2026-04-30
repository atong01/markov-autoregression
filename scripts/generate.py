"""Generate MD trajectories using MarS (and optionally MDGen)."""

import argparse
import os

import mdtraj
import numpy as np
import pandas as pd
import torch
torch.serialization.add_safe_globals([argparse.Namespace])
from tqdm import trange

from markov_autoregression.data.geometry import (
    atom14_to_frames,
    atom14_to_atom37,
    atom37_to_torsions,
    atom14_to_backbone,
    atom14_to_ca,
    center_dense_coords,
)
from markov_autoregression.vendored.openfold.residue_constants import restype_order, atom_order
from markov_autoregression.vendored.openfold.tensor_utils import tensor_tree_map
from markov_autoregression.utils import atom14_to_pdb, set_seed
from markov_autoregression.model.module import MDGenModule, MarSModule
from markov_autoregression.model.autoregressive_model import MarSARModule


def _is_euclidean(model):
    return getattr(model.args, "euclidean", False)


def _is_ca_only(model):
    return getattr(model.args, "ca_only", False)


def _repeat_batch(batch, n):
    return {k: v.repeat(n, *([1] * (v.ndim - 1))) for k, v in batch.items()}


def _recenter(coords):
    # coords: (B, L, n_atoms, 3)
    B = coords.shape[0]
    centroid = coords.reshape(B, -1, 3).mean(dim=1)
    return coords - centroid.reshape(B, 1, 1, 3)


def _update_batch(batch, atom14, *, euclidean=False):
    """Build a new batch from the last frame of a generated trajectory."""
    new_batch = {**batch}
    if euclidean:
        # In euclidean mode the "atom14" tensor returned by inference is
        # actually coords of shape (B, T, L, n_atoms, 3).
        last = _recenter(atom14[:, -1])                   # (B, L, n_atoms, 3)
        last = last.unsqueeze(1)                          # (B, 1, L, n_atoms, 3)
        new_batch["coords"] = last
        new_batch["coords_plus_tau"] = last
        return new_batch

    frames = atom14_to_frames(atom14[:, -1])
    new_batch["trans"] = frames._trans.unsqueeze(1)
    new_batch["rots"] = frames._rots._rot_mats.unsqueeze(1)
    atom37 = atom14_to_atom37(atom14[:, -1].cpu(), batch["seqres"].cpu())
    torsions, _ = atom37_to_torsions(atom37, batch["seqres"].cpu())
    new_batch["torsions"] = torsions.unsqueeze(1).to(atom14.device)
    return new_batch


def _ca_coords_to_atom14(coords):
    """Embed CA-only coords (..., 1, 3) into a full atom14 (..., 14, 3)."""
    pad = list(coords.shape)
    pad[-2] = 14
    out = torch.zeros(*pad, dtype=coords.dtype, device=coords.device)
    out[..., atom_order["CA"], :] = coords[..., 0, :]
    return out


def _save_trajectory(atom14_cat, seqres, out_dir, name):
    pdb_path = os.path.join(out_dir, f"{name}.pdb")
    atom14_to_pdb(atom14_cat[0].cpu().numpy(), seqres[0].cpu().numpy(), pdb_path)
    traj = mdtraj.load(pdb_path)
    traj.superpose(traj)
    traj.save(os.path.join(out_dir, f"{name}.xtc"))
    traj[0].save(pdb_path)


def load_starting_structure(
    data_dir, name, seqres, mdcath=False, temp=320,
    euclidean=False, ca_only=False,
):
    if mdcath:
        arr = np.load(f"{data_dir}/{name}_{temp}_0.npy")
    else:
        arr = np.lib.format.open_memmap(f"{data_dir}/{name}.npy", "r")
    arr = np.copy(arr[0:1]).astype(np.float32)

    seqres = torch.tensor([restype_order[c] for c in seqres])
    mask = torch.ones(len(seqres))

    if euclidean:
        x = torch.from_numpy(arr)                 # (1, L, 14, 3)
        coords = atom14_to_ca(x) if ca_only else atom14_to_backbone(x)
        coords = center_dense_coords(coords)
        return {
            "coords": coords,
            "coords_plus_tau": coords,
            "seqres": seqres,
            "mask": mask,
        }

    frames = atom14_to_frames(torch.from_numpy(arr))
    atom37 = torch.from_numpy(atom14_to_atom37(arr, seqres[None])).float()
    torsions, torsion_mask = atom37_to_torsions(atom37, seqres[None])
    return {
        "torsions": torsions,
        "torsion_mask": torsion_mask[0],
        "trans": frames._trans,
        "rots": frames._rots._rot_mats,
        "seqres": seqres,
        "mask": mask,
    }


def rollout_mars(model, batch, num_steps=50):
    atom14, _ = model.inference(batch, num_steps=num_steps)
    return atom14, _update_batch(batch, atom14, euclidean=_is_euclidean(model))


def rollout_mdgen(model, batch, num_steps=50):
    # MDGen expects time-expanded inputs and returns a single-sample batch,
    # so batch update uses [0] indexing rather than the batched _update_batch.
    expanded_batch = {
        "torsions": batch["torsions"].expand(-1, model.args.num_frames, -1, -1, -1),
        "torsion_mask": batch["torsion_mask"],
        "trans": batch["trans"].expand(-1, model.args.num_frames, -1, -1),
        "rots": batch["rots"].expand(-1, model.args.num_frames, -1, -1, -1),
        "seqres": batch["seqres"],
        "mask": batch["mask"],
    }
    atom14, _ = model.inference(expanded_batch, num_steps=num_steps)
    new_batch = {**batch}
    frames = atom14_to_frames(atom14[:, -1])
    new_batch["trans"] = frames._trans[None]
    new_batch["rots"] = frames._rots._rot_mats[None]
    atom37 = atom14_to_atom37(atom14[0, -1].cpu(), batch["seqres"][0].cpu())
    torsions, _ = atom37_to_torsions(atom37, batch["seqres"][0].cpu())
    new_batch["torsions"] = torsions[None, None].to(atom14.device)
    return atom14, new_batch


def _run_tree_mars(model, batch, args):
    """Breadth-first tree sampling: each level branches calls_mars times,
    stopping once max_mars_samples trajectories have been collected."""
    frontier = [batch]
    collected = []
    total = 0

    while frontier and total < args.max_mars_samples:
        mega_batch = {}
        M, N = len(frontier), args.calls_mars
        for key in frontier[0].keys():
            stacked = torch.cat([p[key] for p in frontier], dim=0)
            expanded = stacked.unsqueeze(1).repeat(*([1, N] + [1] * (stacked.ndim - 1)))
            permuted = expanded.permute(1, 0, *range(2, expanded.ndim))
            mega_batch[key] = permuted.reshape(N * M, *stacked.shape[1:])

        size_key = "coords" if "coords" in mega_batch else "torsions"
        n = mega_batch[size_key].shape[0]
        next_frontier = []
        for start in range(0, n, args.tree_parallel_chunk):
            end = min(start + args.tree_parallel_chunk, n)
            sub_batch = {k: v[start:end] for k, v in mega_batch.items()}
            atom14, stacked_batch = rollout_mars(model, sub_batch, num_steps=args.num_steps)
            collected.append(atom14)
            total += atom14.shape[0]

            for i in range(atom14.shape[0]):
                next_frontier.append({k: v[i : i + 1] for k, v in stacked_batch.items()})

            if total >= args.max_mars_samples:
                return collected

        frontier = next_frontier

    return collected


def _run_flat_mars_mdgen(model_mars, model_mdgen, batch, args):
    """Flat sampling: parallel MarS calls, then optional MDGen continuation."""
    collected = []

    repeated_batch = _repeat_batch(batch, args.calls_mars)
    atom14, stacked_batch = rollout_mars(
        model_mars, repeated_batch, num_steps=args.num_steps,
    )
    collected.append(atom14)

    if args.calls_mdgen > 0:
        mars_batches = [
            {k: v[i : i + 1] for k, v in stacked_batch.items()}
            for i in range(args.calls_mars)
        ]
        for m_batch in mars_batches:
            current_batch = m_batch
            for _ in trange(args.calls_mdgen):
                atom14, current_batch = rollout_mdgen(
                    model_mdgen, current_batch, num_steps=args.num_steps,
                )
                collected.append(atom14)

    return collected


@torch.no_grad()
def generate(args):
    assert args.calls_mars > 0, "MarS calls must be > 0"
    if args.tree:
        assert args.max_mars_samples is not None, "--max_mars_samples required with --tree"
    if args.calls_mdgen > 0:
        assert args.mdgen_ckpt is not None, "--mdgen_ckpt required for --calls_mdgen"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ModelClass = MarSARModule if getattr(args, "ar", False) else MarSModule
    model_mars = ModelClass.load_from_checkpoint(args.mars_ckpt)
    model_mars.eval().to(device)

    model_mdgen = None
    if args.mdgen_ckpt:
        model_mdgen = MDGenModule.load_from_checkpoint(args.mdgen_ckpt)
        model_mdgen.eval().to(device)

    df = pd.read_csv(args.split, index_col="name")

    if args.skip_existing:
        all_names = [
            n for n in df.index
            if not os.path.exists(os.path.join(args.out_dir, f"{n}.pdb"))
        ]
    else:
        all_names = list(df.index)

    print("Number of proteins to process:", len(all_names))
    for name in all_names:
        if args.pdb_id and name not in args.pdb_id:
            continue
        print("Processing:", name)
        seqres = df.seqres[name]

        item = load_starting_structure(
            args.data_dir, name, seqres, mdcath=args.mdcath, temp=args.temp,
            euclidean=_is_euclidean(model_mars),
            ca_only=_is_ca_only(model_mars),
        )
        batch = next(iter(torch.utils.data.DataLoader([item])))
        batch = tensor_tree_map(lambda x: x.to(device), batch)
        all_atom14 = []

        if args.calls_mdgen > 0:
            atom14, _ = rollout_mdgen(
                model_mdgen, batch, num_steps=args.num_steps,
            )
            all_atom14.append(atom14)

        if args.tree:
            all_atom14.extend(_run_tree_mars(model_mars, batch, args))
        else:
            all_atom14.extend(
                _run_flat_mars_mdgen(model_mars, model_mdgen, batch, args)
            )

        if _is_euclidean(model_mars) and _is_ca_only(model_mars):
            # inference returns coords (B, T, L, 1, 3) — embed CA into atom14.
            all_atom14 = [_ca_coords_to_atom14(t) for t in all_atom14]
        all_atom14_cat = torch.cat(
            [t.reshape(1, -1, *t.shape[2:]) for t in all_atom14], 1
        )
        _save_trajectory(all_atom14_cat, batch["seqres"], args.out_dir, name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MD trajectories with MarS (+ optional MDGen).")
    parser.add_argument("--mars_ckpt", type=str, required=True)
    parser.add_argument("--mdgen_ckpt", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--pdb_id", nargs="*", default=[])
    parser.add_argument("--out_dir", type=str, default=".")
    parser.add_argument("--split", type=str, default="splits/4AA_test.csv")
    parser.add_argument("--calls_mars", type=int, default=5,
        help="Number of parallel MarS rollouts.",
    )
    parser.add_argument("--calls_mdgen", type=int, default=0,
        help="MDGen continuation steps after each MarS rollout. "
        "When > 0, one initial MDGen call on the starting structure is also performed.",
    )
    parser.add_argument("--mdcath", action="store_true")
    parser.add_argument("--temp", type=int, default=320)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--tree", action="store_true",
        help="Enable breadth-first tree sampling (requires --max_mars_samples).",
    )
    parser.add_argument("--max_mars_samples", type=int, default=None,
        help="Stop tree sampling after this many trajectories.",
    )
    parser.add_argument("--tree_parallel_chunk", type=int, default=1000,
        help="Max trajectories fed to MarS in a single GPU call.",
    )
    parser.add_argument("--num_steps", type=int, default=50,
        help="Number of ODE integration steps for sampling (flow-matching only; "
             "ignored when --ar is set).",
    )
    parser.add_argument("--ar", action="store_true",
        help="Load checkpoint as MarSARModule (autoregressive model). "
             "Otherwise loads MarSModule (flow-matching).",
    )

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    generate(args)
