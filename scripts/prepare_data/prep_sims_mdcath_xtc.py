"""Prepare MD-CATH simulation trajectories (xtc + pdb) into atom14 .npy arrays."""

import argparse
import os
from functools import partial
from multiprocessing import Pool

# Limit threads per process to prevent oversubscription
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import mdtraj
import numpy as np
import tqdm

from mars.vendored.openfold import residue_constants as rc

TEMPERATURES = [320, 348, 379, 413, 450]
NUM_REPLICAS = 5


def traj_to_atom14(traj):
    arr = np.zeros((traj.n_frames, traj.n_residues, 14, 3), dtype=np.float16)
    for i, resi in enumerate(traj.top.residues):
        res_name = resi.name
        if res_name in ("HSD", "HSE", "HSP"):
            res_name = "HIS"
        if res_name not in rc.restype_name_to_atom14_names:
            continue
        for at in resi.atoms:
            atom_name = at.name
            if resi.name == "MSE" and atom_name == "SE":
                break
            if resi.name == "ILE" and atom_name == "CD":
                atom_name = "CD1"
            if atom_name not in rc.restype_name_to_atom14_names[res_name]:
                continue
            j = rc.restype_name_to_atom14_names[res_name].index(atom_name)
            arr[:, i, j] = traj.xyz[:, at.index] * 10.0
    return arr


def process_domain(name, sim_dir, outdir, temps=TEMPERATURES):
    pdb_path = f"{sim_dir}/topology/{name}.pdb"
    if not os.path.exists(pdb_path):
        print(f"PDB file not found: {pdb_path}")
        return

    topology = mdtraj.load_pdb(pdb_path).topology

    for temp in temps:
        for i in range(NUM_REPLICAS):
            out_path = f"{outdir}/{name}_{temp}_{i}.npy"
            if os.path.exists(out_path):
                continue

            xtc_path = f"{sim_dir}/trajectory/{name}_{temp}_{i}.xtc"
            if not os.path.exists(xtc_path):
                continue

            traj = mdtraj.load(xtc_path, top=topology)
            traj.atom_slice(
                [a.index for a in traj.top.atoms if a.element.symbol != "H"], True
            )
            traj.superpose(traj)
            arr = traj_to_atom14(traj)
            np.save(out_path, arr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=str, default="splits/mdCATH.txt")
    parser.add_argument(
        "--sim_dir",
        type=str,
        required=True,
        help="Directory containing topology/{name}.pdb and trajectory/{name}_{temp}_{i}.xtc",
    )
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument(
        "--temps",
        type=int,
        nargs="+",
        default=TEMPERATURES,
        help="Temperatures to process (default: all)",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    with open(args.split, "r") as f:
        names = [line.strip() for line in f if line.strip()]

    print(f"Processing {len(names)} domains")

    if args.num_workers > 1:
        fn = partial(
            process_domain, sim_dir=args.sim_dir, outdir=args.outdir, temps=args.temps
        )
        with Pool(args.num_workers) as pool:
            list(tqdm.tqdm(pool.imap(fn, names), total=len(names)))
    else:
        for name in tqdm.tqdm(names):
            process_domain(name, args.sim_dir, args.outdir, temps=args.temps)
