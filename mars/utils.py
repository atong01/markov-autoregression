import random

import numpy as np
import torch

from .data.geometry import atom14_to_atom37
from .vendored.openfold import protein


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_offsets(ref_frame, rigids):
    """Compute rigid-body offsets relative to *ref_frame*.

    Chunks along T when T > 500k to avoid OOM on large trajectories.
    """
    B, T, L = rigids.shape
    if T > 500_000:
        offsets1 = ref_frame.invert().compose(rigids[:, :500_000]).to_tensor_7()
        offsets2 = ref_frame.invert().compose(rigids[:, 500_000:]).to_tensor_7()
        return torch.cat([offsets1, offsets2], 1)
    return ref_frame.invert().compose(rigids).to_tensor_7()


def atom14_to_pdb(atom14, aatype, path):
    """Write atom14 coordinates to a multi-model PDB file."""
    prots = []
    for pos in atom14:
        pos37 = atom14_to_atom37(pos, aatype)
        prots.append(_create_full_prot(pos37, aatype=aatype))
    with open(path, "w") as f:
        f.write(_prots_to_pdb(prots))


def _create_full_prot(atom37, aatype=None, b_factors=None):
    n = atom37.shape[0]
    atom37_mask = np.sum(np.abs(atom37), axis=-1) > 1e-7
    return protein.Protein(
        atom_positions=atom37,
        atom_mask=atom37_mask,
        aatype=aatype if aatype is not None else np.zeros(n, dtype=int),
        residue_index=np.arange(n),
        b_factors=b_factors if b_factors is not None else np.zeros([n, 37]),
        chain_index=np.zeros(n, dtype=int),
    )


def _prots_to_pdb(prots):
    parts = []
    for i, prot in enumerate(prots):
        parts.append(f"MODEL {i}")
        pdb_str = protein.to_pdb(prot)
        parts.append("\n".join(pdb_str.split("\n")[2:-3]))
        parts.append("ENDMDL")
    return "\n".join(parts) + "\n"
