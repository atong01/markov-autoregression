"""Prepare tetrapeptide simulation trajectories into atom14 .npy arrays.

Based on the preprocessing script from https://github.com/bjing2016/mdgen
by Bowen Jing, Hannes Stärk, Tommi Jaakkola, and Bonnie Berger (MIT License).
"""

import argparse
import os
from functools import partial
from multiprocessing import Pool

import mdtraj
import numpy as np
import pandas as pd
import tqdm

from mars.vendored.openfold import residue_constants as rc


def traj_to_atom14(traj):
    arr = np.zeros((traj.n_frames, traj.n_residues, 14, 3), dtype=np.float16)
    for i, resi in enumerate(traj.top.residues):
        for at in resi.atoms:
            if at.name not in rc.restype_name_to_atom14_names[resi.name]:
                print(resi.name, at.name, "not found")
                continue
            j = rc.restype_name_to_atom14_names[resi.name].index(at.name)
            arr[:, i, j] = traj.xyz[:, at.index] * 10.0
    return arr


def process_peptide(name, sim_dir, outdir, stride):
    traj = mdtraj.load(
        f"{sim_dir}/{name}/{name}.xtc", top=f"{sim_dir}/{name}/{name}.pdb"
    )
    traj.superpose(traj)
    arr = traj_to_atom14(traj)
    np.save(f"{outdir}/{name}.npy", arr[::stride])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=str, default="splits/4AA.csv")
    parser.add_argument("--sim_dir", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--stride", type=int, default=100)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.split, index_col="name")

    jobs = [
        name for name in df.index
        if not os.path.exists(f"{args.outdir}/{name}.npy")
    ]
    print(f"Processing {len(jobs)}/{len(df)} peptides")

    fn = partial(process_peptide, sim_dir=args.sim_dir, outdir=args.outdir, stride=args.stride)

    if args.num_workers > 1:
        with Pool(args.num_workers) as pool:
            list(tqdm.tqdm(pool.imap(fn, jobs), total=len(jobs)))
    else:
        list(tqdm.tqdm(map(fn, jobs), total=len(jobs)))
