"""Create PCCA MSM cluster assignments for tetrapeptides."""

import argparse
import os
from functools import partial
from multiprocessing import Pool

import numpy as np
import tqdm

import markov_autoregression.analysis


def _msm_cluster_path(data_dir, name, n_clusters):
    return os.path.join(data_dir, f"{name}_msm_cluster_{n_clusters}.npy")


def process_peptide(name, data_dir, n_clusters):
    np.random.seed(2137)

    out_path = _msm_cluster_path(data_dir, name, n_clusters)
    if os.path.exists(out_path):
        print(f"File already exists for {name}, skipping.")
        return name, True

    try:
        _, ref = mars.analysis.get_featurized_traj(
            f"{data_dir}/{name}/{name}", sidechains=True, cossin=True
        )

        tica_model, _ = mars.analysis.get_tica(ref)
        _, ref_kmeans = mars.analysis.get_kmeans(tica_model.transform(ref))

        assignments, _ = mars.analysis.get_msm(ref_kmeans[0], nstates=n_clusters)
        discrete = assignments[ref_kmeans]

        np.save(out_path, discrete)
        print(f"Processed {name}: file saved.")
        return name, True

    except Exception as e:
        print(f"Error processing {name}: {e}")
        return name, False


def process_all(data_dir, num_workers, n_clusters):
    pdb_ids = [
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
        and os.path.exists(f"{data_dir}/{d}/{d}.xtc")
        and os.path.exists(f"{data_dir}/{d}/{d}.pdb")
    ]
    print(f"Number of trajectories: {len(pdb_ids)}")

    fn = partial(process_peptide, data_dir=data_dir, n_clusters=n_clusters)

    if num_workers > 1:
        with Pool(num_workers) as pool:
            results = dict(tqdm.tqdm(pool.imap(fn, pdb_ids), total=len(pdb_ids)))
    else:
        results = dict(tqdm.tqdm(map(fn, pdb_ids), total=len(pdb_ids)))

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing peptide simulation subdirectories")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--msm_num_states", type=int, default=10)
    args = parser.parse_args()

    process_all(args.data_dir, args.num_workers, args.msm_num_states)
