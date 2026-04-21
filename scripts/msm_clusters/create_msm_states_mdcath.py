"""Create k-means cluster assignments for MD-CATH domains using gyration radius and secondary structure."""

import argparse
import os
from functools import partial
from multiprocessing import Pool

# Limit threads per process to prevent oversubscription
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import deeptime.clustering
import h5py
import numpy as np
import tqdm

NUM_REPLICAS = 5


def _assignment_paths(cluster_data_dir, name, n_clusters, temp):
    return [
        os.path.join(cluster_data_dir, f"{name}_{n_clusters}_{temp}_{i}.npy")
        for i in range(NUM_REPLICAS)
    ]


def _load_features_from_h5(data_dir, name, temp):
    """Load [gyration_radius, secondary_structure_fraction] per frame from h5 file."""
    h5_path = os.path.join(data_dir, f"mdcath_dataset_{name}.h5")
    features = []

    with h5py.File(h5_path, "r") as f:
        domain_group = f[name]
        temp_str = str(temp)

        for i in range(NUM_REPLICAS):
            replica_str = str(i)
            replica_group = domain_group[f"{temp_str}/{replica_str}"]

            # Gyration radius (already in nm)
            gr = np.array(replica_group["gyrationRadius"]).reshape(-1, 1)

            # DSSP - secondary structure (byte strings like b'H', b'E', etc.)
            dssp = np.array(replica_group["dssp"])
            # Count helix (H, G, I) and sheet (E, B) residues
            sec_mask = (
                (dssp == b"H") | (dssp == b"G") | (dssp == b"I") |
                (dssp == b"E") | (dssp == b"B")
            )
            sec_frac = (np.sum(sec_mask, axis=1) / dssp.shape[1]).reshape(-1, 1)

            features.append(np.hstack([gr, sec_frac]))

    return features


def _standardize(feature_list):
    stacked = np.concatenate(feature_list, axis=0)
    mean, std = stacked.mean(axis=0), stacked.std(axis=0)
    return [(f - mean) / std for f in feature_list]


def _cluster(feature_list, n_clusters):
    model = deeptime.clustering.KMeans(
        n_clusters=n_clusters, max_iter=100, fixed_seed=2137
    ).fit(feature_list)
    return [model.transform(f) for f in feature_list]


def process_domain(name, data_dir, cluster_data_dir, n_clusters, temp):
    np.random.seed(2137)

    paths = _assignment_paths(cluster_data_dir, name, n_clusters, temp)
    if all(os.path.exists(p) for p in paths):
        return name, True

    try:
        features = _load_features_from_h5(data_dir, name, temp)
        features = _standardize(features)
        discrete = _cluster(features, n_clusters)

        for arr, path in zip(discrete, paths):
            np.save(path, arr)
        return name, True

    except Exception as e:
        print(f"Error processing {name}: {e}")
        return name, False


def process_all(data_dir, cluster_data_dir, n_clusters, temp, input_file, num_workers=1):
    with open(input_file, "r") as f:
        pdb_ids = [line.strip() for line in f if line.strip()]

    # Filter out already-processed domains
    remaining = []
    for name in pdb_ids:
        paths = _assignment_paths(cluster_data_dir, name, n_clusters, temp)
        if not all(os.path.exists(p) for p in paths):
            remaining.append(name)

    print(f"Total domains: {len(pdb_ids)}, Already done: {len(pdb_ids) - len(remaining)}, Remaining: {len(remaining)}")

    if not remaining:
        print("All domains already processed!")
        return

    fn = partial(process_domain, data_dir=data_dir, cluster_data_dir=cluster_data_dir,
                 n_clusters=n_clusters, temp=temp)

    if num_workers > 1:
        with Pool(num_workers) as pool:
            results = list(tqdm.tqdm(pool.imap_unordered(fn, remaining), total=len(remaining)))
    else:
        results = [fn(name) for name in tqdm.tqdm(remaining)]

    processed = sum(1 for _, ok in results if ok)
    print(f"Processed: {processed}/{len(remaining)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing mdcath_dataset_*.h5 files")
    parser.add_argument("--cluster_data_dir", type=str, required=True,
                        help="Directory to save cluster assignment files")
    parser.add_argument("--input_file", type=str, default="./splits/mdCATH.txt")
    parser.add_argument("--msm_num_states", type=int, default=10)
    parser.add_argument("--temp", type=int, default=320)
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of parallel workers")
    args = parser.parse_args()

    os.makedirs(args.cluster_data_dir, exist_ok=True)

    process_all(
        args.data_dir, args.cluster_data_dir, args.msm_num_states,
        args.temp, args.input_file, args.num_workers,
    )
