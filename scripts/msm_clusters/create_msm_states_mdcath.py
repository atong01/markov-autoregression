"""Create k-means cluster assignments for MD-CATH domains using gyration radius and secondary structure."""

import argparse
import os

import deeptime.clustering
import mdtraj
import numpy as np
import tqdm


def _assignment_paths(cluster_data_dir, name, n_clusters, temp):
    return [
        os.path.join(cluster_data_dir, f"{name}_{n_clusters}_{temp}_{i}.npy")
        for i in range(5)
    ]


def _load_trajectories(data_dir, name, temp):
    top_path = os.path.join(data_dir, "topology", f"{name}.pdb")
    return [
        mdtraj.load_xtc(
            os.path.join(data_dir, "trajectory", f"{name}_{temp}_{i}.xtc"),
            top=top_path,
        )
        for i in range(5)
    ]


def _compute_features(trajectories):
    """Compute [gyration_radius, secondary_structure_fraction] per frame for each trajectory."""
    combined = []
    for traj in trajectories:
        gr = mdtraj.compute_rg(traj).reshape(-1, 1)
        dssp = mdtraj.compute_dssp(traj)
        sec_frac = (
            np.sum(np.isin(dssp, ["H", "G", "I", "E", "B"]), axis=1) / dssp.shape[1]
        ).reshape(-1, 1)
        combined.append(np.hstack([gr, sec_frac]))
    return combined


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
        trajectories = _load_trajectories(data_dir, name, temp)
        features = _compute_features(trajectories)
        features = _standardize(features)
        discrete = _cluster(features, n_clusters)

        for arr, path in zip(discrete, paths):
            np.save(path, arr)
        return name, True

    except Exception as e:
        print(f"Error processing {name}: {e}")
        return name, False


def process_all(data_dir, cluster_data_dir, n_clusters, temp, input_file):
    with open(input_file, "r") as f:
        pdb_ids = [line.strip() for line in f if line.strip()]

    print(f"Number of domains: {len(pdb_ids)}")

    processed = 0
    for pdb in tqdm.tqdm(pdb_ids):
        _, ok = process_domain(pdb, data_dir, cluster_data_dir, n_clusters, temp)
        processed += ok

    print(f"Processed: {processed}/{len(pdb_ids)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root directory with topology/ and trajectory/ subdirs")
    parser.add_argument("--cluster_data_dir", type=str, required=True,
                        help="Directory to save cluster assignment files")
    parser.add_argument("--input_file", type=str, default="./splits/mdCATH.txt")
    parser.add_argument("--msm_num_states", type=int, default=10)
    parser.add_argument("--temp", type=int, default=320)
    args = parser.parse_args()

    process_all(
        args.data_dir, args.cluster_data_dir, args.msm_num_states,
        args.temp, args.input_file,
    )
