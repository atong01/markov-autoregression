import os

import deeptime
import numpy as np
import pandas as pd
import torch
import tqdm

from .geometry import atom37_to_torsions, atom14_to_atom37, atom14_to_frames, atom14_to_backbone, atom14_to_ca, center_dense_coords, geometric_augmentation
from ..vendored.openfold.residue_constants import restype_order
from ..vendored.openfold.rigid_utils import Rigid


def _is_rank_0() -> bool:
    """True on rank 0 of a DDP run, or always True if not under DDP."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    # Pre-DDP-init code path (datasets are built before trainer.fit on each
    # rank). Fall back to SLURM env so we still avoid 4× log spam.
    return int(os.environ.get("SLURM_PROCID", "0")) == 0


def _rprint(*a, **kw):
    if _is_rank_0():
        print(*a, **kw)


class MarSDatasetBase(torch.utils.data.Dataset):
    """Shared logic for MSM-guided trajectory sampling."""

    def __init__(self, args, split, repeat=1, translate=False):
        super().__init__()
        self.df = pd.read_csv(split, index_col="name")
        self.args = args
        self.repeat = repeat
        self.num_transitions_per_traj = args.samples_per_cluster * args.clusters_per_batch
        self.num_samples_per_cluster = args.samples_per_cluster
        self.num_clusters_to_sample = args.clusters_per_batch
        self.euclidean = args.euclidean
        self.ca_only = args.euclidean and args.ca_only
        self.translate = translate
        self.files = {}

    def __len__(self):
        return self.repeat * len(self.valid_names)

    def _build_transition_matrix(self, clusters_list):
        counts_estimator = deeptime.markov.TransitionCountEstimator(
            self.args.msm_lagtime, "sliding"
        )
        counts = counts_estimator.fit_fetch(clusters_list)
        msm = deeptime.markov.msm.MaximumLikelihoodMSM(
            allow_disconnected=True, reversible=True
        ).fit_fetch(counts)
        transition_matrix = msm.transition_matrix
        return (transition_matrix + transition_matrix.T) / 2

    def _sample_clusters(
        self, arr, clusters, probabilities, cluster_members, unique_clusters
    ):
        first_cluster = clusters[0]
        if self.num_clusters_to_sample - 1 > 0:
            remaining_clusters = np.setdiff1d(
                unique_clusters, np.array([first_cluster])
            )
            if remaining_clusters.size == 0:
                additional_x0 = np.full(self.num_clusters_to_sample - 1, first_cluster)
            else:
                additional_x0 = np.random.choice(
                    remaining_clusters,
                    size=self.num_clusters_to_sample - 1,
                    replace=True,
                )
            x0_clusters = np.concatenate(([first_cluster], additional_x0))
        else:
            x0_clusters = np.array([first_cluster])

        x1_clusters = np.array(
            [
                np.random.choice(
                    np.arange(probabilities.shape[0]),
                    size=self.num_samples_per_cluster,
                    replace=True,
                    p=probabilities[x0] / probabilities[x0].sum(),
                )
                for x0 in x0_clusters
            ]
        )
        x0_clusters = np.repeat(x0_clusters, self.num_samples_per_cluster)
        x1_clusters = x1_clusters.flatten()

        x_t_indices = self._vectorized_sample(x0_clusters, cluster_members)
        x_t_plus_tau_indices = self._vectorized_sample(x1_clusters, cluster_members)

        return arr[x_t_indices].astype(np.float32), arr[x_t_plus_tau_indices].astype(
            np.float32
        )

    @staticmethod
    def _vectorized_sample(cluster_array, cluster_members):
        sampled = np.empty(cluster_array.shape[0], dtype=int)
        for cl in np.unique(cluster_array):
            group_idxs = np.where(cluster_array == cl)[0]
            members = cluster_members[cl]
            sampled[group_idxs] = members[
                np.random.randint(0, len(members), size=group_idxs.shape[0])
            ]
        return sampled

    def _process_output(self, x_t, x_t_plus_tau, seqres_str, name):
        """Convert raw atom14 arrays to (frames, torsions, and masks) or (coords and masks)"""
        seqres = np.array([restype_order[c] for c in seqres_str])

        if self.euclidean:
            if self.ca_only:
                coords = atom14_to_ca(torch.from_numpy(x_t))
                coords_plus_tau = atom14_to_ca(torch.from_numpy(x_t_plus_tau))
            else:
                coords = atom14_to_backbone(torch.from_numpy(x_t))
                coords_plus_tau = atom14_to_backbone(torch.from_numpy(x_t_plus_tau))
            
            coords = center_dense_coords(coords)
            coords_plus_tau = center_dense_coords(coords_plus_tau)
            if self.args.se3_augmentation:
                coords, coords_plus_tau = geometric_augmentation(coords, coords_plus_tau, s_trans=self.args.s_translation, translate=self.translate)
            
            L = coords.shape[1]
            mask = np.ones(L, dtype=np.float32)

            return {
                "name": name,
                "coords": coords,
                "coords_plus_tau": coords_plus_tau,
                "seqres": seqres,
                "mask": mask,
            }
        
        frames = atom14_to_frames(torch.from_numpy(x_t))
        frames_plus_tau = atom14_to_frames(torch.from_numpy(x_t_plus_tau))

        aatype = torch.from_numpy(seqres)[None].expand(
            self.num_transitions_per_traj, -1
        )

        atom37 = torch.from_numpy(atom14_to_atom37(x_t, aatype)).float()
        atom37_plus_tau = torch.from_numpy(
            atom14_to_atom37(x_t_plus_tau, aatype)
        ).float()

        L = frames.shape[1]
        mask = np.ones(L, dtype=np.float32)
        torsions, torsion_mask = atom37_to_torsions(atom37, aatype)
        torsions_plus_tau, _ = atom37_to_torsions(atom37_plus_tau, aatype)
        torsion_mask = torsion_mask[0]

        return {
            "name": name,
            "torsions": torsions,
            "torsions_plus_tau": torsions_plus_tau,
            "torsion_mask": torsion_mask,
            "trans": frames._trans,
            "trans_plus_tau": frames_plus_tau._trans,
            "rots": frames._rots._rot_mats,
            "rots_plus_tau": frames_plus_tau._rots._rot_mats,
            "seqres": seqres,
            "mask": mask,
        }

    @staticmethod
    def _read_trajectory(path):
        # Check file size first to avoid reading corrupted/empty files
        if os.path.getsize(path) < 128:  # Minimum valid npy file size
            raise ValueError(f"File too small, likely corrupted: {path}")
        mm = np.lib.format.open_memmap(path, mode="r")
        try:
            return np.array(mm, copy=True)
        finally:
            mm._mmap.close()
            del mm


# ---------------------------------------------------------------------------
# Tetrapeptide dataset
# ---------------------------------------------------------------------------


class MarSDataset4AA(MarSDatasetBase):
    def __init__(self, args, split, repeat=1, translate=False):
        super().__init__(args, split, repeat, translate=translate)

        for name in self.df.index:
            file_path, msm_cluster_file = self._construct_file_paths(name)
            if not (os.path.exists(file_path) and os.path.exists(msm_cluster_file)):
                _rprint(f"Missing file for {name}")
                continue
            # Check file sizes to skip corrupted files
            if os.path.getsize(file_path) < 128 or os.path.getsize(msm_cluster_file) < 128:
                _rprint(f"Corrupted file for {name} (file too small)")
                continue
            try:
                self.files[name] = self._load_data(file_path, msm_cluster_file)
            except (ValueError, OSError) as e:
                _rprint(f"Failed to load {name}: {e}")
                continue

        self.valid_names = list(self.files.keys())

    def _construct_file_paths(self, name):
        base = self.args.data_dir
        n = self.args.msm_num_states
        file_path = f"{base}/{name}.npy"
        msm_cluster_file = f"{base}/{name}_msm_cluster_{n}.npy"
        return file_path, msm_cluster_file

    def _load_data(self, file_path, msm_cluster_file):
        clusters = np.load(msm_cluster_file).flatten()[::100]
        transition_matrix = self._build_transition_matrix(clusters)
        cluster_members = [
            np.where(clusters == c)[0] for c in range(transition_matrix.shape[0])
        ]
        probabilities = transition_matrix.copy()

        missing_clusters = np.setdiff1d(np.arange(probabilities.shape[0]), clusters)
        if missing_clusters.size > 0:
            probabilities[missing_clusters, :] = 0
            probabilities[:, missing_clusters] = 0

        return {
            "arr_path": file_path,
            "clusters": clusters,
            "probabilities": probabilities,
            "cluster_members": cluster_members,
            "unique_clusters": np.unique(clusters),
        }

    def __getitem__(self, idx):
        idx = idx % len(self.valid_names)
        # Try up to 3 different samples if we hit corrupted files
        for attempt in range(3):
            try:
                name = self.valid_names[(idx + attempt) % len(self.valid_names)]
                data = self.files[name]

                arr = self._read_trajectory(data["arr_path"])
                x_t, x_t_plus_tau = self._sample_clusters(
                    arr, data["clusters"], data["probabilities"],
                    data["cluster_members"], data["unique_clusters"],
                )
                return self._process_output(x_t, x_t_plus_tau, name, name)
            except (ValueError, OSError) as e:
                if attempt == 2:
                    raise RuntimeError(f"Failed to load data after 3 attempts: {e}")
                continue


# ---------------------------------------------------------------------------
# MD-CATH dataset
# ---------------------------------------------------------------------------


class MarSDatasetMDCath(MarSDatasetBase):
    def __init__(self, args, split, repeat=1, translate=False):
        super().__init__(args, split, repeat, translate=translate)
        num_single_state = 0

        for name in tqdm.tqdm(self.df.index):
            file_paths, msm_cluster_files = self._construct_file_paths(name)

            if not self._validate_files(file_paths):
                _rprint(f"Missing trajectory file(s) for {name}. Skipping...")
                continue

            arr_list, clusters_list = self._load_arrays_and_clusters(
                file_paths, msm_cluster_files
            )
            arr_list, clusters_list, single_state = self._filter_single_state(
                arr_list, clusters_list
            )
            if len(arr_list) == 0:
                _rprint(f"Protein {name} has only one state. Skipping...")
                num_single_state += 1
                continue

            transition_matrix = self._build_transition_matrix(clusters_list)
            cluster_members_list = self._compute_cluster_members(
                clusters_list, transition_matrix
            )
            probabilities = transition_matrix.copy()
            probabilities, unique_clusters = self._fix_missing_clusters(
                probabilities, clusters_list
            )

            self.files[name] = {
                "arr_paths": arr_list,
                "clusters": clusters_list,
                "probabilities": probabilities,
                "cluster_members": cluster_members_list,
                "unique_clusters": unique_clusters,
            }

            if all(single_state):
                num_single_state += 1

        self.valid_names = list(self.files.keys())
        _rprint("Number of proteins with only one state:", num_single_state)

    def _construct_file_paths(self, name):
        base = self.args.data_dir
        temp = self.args.data_temperature
        n = self.args.msm_num_states
        file_paths = [f"{base}/{name}_{temp}_{i}.npy" for i in range(5)]
        msm_cluster_files = [f"{base}/{name}_{n}_{temp}_{i}.npy" for i in range(5)]
        return file_paths, msm_cluster_files

    @staticmethod
    def _validate_files(file_paths):
        for fp in file_paths:
            if not os.path.exists(fp):
                return False
            # Check file size - empty/corrupted files cause EOF errors
            if os.path.getsize(fp) < 128:
                return False
        return True

    def _load_arrays_and_clusters(self, file_paths, msm_cluster_files):
        clusters_list = []
        for cf, fp in zip(msm_cluster_files, file_paths):
            if os.path.exists(cf) and os.path.getsize(cf) >= 128:
                try:
                    clusters_list.append(np.load(cf, mmap_mode=None))
                except (ValueError, OSError) as e:
                    _rprint(f"Warning: Failed to load cluster file {cf}: {e}")
                    length = self._read_trajectory(fp).shape[0]
                    clusters_list.append(np.zeros(length, dtype=np.int32))
            else:
                length = self._read_trajectory(fp).shape[0]
                clusters_list.append(np.zeros(length, dtype=np.int32))
        return file_paths, clusters_list

    @staticmethod
    def _filter_single_state(arr_list, clusters_list):
        single_state = [np.unique(c).shape[0] == 1 for c in clusters_list]
        arr_list = [a for a, s in zip(arr_list, single_state) if not s]
        clusters_list = [c for c, s in zip(clusters_list, single_state) if not s]
        return arr_list, clusters_list, single_state

    @staticmethod
    def _compute_cluster_members(clusters_list, transition_matrix):
        num_clusters = transition_matrix.shape[0]
        return [
            [np.where(clusters == c)[0] for c in range(num_clusters)]
            for clusters in clusters_list
        ]

    @staticmethod
    def _fix_missing_clusters(probabilities, clusters_list):
        num_clusters = probabilities.shape[0]
        all_clusters = np.concatenate(clusters_list)
        missing = np.setdiff1d(np.arange(num_clusters), all_clusters)
        if missing.size > 0:
            probabilities[missing, :] = 0
            probabilities[:, missing] = 0
        return probabilities, np.unique(all_clusters)

    def __getitem__(self, idx):
        idx = idx % len(self.valid_names)
        # Try up to 3 different samples if we hit corrupted files
        for attempt in range(3):
            try:
                name = self.valid_names[(idx + attempt) % len(self.valid_names)]
                seqres = self.df.seqres[name]
                full_name = f"{name}_{self.args.data_temperature}"
                data = self.files[name]

                arr, clusters, probabilities, cluster_members, unique_clusters = (
                    self._merge_replicas(data)
                )
                x_t, x_t_plus_tau = self._sample_clusters(
                    arr, clusters, probabilities, cluster_members, unique_clusters
                )
                return self._crop_pad_output(
                    self._process_output(x_t, x_t_plus_tau, seqres, full_name)
                )
            except (ValueError, OSError) as e:
                if attempt == 2:
                    raise RuntimeError(f"Failed to load data after 3 attempts: {e}")
                continue

    def _merge_replicas(self, data):
        arr = np.concatenate(
            [self._read_trajectory(p) for p in data["arr_paths"]]
        )
        clusters = np.concatenate(data["clusters"])
        if arr.shape[0] != clusters.shape[0]:
            raise ValueError("Array and clusters have different lengths")
        probabilities = data["probabilities"].copy()
        cluster_members = self._concat_replica_members(data["cluster_members"])
        return arr, clusters, probabilities, cluster_members, data["unique_clusters"]

    @staticmethod
    def _concat_replica_members(replica_lists):
        num_states = len(replica_lists[0])
        max_index = 0
        concatenated = [np.array([], dtype=int) for _ in range(num_states)]
        for replica in replica_lists:
            for state_idx, state_array in enumerate(replica):
                concatenated[state_idx] = np.concatenate(
                    (concatenated[state_idx], state_array + max_index)
                )
            non_empty = [a for a in replica if a.size > 0]
            if non_empty:
                max_index += max(map(max, non_empty)) + 1
        return concatenated

    def _crop_pad_output(self, out):
        """Crop or pad variable-length proteins to self.args.crop."""
        crop = self.args.crop
        L = out["mask"].shape[0]

        if L > crop:
            s = np.random.randint(0, L - crop + 1)
            e = s + crop
            if self.euclidean:
                out["coords"] = out["coords"][:, s:e]
                out["coords_plus_tau"] = out["coords_plus_tau"][:, s:e]
            else:
                out["torsions"] = out["torsions"][:, s:e]
                out["torsions_plus_tau"] = out["torsions_plus_tau"][:, s:e]
                out["torsion_mask"] = out["torsion_mask"][s:e]
                out["trans"] = out["trans"][:, s:e]
                out["trans_plus_tau"] = out["trans_plus_tau"][:, s:e]
                out["rots"] = out["rots"][:, s:e]
                out["rots_plus_tau"] = out["rots_plus_tau"][:, s:e]
            out["seqres"] = out["seqres"][s:e]
            out["mask"] = out["mask"][s:e]

        elif L < crop:
            pad = crop - L
            n = self.num_transitions_per_traj

            out["mask"] = np.concatenate([out["mask"], np.zeros(pad, dtype=np.float32)])
            out["seqres"] = np.concatenate([out["seqres"], np.zeros(pad, dtype=int)])

            if self.euclidean:
                n_atoms = out["coords"].shape[-2]
                pad_coords = torch.zeros((n, pad, n_atoms, 3), dtype=torch.float32)
                out["coords"] = torch.cat([out["coords"], pad_coords], dim=1)
                out["coords_plus_tau"] = torch.cat([out["coords_plus_tau"], pad_coords], dim=1)

            else:
                pad_rigid = Rigid.identity((n, pad), requires_grad=False, fmt="rot_mat")

                out["trans"] = torch.cat([out["trans"], pad_rigid._trans], 1)
                out["trans_plus_tau"] = torch.cat([out["trans_plus_tau"], pad_rigid._trans], 1)
                out["rots"] = torch.cat([out["rots"], pad_rigid._rots._rot_mats], 1)
                out["rots_plus_tau"] = torch.cat([out["rots_plus_tau"], pad_rigid._rots._rot_mats], 1)
                out["torsions"] = torch.cat(
                    [out["torsions"], torch.zeros((n, pad, 7, 2), dtype=torch.float32)], 1
                )
                out["torsions_plus_tau"] = torch.cat(
                    [out["torsions_plus_tau"], torch.zeros((n, pad, 7, 2), dtype=torch.float32)], 1
                )
                out["torsion_mask"] = torch.cat(
                    [out["torsion_mask"], torch.zeros((pad, 7), dtype=torch.float32)]
                )

        return out
