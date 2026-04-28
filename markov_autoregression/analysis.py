"""
Shared analysis utilities for tetrapeptide and MD-CATH pipelines.
"""

import os

import mdtraj
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy

import deeptime.clustering
import deeptime.decomposition
import deeptime.markov
import deeptime.markov.msm


# ---------------------------------------------------------------------------
# Divergence metrics (JSD and KL)
# ---------------------------------------------------------------------------


def jsd_from_counts(h1, h2, base=2.0):
    """Squared Jensen-Shannon divergence from pre-computed histograms or probability vectors.

    With ``base=2`` (default) the result is bounded in [0, 1] (bits).
    Pass ``base=None`` for natural-log (nats).
    """
    return float(jensenshannon(np.asarray(h1).ravel(), np.asarray(h2).ravel(), base=base) ** 2)


def compute_jsd(a, b, bins=100, data_range=None, base=2.0):
    """Squared Jensen-Shannon divergence between two 1-D sample arrays."""
    a, b = np.asarray(a), np.asarray(b)
    if data_range is None:
        data_range = (min(float(a.min()), float(b.min())),
                      max(float(a.max()), float(b.max())))
    h1, _ = np.histogram(a, bins=bins, range=data_range)
    h2, _ = np.histogram(b, bins=bins, range=data_range)
    return jsd_from_counts(h1, h2, base=base)


def kl_from_counts(h1, h2, epsilon=1e-5):
    """Forward KL divergence D_KL(P||Q) from pre-computed counts or probability vectors."""
    p = np.asarray(h1, dtype=float).ravel()
    q = np.asarray(h2, dtype=float).ravel()
    p = p / p.sum()
    q = q / q.sum()
    q = np.where(q == 0, epsilon, q)
    q = q / q.sum()
    return float(entropy(p, q))


def compute_kl(a, b, bins=100, data_range=None, epsilon=1e-5):
    """Forward KL divergence D_KL(P||Q) between two 1-D sample arrays."""
    a, b = np.asarray(a), np.asarray(b)
    if data_range is None:
        data_range = (min(float(a.min()), float(b.min())),
                      max(float(a.max()), float(b.max())))
    h1, _ = np.histogram(a, bins=bins, range=data_range)
    h2, _ = np.histogram(b, bins=bins, range=data_range)
    return kl_from_counts(h1, h2, epsilon=epsilon)


# ---------------------------------------------------------------------------
# Featurization
# ---------------------------------------------------------------------------


class _TorsionFeatures:
    """Lightweight wrapper that mimics the pyemma featurizer ``describe()`` API."""

    def __init__(self, labels):
        self._labels = labels

    def describe(self):
        return list(self._labels)


def _compute_torsions(traj, sidechains=False):
    """Compute torsion angles and human-readable labels from an mdtraj trajectory."""
    angles_list = []
    labels = []

    for compute_fn, prefix in [
        (mdtraj.compute_phi, "PHI"),
        (mdtraj.compute_psi, "PSI"),
        (mdtraj.compute_omega, "OMEGA"),
    ]:
        indices, values = compute_fn(traj)
        for i in range(values.shape[1]):
            res = traj.topology.atom(indices[i, 1]).residue
            labels.append(f"{prefix} {res.index} {res.name}")
            angles_list.append(values[:, i])

    if sidechains:
        for compute_fn, prefix in [
            (mdtraj.compute_chi1, "CHI1"),
            (mdtraj.compute_chi2, "CHI2"),
            (mdtraj.compute_chi3, "CHI3"),
            (mdtraj.compute_chi4, "CHI4"),
        ]:
            indices, values = compute_fn(traj)
            for i in range(values.shape[1]):
                res = traj.topology.atom(indices[i, 1]).residue
                labels.append(f"{prefix} {res.index} {res.name}")
                angles_list.append(values[:, i])

    if not angles_list:
        return labels, np.empty((traj.n_frames, 0))
    return labels, np.column_stack(angles_list)


def get_featurized_traj(name, sidechains=False, cossin=True):
    """Load an XTC trajectory and compute torsion features.

    Returns ``(features, data)`` where *features* has a ``.describe()``
    method returning human-readable labels and *data* is a 2-D numpy array.
    """
    traj = mdtraj.load_xtc(name + ".xtc", top=name + ".pdb")
    labels, data = _compute_torsions(traj, sidechains)

    if cossin:
        cossin_data = np.empty((data.shape[0], data.shape[1] * 2))
        cossin_data[:, 0::2] = np.cos(data)
        cossin_data[:, 1::2] = np.sin(data)
        cossin_labels = []
        for label in labels:
            cossin_labels.append(f"COS({label})")
            cossin_labels.append(f"SIN({label})")
        return _TorsionFeatures(cossin_labels), cossin_data

    return _TorsionFeatures(labels), data


# ---------------------------------------------------------------------------
# TICA
# ---------------------------------------------------------------------------


def get_tica(traj_data, lag=1000, var_cutoff=0.95):
    """Fit TICA on *traj_data* and return ``(model, transformed)``."""
    tica = deeptime.decomposition.TICA(
        lagtime=lag, var_cutoff=var_cutoff, scaling="kinetic_map"
    )
    model = tica.fit(traj_data).fetch_model()
    transformed = model.transform(traj_data)
    return model, transformed


# ---------------------------------------------------------------------------
# K-means clustering
# ---------------------------------------------------------------------------


def get_kmeans(traj):
    """Cluster TICA-transformed data into 100 micro-states.

    Returns ``(model, [labels])`` where *labels* is a 1-D array of
    cluster assignments (wrapped in a list for backward compatibility).
    """
    km = deeptime.clustering.KMeans(
        n_clusters=100, max_iter=100, fixed_seed=137, n_jobs=1
    )
    model = km.fit(traj).fetch_model()
    labels = model.transform(traj)
    return model, [labels]


# ---------------------------------------------------------------------------
# MSM + PCCA
# ---------------------------------------------------------------------------


def get_msm(traj, lag=1000, nstates=10):
    """Estimate MSM → PCCA → coarse MSM.

    Returns ``(assignments, cmsm)`` where *assignments* maps every
    micro-state index to a metastable (PCCA) state.  Disconnected
    micro-states (pruned by the reversible MSM) are assigned to the
    most populated metastable state so that indexing with the full
    k-means label array never goes out of bounds.
    """
    counts = deeptime.markov.TransitionCountEstimator(
        lagtime=lag, count_mode="sliding"
    ).fit_fetch([traj])
    msm = deeptime.markov.msm.MaximumLikelihoodMSM(
        reversible=True
    ).fit_fetch(counts)

    pcca = msm.pcca(nstates)

    n_micro = int(traj.max()) + 1
    active = msm.state_symbols()
    if len(active) < n_micro:
        assignments = np.empty(n_micro, dtype=pcca.assignments.dtype)
        assignments[active] = pcca.assignments
        fallback = int(np.bincount(pcca.assignments).argmax())
        disconnected = np.setdiff1d(np.arange(n_micro), active)
        assignments[disconnected] = fallback
    else:
        assignments = pcca.assignments

    coarse_traj = assignments[traj]
    counts_c = deeptime.markov.TransitionCountEstimator(
        lagtime=lag, count_mode="sliding"
    ).fit_fetch([coarse_traj])
    cmsm = deeptime.markov.msm.MaximumLikelihoodMSM(
        allow_disconnected=True, reversible=True
    ).fit_fetch(counts_c)

    return assignments, cmsm


def discretize(traj, kmeans_model, assignments):
    """Map continuous TICA data → metastable-state assignments."""
    return assignments[kmeans_model.transform(traj)]


# ---------------------------------------------------------------------------
# Feature-based MSM helpers
# ---------------------------------------------------------------------------


def compute_feature(traj, feature, global_reference):
    """Return a (n_frames, 1) numpy array for the chosen observable."""
    if feature == "gr":
        return mdtraj.compute_rg(traj).reshape(-1, 1)
    if feature == "secondary":
        dssp = mdtraj.compute_dssp(traj)
        ss_codes = ["H", "G", "I", "E", "B"]
        frac = np.sum(np.isin(dssp, ss_codes), axis=1) / dssp.shape[1]
        return frac.reshape(-1, 1)
    if feature == "rmsd":
        return mdtraj.rmsd(traj, global_reference).reshape(-1, 1)
    raise ValueError(f"Unknown feature {feature!r}")


def fit_kmeans_on_reference(ref_trajs, features, n_states):
    """Fit k-means only on reference trajectories; return model and discrete refs."""
    global_ref = ref_trajs[0][0]

    ref_feat_mats = [
        np.hstack([compute_feature(t, f, global_ref) for f in features])
        for t in ref_trajs
    ]
    kmeans = deeptime.clustering.KMeans(
        n_clusters=n_states, max_iter=100, fixed_seed=2137, n_jobs=1
    ).fit(ref_feat_mats)
    disc_ref = [kmeans.transform(feat) for feat in ref_feat_mats]
    return kmeans, disc_ref, global_ref


def stationary_dist(discrete_trajs, lag):
    counts = deeptime.markov.TransitionCountEstimator(
        lagtime=lag, count_mode="sliding"
    ).fit_fetch(discrete_trajs)
    msm = deeptime.markov.msm.MaximumLikelihoodMSM(
        allow_disconnected=True, reversible=True
    ).fit_fetch(counts)
    return msm.stationary_distribution, msm.state_symbols()


# ---------------------------------------------------------------------------
# MD-CATH trajectory loading
# ---------------------------------------------------------------------------


def _load_mdcath_trajs(data_dir, pdb_dir, name, temp, gen_replicas, truncate):
    """Load and split MD-CATH replicas into (ref_traj, sampler_traj).

    When *gen_replicas* is ``None`` the sampler trajectory is loaded from
    *pdb_dir*; otherwise the specified replica indices are used as the
    sampler set and the remaining replicas become the reference.
    """
    def _load_replicas(indices):
        trajs = []
        top = f"{data_dir}/topology/{name}.pdb"
        for i in indices:
            xtc = f"{data_dir}/trajectory/{name}_{temp}_{i}.xtc"
            if os.path.exists(xtc):
                trajs.append(mdtraj.load_xtc(xtc, top=top))
        if not trajs:
            raise RuntimeError(f"No MD replicas found for {name} indices={indices}")
        return mdtraj.join(trajs) if len(trajs) > 1 else trajs[0]

    if gen_replicas is None:
        ref_traj = _load_replicas(range(5))
        samp_xtc = f"{pdb_dir}/{name}.xtc"
        samp_pdb = f"{pdb_dir}/{name}.pdb"
        if not os.path.exists(samp_xtc):
            raise RuntimeError(f"Sampler trajectory {samp_xtc} not found.")
        sampler_traj = mdtraj.load_xtc(samp_xtc, top=samp_pdb)
    else:
        gen_set = set(gen_replicas)
        if not gen_set.issubset({0, 1, 2, 3, 4}) or len(gen_set) in (0, 5):
            raise ValueError("gen_replicas must list 1-4 indices from 0-4")
        sampler_traj = _load_replicas(gen_set)
        ref_traj = _load_replicas([i for i in range(5) if i not in gen_set])

    if truncate is not None:
        sampler_traj = sampler_traj[:truncate]

    return ref_traj, sampler_traj


# ---------------------------------------------------------------------------
# Distribution comparison helpers
# ---------------------------------------------------------------------------


def compute_divergences(ref_values, sampler_values, bins=100, epsilon=1e-5):
    """Forward KL and squared JSD (base-e) between two 1-D sample arrays."""
    return {
        "forward_kl_divergence": compute_kl(ref_values, sampler_values, bins=bins, epsilon=epsilon),
        "jensen_shannon_divergence": compute_jsd(ref_values, sampler_values, bins=bins, base=None),
    }


def compute_gyration_radius(traj, random_frames=None):
    """Radius of gyration for (optionally sampled) frames."""
    if random_frames is not None and random_frames < traj.n_frames:
        np.random.seed(42)
        frame_indices = np.random.choice(
            traj.n_frames, size=random_frames, replace=False
        )
        traj = traj[frame_indices]
    return mdtraj.compute_rg(traj)


def compute_secondary_structure_fraction(traj, secondary_codes=None):
    """Fraction of residues in secondary structure per frame."""
    if secondary_codes is None:
        secondary_codes = ["H", "G", "I", "E", "B"]
    dssp = mdtraj.compute_dssp(traj)
    return np.sum(np.isin(dssp, secondary_codes), axis=1) / dssp.shape[1]


# ---------------------------------------------------------------------------
# MD-CATH comparison functions
# ---------------------------------------------------------------------------


def compare_gyration_radius_mdcath(
    data_dir, pdb_dir, name, temp=320, random_frames=None,
    truncate=None, gen_replicas=None,
):
    """Compare Rg distributions between MD replicas and a sampled trajectory."""
    ref_traj, sampler_traj = _load_mdcath_trajs(
        data_dir, pdb_dir, name, temp, gen_replicas, truncate,
    )

    ref_rg = compute_gyration_radius(ref_traj, random_frames=random_frames)
    sampler_rg = compute_gyration_radius(sampler_traj, random_frames=random_frames)

    return {
        "ref_gyration_radius_mean": ref_rg.mean(),
        "sampler_gyration_radius_mean": sampler_rg.mean(),
        "gyration_radius_difference": abs(ref_rg.mean() - sampler_rg.mean()),
        **compute_divergences(ref_rg, sampler_rg),
    }


def compare_secondary_structure_mdcath(
    data_dir, pdb_dir, name, temp=320, random_frames=None,
    truncate=None, gen_replicas=None,
):
    """Compare secondary-structure fraction distributions."""
    ref_traj, sampler_traj = _load_mdcath_trajs(
        data_dir, pdb_dir, name, temp, gen_replicas, truncate,
    )

    ref_fractions = compute_secondary_structure_fraction(ref_traj)
    sampler_fractions = compute_secondary_structure_fraction(sampler_traj)

    return {
        "ref_secondary_structure_fraction_mean": ref_fractions.mean(),
        "sampler_secondary_structure_fraction_mean": sampler_fractions.mean(),
        "mean_difference": abs(ref_fractions.mean() - sampler_fractions.mean()),
        **compute_divergences(ref_fractions, sampler_fractions),
    }
