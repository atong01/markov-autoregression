"""
Tetrapeptide (4AA) analysis script.

Computes per-peptide torsion JSD, TICA JSD, MSM state JSD, macrostate MAE,
and decorrelation curves, then aggregates results into a summary CSV.
"""

import argparse
import os
from multiprocessing import Pool

import deeptime.markov
import deeptime.markov.msm
import deeptime.plots
import deeptime.util
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from scipy.spatial.distance import jensenshannon
from statsmodels.tsa.stattools import acovf

import mars.analysis
from mars.utils import set_seed

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Tetrapeptide (4AA) analysis.")
parser.add_argument("--mddir", type=str, required=True)
parser.add_argument("--pdbdir", type=str, required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--plot", action="store_true")
parser.add_argument("--pdb_id", nargs="*", default=[])
parser.add_argument("--truncate", type=int, default=None)
parser.add_argument("--msm_lag", type=int, default=10)
parser.add_argument("--n_lag_traj", type=int, default=1000)
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--numstates_tica", type=int, default=10)
args = parser.parse_args()

COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
kB = 0.0019872041  # kcal/(mol·K)
TEMPERATURE = 350.0
kBT = kB * TEMPERATURE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _circular_acovf(signal, nlag):
    """Normalized autocovariance of an angular signal via sin/cos decomposition."""
    autocorr = (
        acovf(np.sin(signal), demean=False, adjusted=True, nlag=nlag)
        + acovf(np.cos(signal), demean=False, adjusted=True, nlag=nlag)
    )
    baseline = np.sin(signal).mean() ** 2 + np.cos(signal).mean() ** 2
    return (autocorr.astype(np.float16) - baseline) / (1 - baseline)


def _is_backbone(feat: str) -> bool:
    f = feat.lower()
    return any(tok in f for tok in ("phi", "psi", "omega")) and "|" not in feat


def _is_sidechain(feat: str) -> bool:
    f = feat.lower()
    return "chi" in f and "|" not in feat


# ---------------------------------------------------------------------------
# Per-peptide analysis
# ---------------------------------------------------------------------------


def analyze_peptide(name):
    """Run all analyses for a single peptide and return (name, results_dict)."""
    out = {}
    fig, axs = plt.subplots(4, 4, figsize=(20, 20))
    set_seed(args.seed)

    feats, traj = mars.analysis.get_featurized_traj(
        f"{args.pdbdir}/{name}", sidechains=True, cossin=False
    )
    if args.truncate:
        traj = traj[: args.truncate]
    feats, ref = mars.analysis.get_featurized_traj(
        f"{args.mddir}/{name}/{name}", sidechains=True, cossin=False
    )
    feat_names = feats.describe()

    # -- Torsion JSD --
    out["JSD"] = {}
    for i, feat in enumerate(feat_names):
        ref_p = np.histogram(ref[:, i], range=(-np.pi, np.pi), bins=100)[0]
        traj_p = np.histogram(traj[:, i], range=(-np.pi, np.pi), bins=100)[0]
        out["JSD"][feat] = jensenshannon(ref_p, traj_p)

    # -- Torsion decorrelation (MD reference) --
    out["md_decorrelation"] = {}
    for i, feat in enumerate(feat_names):
        curve = _circular_acovf(ref[:, i], nlag=100000)
        out["md_decorrelation"][feat] = curve
        ax = axs[0, 1] if ("PHI" in feat or "PSI" in feat) else axs[0, 2]
        ax.plot(1 + np.arange(len(curve)), curve, color=COLORS[i % len(COLORS)])

    axs[0, 1].set_title("Backbone decorrelation (MD)")
    axs[0, 2].set_title("Sidechain decorrelation (MD)")
    axs[0, 1].set_xscale("log")
    axs[0, 2].set_xscale("log")

    # -- Torsion decorrelation (generated) --
    out["our_decorrelation"] = {}
    for i, feat in enumerate(feat_names):
        curve = _circular_acovf(traj[:, i], nlag=args.n_lag_traj)
        out["our_decorrelation"][feat] = curve
        ax = axs[1, 1] if ("PHI" in feat or "PSI" in feat) else axs[1, 2]
        ax.plot(1 + np.arange(len(curve)), curve, color=COLORS[i % len(COLORS)])

    axs[1, 1].set_title("Backbone decorrelation (ours)")
    axs[1, 2].set_title("Sidechain decorrelation (ours)")
    axs[1, 1].set_xscale("log")
    axs[1, 2].set_xscale("log")

    # -- TICA JSD --
    feats, traj = mars.analysis.get_featurized_traj(
        f"{args.pdbdir}/{name}", sidechains=True, cossin=True
    )
    if args.truncate:
        traj = traj[: args.truncate]
    feats, ref = mars.analysis.get_featurized_traj(
        f"{args.mddir}/{name}/{name}", sidechains=True, cossin=True
    )

    tica_model, _ = mars.analysis.get_tica(ref)
    ref_tica = tica_model.transform(ref)
    traj_tica = tica_model.transform(traj)

    t0_min = min(ref_tica[:, 0].min(), traj_tica[:, 0].min())
    t0_max = max(ref_tica[:, 0].max(), traj_tica[:, 0].max())
    t1_min = min(ref_tica[:, 1].min(), traj_tica[:, 1].min())
    t1_max = max(ref_tica[:, 1].max(), traj_tica[:, 1].max())

    ref_p = np.histogram(ref_tica[:, 0], range=(t0_min, t0_max), bins=100)[0]
    traj_p = np.histogram(traj_tica[:, 0], range=(t0_min, t0_max), bins=100)[0]
    out["JSD"]["TICA-0"] = jensenshannon(ref_p, traj_p)

    ref_p = np.histogram2d(
        *ref_tica[:, :2].T, range=((t0_min, t0_max), (t1_min, t1_max)), bins=50
    )[0]
    traj_p = np.histogram2d(
        *traj_tica[:, :2].T, range=((t0_min, t0_max), (t1_min, t1_max)), bins=50
    )[0]
    out["JSD"]["TICA-0,1"] = jensenshannon(ref_p.flatten(), traj_p.flatten())

    # -- TICA FES plots --
    if args.plot:
        energies_ref = deeptime.util.energy2d(*ref_tica[::100, :2].T)
        deeptime.plots.plot_energy2d(energies_ref, ax=axs[2, 0], cbar=False)
        energies_traj = deeptime.util.energy2d(*traj_tica[:, :2].T)
        deeptime.plots.plot_energy2d(energies_traj, ax=axs[2, 1], cbar=False)
        axs[2, 0].set_title("TICA FES (MD)")
        axs[2, 1].set_title("TICA FES (ours)")

    # -- TICA decorrelation --
    nlag_md = 100000 if not args.truncate else int(args.truncate / 10)
    autocorr_md = acovf(ref_tica[:, 0], nlag=nlag_md, adjusted=True, demean=False)
    out["md_decorrelation"]["tica"] = autocorr_md.astype(np.float16)

    autocorr_gen = acovf(
        traj_tica[:, 0], nlag=args.n_lag_traj, adjusted=True, demean=False
    )
    out["our_decorrelation"]["tica"] = autocorr_gen.astype(np.float16)

    if args.plot:
        axs[0, 3].plot(autocorr_md)
        axs[0, 3].set_title("MD TICA")
        axs[1, 3].plot(autocorr_gen)
        axs[1, 3].set_title("Traj TICA")

    # -- MSM metrics (macrostate MAE, stationary distributions) --
    kmeans_model, ref_kmeans = mars.analysis.get_kmeans(tica_model.transform(ref))
    ref_kmeans = ref_kmeans[0]
    assignments, cmsm = mars.analysis.get_msm(
        ref_kmeans, nstates=args.numstates_tica, lag=100 * args.msm_lag
    )

    traj_discrete = mars.analysis.discretize(
        tica_model.transform(traj), kmeans_model, assignments
    )
    ref_discrete = mars.analysis.discretize(
        tica_model.transform(ref), kmeans_model, assignments
    )

    n_states = args.numstates_tica
    traj_meta = (traj_discrete == np.arange(n_states)[:, None]).mean(1)
    ref_meta = (ref_discrete == np.arange(n_states)[:, None]).mean(1)

    G_ref = -kBT * np.log(np.maximum(ref_meta, 1e-4))
    G_traj = -kBT * np.log(np.maximum(traj_meta, 1e-4))
    out["mMAE"] = np.mean(np.abs(G_traj - G_ref))

    msm_pi = np.zeros(n_states)
    msm_pi[cmsm.state_symbols()] = cmsm.stationary_distribution
    out["msm_pi"] = msm_pi

    traj_counts = deeptime.markov.TransitionCountEstimator(
        lagtime=args.msm_lag, count_mode="sliding"
    ).fit_fetch([traj_discrete])
    traj_cmsm = deeptime.markov.msm.MaximumLikelihoodMSM(
        allow_disconnected=True, reversible=True
    ).fit_fetch(traj_counts)
    traj_pi = np.zeros(n_states)
    traj_pi[traj_cmsm.state_symbols()] = traj_cmsm.stationary_distribution
    out["traj_pi"] = traj_pi

    if args.plot:
        fig.savefig(f'{args.pdbdir}/{name}_{args.pdbdir.split("/")[-1]}.pdf')
    plt.close(fig)

    return name, out


# ---------------------------------------------------------------------------
# Corpus-level aggregation (replaces the former read_pkl.py)
# ---------------------------------------------------------------------------


def summarize(data):
    """Aggregate per-peptide results into corpus-level mean/std statistics."""
    collectors = {
        "Torsions (bb)": [],
        "Torsions (sc)": [],
        "Torsions (all)": [],
        "TICA-0": [],
        "TICA-0,1 joint": [],
        "MSM states": [],
        "Macrostate MAE": [],
    }

    for results in data.values():
        jsd = results.get("JSD", {})
        for feat, value in jsd.items():
            if _is_backbone(feat):
                collectors["Torsions (bb)"].append(value)
                collectors["Torsions (all)"].append(value)
            elif _is_sidechain(feat):
                collectors["Torsions (sc)"].append(value)
                collectors["Torsions (all)"].append(value)

        if "TICA-0" in jsd:
            collectors["TICA-0"].append(jsd["TICA-0"])
        if "TICA-0,1" in jsd:
            collectors["TICA-0,1 joint"].append(jsd["TICA-0,1"])

        if "msm_pi" in results and "traj_pi" in results:
            collectors["MSM states"].append(
                jensenshannon(np.array(results["msm_pi"]), np.array(results["traj_pi"]))
            )

        if "mMAE" in results:
            collectors["Macrostate MAE"].append(results["mMAE"])

    return {
        key: [np.mean(vals), np.std(vals)] for key, vals in collectors.items()
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if args.pdb_id:
        pdb_ids = args.pdb_id
    else:
        pdb_ids = [
            f.split(".")[0]
            for f in os.listdir(args.pdbdir)
            if ".pdb" in f and "_traj" not in f
        ]
    pdb_ids = [p for p in pdb_ids if os.path.exists(f"{args.pdbdir}/{p}.xtc")]
    print("Number of trajectories:", len(pdb_ids))

    if args.num_workers > 1:
        with Pool(args.num_workers) as pool:
            results = dict(
                tqdm.tqdm(pool.imap(analyze_peptide, pdb_ids), total=len(pdb_ids))
            )
    else:
        results = dict(
            tqdm.tqdm(map(analyze_peptide, pdb_ids), total=len(pdb_ids))
        )

    summary_df = pd.DataFrame(summarize(results), index=["Mean", "Std"])
    csv_path = os.path.join(args.pdbdir, "analysis.csv")
    summary_df.to_csv(csv_path, float_format="%.4f")
    print(summary_df)
    print(f"Analysis saved to {csv_path}")
