"""
Combined MD-CATH analysis script.

Runs structural (RMSF, pairwise RMSD, JSD), observable (gyration radius,
secondary structure, feature-based MSM), and folding free-energy (ΔGfold)
analysis for each domain, then directly saves a summary CSV.
"""

import argparse
import csv
import functools
import operator
import os
import threading
import time
from multiprocessing import Manager, Pool

import numpy as np
import pandas as pd
import scipy.stats
import torch
import tqdm
from scipy.stats import gaussian_kde

import mars.analysis
from mars.vendored.openfold.residue_constants import restype_order
from mars.utils import atom14_to_pdb, set_seed

from bioemu_benchmarks.eval.folding_free_energies.fraction_native_contacts import (
    FNCSettings,
    get_fnc_from_samples_trajectory,
)

parser = argparse.ArgumentParser(
    description="Combined structural + observable analysis for MD-CATH domains."
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--pdbdir", type=str, required=True)
parser.add_argument("--mdcath_processed_dir", type=str, required=True)
parser.add_argument("--mddir", type=str, required=True)
parser.add_argument("--split", type=str, default="splits/mdCATH_test.csv")
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--temp", type=int, default=320)
parser.add_argument("--truncate", type=int, default=None)
parser.add_argument("--pdb_id", nargs="*", default=[])
parser.add_argument("--xtc", action="store_true")
parser.add_argument("--msm_lag", type=int, default=1)
parser.add_argument(
    "--feature_states",
    type=int,
    default=10,
    help="Number of k-means clusters for the feature-based MSM metric.",
)
parser.add_argument(
    "--gen_replicas",
    type=int,
    nargs="+",
    default=None,
    help=(
        "Index(es) of MD-CATH replicas (0-4) to treat as the *generated* ensemble. "
        "All remaining replicas become the reference MD ensemble. "
        "If omitted, the trajectory in --pdbdir is the generated ensemble."
    ),
)
parser.add_argument(
    "--gen_replicas_savedir",
    type=str,
    default="./workdir/MD_baselines_4",
)
args = parser.parse_args()
set_seed(args.seed)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def remove_hydrogens(traj):
    traj.atom_slice(
        [a.index for a in traj.top.atoms if a.element.symbol != "H"], True
    )


def get_rmsds(traj1, traj2, broadcast=False):
    n_atoms = traj1.shape[1]
    traj1 = traj1.reshape(traj1.shape[0], n_atoms * 3)
    traj2 = traj2.reshape(traj2.shape[0], n_atoms * 3)
    if broadcast:
        traj1, traj2 = traj1[:, None], traj2[None]
    return np.square(traj1 - traj2).sum(-1) ** 0.5 / n_atoms**0.5 * 10


def align_tops(top1, top2):
    names1 = [repr(a) for a in top1.atoms]
    names2 = [repr(a) for a in top2.atoms]
    intersection = [nam for nam in names1 if nam in names2]
    mask1 = [names1.index(nam) for nam in intersection]
    mask2 = [names2.index(nam) for nam in intersection]
    return mask1, mask2


def load_domain_sequences():
    """Load domain sequences from split CSV files and return {name: length}."""
    csv_files = [
        "splits/mdCATH_train.csv",
        "splits/mdCATH_val.csv",
        "splits/mdCATH_test.csv",
    ]
    domain_to_seq = {}
    for csv_file in csv_files:
        if not os.path.exists(csv_file):
            continue
        with open(csv_file, "r", newline="") as f:
            for row in csv.DictReader(f):
                if row["seqres"]:
                    domain_to_seq[row["name"]] = len(row["seqres"])
    return domain_to_seq


def load_md_replicas(mddir, name, temp, replica_indices=None):
    import mdtraj

    if replica_indices is None:
        replica_indices = range(5)
    trajs = []
    for i in replica_indices:
        xtc_path = f"{mddir}/trajectory/{name}_{temp}_{i}.xtc"
        pdb_path = f"{mddir}/topology/{name}.pdb"
        if os.path.exists(xtc_path):
            trajs.append(mdtraj.load_xtc(xtc_path, top=pdb_path))
    if not trajs:
        raise RuntimeError(
            f"No MD replicas found for {name} with indices={list(replica_indices)}"
        )
    return trajs


# ---------------------------------------------------------------------------
# Folding free-energy helpers (ΔGfold via fraction of native contacts)
# ---------------------------------------------------------------------------

K_BOLTZMANN = 0.001987203599772605  # kcal mol⁻¹ K⁻¹


def _foldedness_from_fnc(fnc, p_fold_thr, steepness):
    """Sigmoid foldedness from fraction of native contacts."""
    return 1 / (1 + np.exp(-2 * steepness * (fnc - p_fold_thr)))


def _compute_dG(fnc, temperature, p_fold_thr, steepness):
    """ΔG from sigmoid of fraction of native contacts."""
    p_fold = _foldedness_from_fnc(fnc, p_fold_thr, steepness).mean()
    p_fold = np.clip(p_fold, 1e-10, 1 - 1e-10)
    ratio = np.clip(p_fold / (1 - p_fold), 1e-10, 1e10)
    return -np.log(ratio) * K_BOLTZMANN * temperature


def _kde_minimum(q_vals):
    """Find the deepest minimum in the KDE of Q values (folding threshold)."""
    xs = np.linspace(0.0, 1.0, 1000)
    pdf = gaussian_kde(q_vals)(xs)
    minima = np.flatnonzero((pdf[:-2] > pdf[1:-1]) & (pdf[2:] > pdf[1:-1])) + 1
    return xs[minima[np.argmin(pdf[minima])]] if minima.size else 0.70


# ---------------------------------------------------------------------------
# Per-domain analysis functions
# ---------------------------------------------------------------------------


def run_folding_analysis(name, mdtraj):
    """Compute per-domain ΔGfold for generated and MD ensembles."""
    settings = FNCSettings()
    top_md = f"{args.mddir}/topology/{name}.pdb"

    if args.gen_replicas is not None:
        gen_set = set(args.gen_replicas)
        all_replicas = []
        for r in range(5):
            xtc = f"{args.mddir}/trajectory/{name}_{args.temp}_{r}.xtc"
            if os.path.exists(xtc):
                all_replicas.append(mdtraj.load_xtc(xtc, top=top_md))
        if len(all_replicas) < 5:
            return {}
        traj_samp = all_replicas[args.gen_replicas[0]]
        ref_trajs = [all_replicas[r] for r in range(5) if r not in gen_set]
    else:
        samp_xtc = f"{args.pdbdir}/{name}.xtc"
        samp_pdb = f"{args.pdbdir}/{name}.pdb"
        if not os.path.exists(samp_xtc):
            return {}
        traj_samp = mdtraj.load_xtc(samp_xtc, top=samp_pdb)
        ref_trajs = []
        for r in range(5):
            xtc = f"{args.mddir}/trajectory/{name}_{args.temp}_{r}.xtc"
            if os.path.exists(xtc):
                ref_trajs.append(mdtraj.load_xtc(xtc, top=top_md))
        if not ref_trajs:
            return {}

    if args.truncate:
        traj_samp = traj_samp[: args.truncate]

    xtc_ref = f"{args.mddir}/trajectory/{name}_320_0.xtc"
    if not os.path.exists(xtc_ref):
        return {}
    ref_one = mdtraj.load_xtc(xtc_ref, top=top_md)[0:1]

    q_samp = get_fnc_from_samples_trajectory(traj_samp, ref_one, **vars(settings))
    q_md = np.concatenate(
        [get_fnc_from_samples_trajectory(t, ref_one, **vars(settings)) for t in ref_trajs]
    )

    q_half = _kde_minimum(q_md)
    steepness = 10
    dg_md = _compute_dG(q_md, args.temp, p_fold_thr=q_half, steepness=steepness)
    dg_samp = _compute_dG(q_samp, args.temp, p_fold_thr=q_half, steepness=steepness)

    return {"dg_samp": float(dg_samp), "dg_md": float(dg_md)}


def run_structural_analysis(name, seqres, mdtraj):
    out = {}
    topfile = f"{args.mdcath_processed_dir}/{name}_analysis.pdb"
    xtc_files = [
        f"{args.mdcath_processed_dir}/{name}_{args.temp}_{i}_analysis.xtc"
        for i in range(5)
    ]
    pdb_files = [
        f"{args.mdcath_processed_dir}/{name}_{args.temp}_{i}_analysis.pdb"
        for i in range(5)
    ]

    if not all(os.path.exists(f) for f in xtc_files + [topfile]):
        seqres_tensor = torch.tensor([restype_order[c] for c in seqres])
        for i in range(5):
            atom14 = np.load(
                f"{args.mdcath_processed_dir}/{name}_{args.temp}_{i}.npy"
            )
            atom14_to_pdb(atom14, seqres_tensor[None][0].numpy(), pdb_files[i])
            traj = mdtraj.load(pdb_files[i])
            traj.superpose(traj)
            traj.save(xtc_files[i])
            traj[0].save(topfile)

    replicate_trajs = [mdtraj.load(xtc_files[i], top=topfile) for i in range(5)]

    if args.gen_replicas is not None:
        gen_set = set(args.gen_replicas)
        if not gen_set.issubset({0, 1, 2, 3, 4}):
            raise ValueError("--gen_replicas must be indices 0-4")
        if len(gen_set) == 0 or len(gen_set) == 5:
            raise ValueError("--gen_replicas must specify 1-4 replicas")
        aftraj_aa = functools.reduce(
            operator.add, [replicate_trajs[i] for i in gen_set]
        )
        traj_aa = functools.reduce(
            operator.add,
            [replicate_trajs[i] for i in range(5) if i not in gen_set],
        )
        ref_aa = traj_aa[0]
        use_af2 = False
    else:
        traj_aa = functools.reduce(operator.add, replicate_trajs)
        ref_aa = traj_aa[0]
        use_af2 = True

    if use_af2:
        if args.xtc:
            aftraj_aa = mdtraj.load(
                f"{args.pdbdir}/{name}.xtc", top=f"{args.pdbdir}/{name}.pdb"
            )
        else:
            aftraj_aa = mdtraj.load(f"{args.pdbdir}/{name}.pdb")

    if args.truncate:
        aftraj_aa = aftraj_aa[: args.truncate]

    remove_hydrogens(traj_aa)
    remove_hydrogens(ref_aa)
    remove_hydrogens(aftraj_aa)

    refmask, afmask = align_tops(traj_aa.top, aftraj_aa.top)
    traj_aa.atom_slice(refmask, True)
    ref_aa.atom_slice(refmask, True)
    aftraj_aa.atom_slice(afmask, True)

    np.random.seed(137)
    RAND1 = np.random.randint(0, traj_aa.n_frames, aftraj_aa.n_frames)
    RAND2 = np.random.randint(0, traj_aa.n_frames, aftraj_aa.n_frames)

    traj_aa.superpose(ref_aa)
    aftraj_aa.superpose(ref_aa)

    ca_mask = [a.index for a in traj_aa.top.atoms if a.name == "CA"]
    traj = traj_aa.atom_slice(ca_mask, False)
    ref = ref_aa.atom_slice(ca_mask, False)
    aftraj = aftraj_aa.atom_slice(ca_mask, False)

    traj.superpose(ref)
    aftraj.superpose(ref)

    out["af_rmsf"] = mdtraj.rmsf(aftraj_aa, ref_aa) * 10
    out["ref_rmsf"] = mdtraj.rmsf(traj_aa, ref_aa) * 10
    out["jsd_rmsf"] = mars.analysis.compute_jsd(out["ref_rmsf"], out["af_rmsf"], bins=50)

    ref_pw = get_rmsds(traj[RAND1].xyz, traj[RAND2].xyz, broadcast=True)
    af_pw = get_rmsds(aftraj.xyz, aftraj.xyz, broadcast=True)
    out["jsd_pairwise_rmsd"] = mars.analysis.compute_jsd(ref_pw.ravel(), af_pw.ravel(), bins=50)
    out["ref_mean_pairwise_rmsd"] = ref_pw.mean()
    out["af_mean_pairwise_rmsd"] = af_pw.mean()

    return out


def run_observable_analysis(name, mdtraj):
    out = {}
    set_seed(args.seed)

    gr_stats = mars.analysis.compare_gyration_radius_mdcath(
        args.mddir, args.pdbdir, name,
        temp=args.temp, truncate=args.truncate, gen_replicas=args.gen_replicas,
    )
    out["gyration_radius_difference"] = gr_stats["gyration_radius_difference"]
    out["gyration_radius_KL"] = gr_stats["forward_kl_divergence"]
    out["gyration_radius_JSD"] = gr_stats["jensen_shannon_divergence"]

    ss_stats = mars.analysis.compare_secondary_structure_mdcath(
        args.mddir, args.pdbdir, name,
        temp=args.temp, truncate=args.truncate, gen_replicas=args.gen_replicas,
    )
    out["ss_difference"] = ss_stats["mean_difference"]
    out["ss_KL"] = ss_stats["forward_kl_divergence"]
    out["ss_JSD"] = ss_stats["jensen_shannon_divergence"]

    if args.gen_replicas is not None:
        gen_set = set(args.gen_replicas)
        if not gen_set.issubset({0, 1, 2, 3, 4}) or len(gen_set) in (0, 5):
            raise ValueError("--gen_replicas must contain 1-4 indices from 0-4")
        samp_trajs = load_md_replicas(args.mddir, name, args.temp, gen_set)
        ref_trajs = load_md_replicas(
            args.mddir, name, args.temp, [i for i in range(5) if i not in gen_set]
        )
        traj_samp = functools.reduce(operator.add, samp_trajs)
    else:
        samp_xtc = f"{args.pdbdir}/{name}.xtc"
        samp_pdb = f"{args.pdbdir}/{name}.pdb"
        if not os.path.exists(samp_xtc):
            raise RuntimeError(f"Sampler trajectory {samp_xtc} not found.")
        traj_samp = mdtraj.load_xtc(samp_xtc, top=samp_pdb)
        ref_trajs = load_md_replicas(args.mddir, name, args.temp)

    if args.truncate:
        traj_samp = traj_samp[: args.truncate]

    FEATURES = [["gr", "secondary"]]
    out["feature_MSM"] = {}

    for feat in FEATURES:
        kmeans, disc_ref, _ = mars.analysis.fit_kmeans_on_reference(
            ref_trajs, feat, args.feature_states
        )
        disc_samp = kmeans.transform(
            np.hstack([
                mars.analysis.compute_feature(traj_samp, f, traj_samp[0])
                for f in feat
            ])
        )
        pi_ref, symbols_ref = mars.analysis.stationary_dist(
            disc_ref, lag=args.msm_lag
        )
        pi_samp, symbols_samp = mars.analysis.stationary_dist(
            [disc_samp], lag=args.msm_lag
        )

        pi_full_ref = np.zeros(args.feature_states)
        pi_full_ref[symbols_ref] = pi_ref
        pi_full_samp = np.zeros(args.feature_states)
        pi_full_samp[symbols_samp] = pi_samp

        feat_key = ",".join(feat)
        out["feature_MSM"][f"{feat_key}_JSD"] = mars.analysis.jsd_from_counts(
            pi_full_ref, pi_full_samp, base=None
        )
        out["feature_MSM"][f"{feat_key}_KL"] = mars.analysis.kl_from_counts(
            pi_full_ref, pi_full_samp
        )

    return out


# ---------------------------------------------------------------------------
# Combined entry point (called per domain in each worker)
# ---------------------------------------------------------------------------


def main(name, seqres):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    import mdtraj

    structural_out = run_structural_analysis(name, seqres, mdtraj)
    observable_out = run_observable_analysis(name, mdtraj)

    folding_out = run_folding_analysis(name, mdtraj)
    if folding_out:
        if observable_out is not None:
            observable_out.update(folding_out)
        else:
            observable_out = folding_out

    return name, structural_out, observable_out


def wrapper_main(task, finished):
    name, structural_out, observable_out = main(*task)
    finished.append(name)
    return name, structural_out, observable_out


def monitor_pending(all_names, finished, interval=120):
    while True:
        pending = set(all_names) - set(finished)
        print("Pending tasks:", pending)
        time.sleep(interval)


# ---------------------------------------------------------------------------
# Corpus-level aggregation (replaces the former complete_read_pkl.py)
# ---------------------------------------------------------------------------


def summarize_structural(data):
    """Aggregate per-domain structural results into corpus-level metrics."""
    df_list = []
    for out in data.values():
        pearson, _ = scipy.stats.pearsonr(out["af_rmsf"], out["ref_rmsf"])
        df_list.append({
            "md_pairwise": out.get("ref_mean_pairwise_rmsd", np.nan),
            "af_pairwise": out.get("af_mean_pairwise_rmsd", np.nan),
            "jsd_rmsf": out.get("jsd_rmsf", np.nan),
            "jsd_pairwise_rmsd": out.get("jsd_pairwise_rmsd", np.nan),
            "rmsf_pearson": pearson,
        })

    df = pd.DataFrame(df_list)
    result = {"count": len(df)}
    if len(df) == 0:
        return result

    result["MD_pairwise_RMSD"] = df["md_pairwise"].median()
    result["Pairwise_RMSD"] = df["af_pairwise"].median()
    result["Pairwise_RMSD_r"] = scipy.stats.pearsonr(
        df["md_pairwise"], df["af_pairwise"]
    )[0]

    all_ref_rmsf = np.concatenate([v["ref_rmsf"] for v in data.values()])
    all_af_rmsf = np.concatenate([v["af_rmsf"] for v in data.values()])
    result["MD_RMSF"] = np.median(all_ref_rmsf)
    result["RMSF"] = np.median(all_af_rmsf)
    result["Global_RMSF_r"] = scipy.stats.pearsonr(all_ref_rmsf, all_af_rmsf)[0]

    result["Per_target_RMSF_r"] = df["rmsf_pearson"].median()
    result["RMSF_JSD"] = df["jsd_rmsf"].mean()
    result["Pairwise_RMSD_JSD"] = df["jsd_pairwise_rmsd"].mean()
    return result


def summarize_observable(data):
    """Aggregate per-domain observable results into corpus-level metrics."""
    collectors = {
        "Gyration Radius Difference": [],
        "Gyration Radius KL": [],
        "Gyration Radius JSD": [],
        "Secondary Structure Difference": [],
        "Secondary Structure KL": [],
        "Secondary Structure JSD": [],
        "MSM JSD": [],
        "MSM KL": [],
    }
    KEY_MAP = [
        ("gyration_radius_difference", "Gyration Radius Difference"),
        ("gyration_radius_KL", "Gyration Radius KL"),
        ("gyration_radius_JSD", "Gyration Radius JSD"),
        ("ss_difference", "Secondary Structure Difference"),
        ("ss_KL", "Secondary Structure KL"),
        ("ss_JSD", "Secondary Structure JSD"),
    ]

    for results in data.values():
        for old, new in KEY_MAP:
            if old in results:
                collectors[new].append(results[old])
        feat_msm = results.get("feature_MSM", {})
        if "gr,secondary_JSD" in feat_msm:
            collectors["MSM JSD"].append(feat_msm["gr,secondary_JSD"])
        if "gr,secondary_KL" in feat_msm:
            collectors["MSM KL"].append(feat_msm["gr,secondary_KL"])

    result = {}
    for key, vals in collectors.items():
        result[f"{key}_mean"] = np.mean(vals) if vals else np.nan
        result[f"{key}_std"] = np.std(vals) if vals else np.nan

    dg_samp_vals, dg_md_vals = [], []
    for results in data.values():
        if "dg_samp" in results and "dg_md" in results:
            dg_samp_vals.append(results["dg_samp"])
            dg_md_vals.append(results["dg_md"])

    if dg_samp_vals:
        dg_s = np.array(dg_samp_vals)
        dg_m = np.array(dg_md_vals)
        result["dG_fold_MAE"] = float(np.mean(np.abs(dg_s - dg_m)))
        result["dG_fold_r"] = float(scipy.stats.pearsonr(dg_s, dg_m)[0])
    else:
        result["dG_fold_MAE"] = np.nan
        result["dG_fold_r"] = np.nan

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = pd.read_csv(args.split, index_col="name")

    if args.pdb_id:
        pdb_ids = list(args.pdb_id)
    else:
        pdb_ids = [
            nam.split(".")[0]
            for nam in os.listdir(args.pdbdir)
            if ".pdb" in nam and "_traj" not in nam
        ]
    pdb_ids = [nam for nam in pdb_ids if os.path.exists(f"{args.pdbdir}/{nam}.xtc")]
    pdb_ids = [nam for nam in pdb_ids if nam in df.index]

    for bad_id in ("1gkgA02", "1ia5A00", "3amrA00"):
        if bad_id in pdb_ids:
            pdb_ids.remove(bad_id)

    domain_to_seq = load_domain_sequences()
    print("Number of domains:", len(pdb_ids))
    print(
        "Average sequence length:",
        np.mean([domain_to_seq[name] for name in pdb_ids if name in domain_to_seq]),
    )

    basedir = args.pdbdir
    if args.gen_replicas is not None:
        basedir = args.gen_replicas_savedir

    csv_path = os.path.join(basedir, "analysis.csv")
    if args.truncate and args.truncate != 500:
        csv_path = os.path.join(basedir, f"analysis_trunc{args.truncate}.csv")

    tasks = [(id_, df.seqres.loc[id_]) for id_ in pdb_ids]

    manager = Manager()
    finished = manager.list()

    monitor_thread = threading.Thread(
        target=monitor_pending, args=(pdb_ids, finished, 60), daemon=True
    )
    monitor_thread.start()

    with Pool(args.num_workers) as pool:
        results = [
            pool.apply_async(wrapper_main, args=(task, finished)) for task in tasks
        ]
        for r in tqdm.tqdm(results):
            r.wait()

    structural_data = {}
    observable_data = {}
    for r in results:
        name, s_out, o_out = r.get()
        if s_out is not None:
            structural_data[name] = s_out
        if o_out is not None:
            observable_data[name] = o_out

    row = {}
    if structural_data:
        row.update(summarize_structural(structural_data))
    if observable_data:
        row.update(summarize_observable(observable_data))

    df_final = pd.DataFrame([row])
    df_final.to_csv(csv_path, float_format="%.3f", index=False)
    print(f"Analysis saved to {csv_path}")
