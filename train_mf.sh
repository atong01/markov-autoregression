#!/bin/bash
#SBATCH --job-name=mars-mf.sh
#SBATCH --partition=long
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:l40s:2
#SBATCH --mem=48G
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

# ---- Run identity --------------------------------------------------------
# Full mdCATH training with the mean-flow (flow-map) objective in place of
# conventional flow matching. The model is u_θ(z, r, t): the average drift
# from time r to t along a linear interpolant z = t·ε + (1-t)·x with the
# reversed time convention (t=0 → data, t=1 → noise). Training target is the
# stop-grad mean-flow identity:
#     u_tgt = (ε − x) − (t − r) · ∂_t u
# computed via torch.func.jvp at (z, r, t) with tangent (v, 0, 1).
# Sampling at inference is multi-step Euler from t=1 down to t=0
# (--mf_n_steps controls the number of steps; default 8).
RUN_NAME="mdcath_mf_v1"
WANDB_NAME="${RUN_NAME}"

WORKDIR_ROOT="/home/mila/d/danyal.rehman/scratch/mdcath/workdir"
export WANDB_DIR="${WORKDIR_ROOT}"
export WANDB_CACHE_DIR="${WORKDIR_ROOT}/.wandb_cache"
export WANDB_CONFIG_DIR="${WORKDIR_ROOT}/.wandb_config"
export TORCH_HOME="${WORKDIR_ROOT}/.torch"
export HF_HOME="${WORKDIR_ROOT}/.hf"

mkdir -p logs "${WORKDIR_ROOT}" "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}" \
         "${TORCH_HOME}" "${HF_HOME}"

# Hyperparameters mirror train_ar.sh's data/MSM setup. Mean-flow specifics:
#   --mean_flow              -> dispatch MarSMeanFlowModule (u(z, r, t))
#   --mf_neq_frac 0.25       -> 25% of samples have r ≠ t (flow-map regime),
#                                75% collapse to r = t (boundary regression
#                                that pins u(z, t, t) = ε − x)
#   --mf_uniform_t           -> default; t ~ U(0,1), r = u·t with u ~ U(0,1)
#   --mf_n_steps 8           -> Euler integration steps at inference
srun python -m scripts.train \
  --train_split splits/mdCATH_train.csv --val_split splits/mdCATH_val.csv \
  --data_dir /home/mila/d/danyal.rehman/scratch/mdcath/md_cath_processed \
  --model_dir "${WORKDIR_ROOT}" \
  --batch_size 1 --crop 256 --val_repeat 5 --epochs 1000 \
  --mdcath --ckpt_freq 25 --wandb \
  --run_name "${RUN_NAME}" --wandb_name "${WANDB_NAME}" \
  --msm_num_states 10 --clusters_per_batch 2 --samples_per_cluster 12 \
  --msm_lagtime 50 --data_temperature 450 \
  --mean_flow --mf_neq_frac 0.25 --mf_n_steps 8 --bf16
