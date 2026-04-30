#!/bin/bash
#SBATCH --job-name=mars-mf-backbone.sh
#SBATCH --partition=long
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:l40s:2
#SBATCH --mem=128G
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

# ---- Run identity --------------------------------------------------------
# Full mdCATH training with the mean-flow (flow-map) objective on Cartesian
# backbone coordinates (N, CA, C, O). Per-residue latent has 4 atoms × 3
# coords = 12 channels. IPA is auto-disabled under --euclidean (the encoder
# reduces to RoPE-MHA over per-residue starting coords + sequence).
# Mean-flow target is u_tgt = (ε − x) − (t − r) · ∂_t u with t=0 → data,
# t=1 → noise; sampling integrates Euler from t=1 to t=0 (--mf_n_steps).
RUN_NAME="mdcath_mf_backbone_v1"
WANDB_NAME="${RUN_NAME}"

WORKDIR_ROOT="/home/mila/d/danyal.rehman/scratch/mdcath/workdir"
export WANDB_DIR="${WORKDIR_ROOT}"
export WANDB_CACHE_DIR="${WORKDIR_ROOT}/.wandb_cache"
export WANDB_CONFIG_DIR="${WORKDIR_ROOT}/.wandb_config"
export TORCH_HOME="${WORKDIR_ROOT}/.torch"
export HF_HOME="${WORKDIR_ROOT}/.hf"

mkdir -p logs "${WORKDIR_ROOT}" "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}" \
         "${TORCH_HOME}" "${HF_HOME}"

# Mean-flow + Euclidean-backbone specifics:
#   --euclidean (no --ca_only)  -> latent_dim=12 (N, CA, C, O × 3 coords)
#   --s_translation 1.0         -> mild translation augmentation
#   --mean_flow                 -> MarSMeanFlowModule (u(z, r, t))
#   --mf_neq_frac 0.25          -> 25% non-equal r,t pairs; rest boundary
#   --mf_n_steps 8              -> Euler integration steps at inference
srun python -m scripts.train \
  --train_split splits/mdCATH_train.csv --val_split splits/mdCATH_val.csv \
  --data_dir /home/mila/d/danyal.rehman/scratch/mdcath/md_cath_processed \
  --model_dir "${WORKDIR_ROOT}" \
  --batch_size 1 --crop 256 --val_repeat 5 --epochs 1000 \
  --mdcath --ckpt_freq 25 --wandb \
  --run_name "${RUN_NAME}" --wandb_name "${WANDB_NAME}" \
  --msm_num_states 10 --clusters_per_batch 2 --samples_per_cluster 12 \
  --msm_lagtime 50 --data_temperature 450 \
  --euclidean --s_translation 1.0 \
  --mean_flow --mf_neq_frac 0.25 --mf_n_steps 8 --bf16
