#!/bin/bash
#SBATCH --job-name=mars-ar-ca.sh
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
# Full mdCATH AR training over Cartesian C-alpha coordinates instead of
# frames+torsions. The latent for each residue is its 3-vector CA position,
# so per-residue tokens = 3 (x, y, z) and AR generates them residue-major.
# The discretizer is calibrated per-channel from training data: 3 independent
# [μ-kσ, μ+kσ] ranges over the (x, y, z) marginals, each carved into num_bins.
# IPA is auto-disabled in the encoder under --euclidean (see CausalARModel),
# so the encoder is RoPE-MHA over the per-residue starting coords + sequence.
RUN_NAME="mdcath_ar_ca_v6"
WANDB_NAME="${RUN_NAME}"

WORKDIR_ROOT="/home/mila/d/danyal.rehman/scratch/mdcath/workdir"
export WANDB_DIR="${WORKDIR_ROOT}"
export WANDB_CACHE_DIR="${WORKDIR_ROOT}/.wandb_cache"
export WANDB_CONFIG_DIR="${WORKDIR_ROOT}/.wandb_config"
export TORCH_HOME="${WORKDIR_ROOT}/.torch"
export HF_HOME="${WORKDIR_ROOT}/.hf"

mkdir -p logs "${WORKDIR_ROOT}" "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}" \
         "${TORCH_HOME}" "${HF_HOME}"

# Hyperparameters mirror train_ar.sh except for the Euclidean-CA-specific
# additions:
#   --euclidean --ca_only       -> latent_dim=3 (one CA atom × 3 coords)
#   --s_translation 1.0         -> mild translation augmentation; widens bins
#                                  by ~5%, keep small to preserve precision
# num_bins kept at 8192: with ±4σ over a ~20Å protein extent, bin width is
# ~0.01Å — fine enough to faithfully represent CA dynamics. Calibration runs
# on 32 batches before fit() so all DDP ranks derive identical bounds from
# the same RNG-deterministic loader pass.
srun python -m scripts.train \
  --train_split splits/mdCATH_train.csv --val_split splits/mdCATH_val.csv \
  --data_dir /home/mila/d/danyal.rehman/scratch/mdcath/md_cath_processed \
  --model_dir "${WORKDIR_ROOT}" \
  --batch_size 1 --crop 256 --val_repeat 5 --epochs 1000 \
  --mdcath --ckpt_freq 25 --wandb \
  --run_name "${RUN_NAME}" --wandb_name "${WANDB_NAME}" \
  --msm_num_states 10 --clusters_per_batch 2 --samples_per_cluster 12 \
  --msm_lagtime 50 --data_temperature 450 \
  --euclidean --ca_only --s_translation 1.0 \
  --ar --num_bins 8192 --auto_discretize_range --discretize_std_k 4.0 \
  --calibration_batches 32 --label_smoothing 0.05 --bf16
