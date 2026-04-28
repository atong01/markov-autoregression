#!/bin/bash
#SBATCH --job-name=mars-ar.sh
#SBATCH --partition=main
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
RUN_NAME="mdcath_ar_v4_bs_1"
WANDB_NAME="${RUN_NAME}"

# Checkpoints + wandb cache live on $SCRATCH (not the home filesystem) for
# I/O speed and quota reasons. Only the small SLURM .out/.err logs stay in
# ./logs/ on the project filesystem.
WORKDIR_ROOT="/home/mila/d/danyal.rehman/scratch/mdcath/workdir"
export WANDB_DIR="${WORKDIR_ROOT}"
export WANDB_CACHE_DIR="${WORKDIR_ROOT}/.wandb_cache"
export WANDB_CONFIG_DIR="${WORKDIR_ROOT}/.wandb_config"
export TORCH_HOME="${WORKDIR_ROOT}/.torch"
export HF_HOME="${WORKDIR_ROOT}/.hf"

mkdir -p logs "${WORKDIR_ROOT}" "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}" \
         "${TORCH_HOME}" "${HF_HOME}"

# Each array task picks up from ${WORKDIR_ROOT}/${RUN_NAME}/last.ckpt
# automatically (saved every epoch by the rolling ModelCheckpoint in
# scripts/train.py).
srun python -m scripts.train \
  --train_split splits/mdCATH_train.csv --val_split splits/mdCATH_val.csv \
  --data_dir /home/mila/d/danyal.rehman/scratch/mdcath/md_cath_processed \
  --model_dir "${WORKDIR_ROOT}" \
  --batch_size 1 --crop 256 --val_repeat 5 --epochs 1000 \
  --mdcath --ckpt_freq 25 --wandb \
  --run_name "${RUN_NAME}" --wandb_name "${WANDB_NAME}" \
  --msm_num_states 10 --clusters_per_batch 2 --samples_per_cluster 12 \
  --msm_lagtime 50 --data_temperature 450 \
  --ar --num_bins 8192 --auto_discretize_range --discretize_std_k 4.0 \
  --calibration_batches 32 --label_smoothing 0.05 --bf16
