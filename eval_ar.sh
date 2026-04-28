#!/bin/bash
#SBATCH --job-name=mars-ar-eval
#SBATCH --partition=short-unkillable
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=48G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# ---- Run identity --------------------------------------------------------
# Checkpoints + sample outputs + caches all live on $SCRATCH; only the small
# SLURM .out/.err files stay in ./logs/.
RUN_NAME="mdcath_ar_v0"
WORKDIR_ROOT="/home/mila/d/danyal.rehman/scratch/mdcath/workdir"
CKPT_PATH="${WORKDIR_ROOT}/${RUN_NAME}/last.ckpt"
OUT_DIR="/home/mila/d/danyal.rehman/scratch/mdcath/eval_out/${RUN_NAME}"
export WANDB_DIR="${WORKDIR_ROOT}"
export WANDB_CACHE_DIR="${WORKDIR_ROOT}/.wandb_cache"
export WANDB_CONFIG_DIR="${WORKDIR_ROOT}/.wandb_config"
export TORCH_HOME="${WORKDIR_ROOT}/.torch"
export HF_HOME="${WORKDIR_ROOT}/.hf"
mkdir -p "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}" "${TORCH_HOME}" "${HF_HOME}"

# ---- Note ---------------------------------------------------------------
# Generation only needs the processed npy files (load_starting_structure
# reads ${data_dir}/{name}_${temp}_0.npy), so we point --data_dir at the
# processed dir. The downstream analysis script (analyze_mdcath.py) DOES
# need the raw mdCATH H5 trajectories — it's not run here.

mkdir -p logs "${OUT_DIR}"

# Tree sampling, AR-only. AR per-call cost is much higher than flow
# (one forward pass per token × L*21 tokens), so we use smaller defaults
# than the README's flow recipe — tune up if compute permits.
srun python -m scripts.generate \
  --ar \
  --mars_ckpt "${CKPT_PATH}" \
  --data_dir /home/mila/d/danyal.rehman/scratch/mdcath/md_cath_processed \
  --split splits/mdCATH_test.csv \
  --out_dir "${OUT_DIR}" \
  --mdcath --temp 450 \
  --calls_mars 50 --tree \
  --max_mars_samples 100 --tree_parallel_chunk 25 \
  --skip_existing
