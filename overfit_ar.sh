#!/bin/bash
#SBATCH --job-name=mars-ar-overfit
#SBATCH --partition=main
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=24G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# ---- Run identity --------------------------------------------------------
# A pipeline sanity check: train on a single protein and watch the loss
# drop substantially over many epochs. If the loss stays near log(num_bins)
# something is wrong end-to-end.
RUN_NAME="mdcath_ar_overfit_v2"
WANDB_NAME="${RUN_NAME}"
OVERFIT_PROTEIN="${OVERFIT_PROTEIN:-12asA00}"

# Checkpoints + wandb cache live on $SCRATCH (not the home filesystem) for
# I/O speed and quota reasons. Only the small SLURM .out/.err logs stay in
# ./logs/ on the project filesystem.
WORKDIR_ROOT="/home/mila/d/danyal.rehman/scratch/mdcath/workdir"
export WANDB_DIR="${WORKDIR_ROOT}"
export WANDB_CACHE_DIR="${WORKDIR_ROOT}/.wandb_cache"
export WANDB_CONFIG_DIR="${WORKDIR_ROOT}/.wandb_config"
export TORCH_HOME="${WORKDIR_ROOT}/.torch"
export HF_HOME="${WORKDIR_ROOT}/.hf"
mkdir -p "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}" "${TORCH_HOME}" "${HF_HOME}"

# Always start fresh so the loss curve is interpretable from random init.
# Lightning auto-resumes from last.ckpt otherwise; a previously-finished run
# will just hit max_epochs and exit. Override with FRESH=0 to opt out.
FRESH="${FRESH:-1}"
if [[ "${FRESH}" == "1" ]]; then
  rm -rf "${WORKDIR_ROOT}/${RUN_NAME}"
  echo "Wiped ${WORKDIR_ROOT}/${RUN_NAME} for a fresh overfit run."
fi
mkdir -p "${WORKDIR_ROOT}"

mkdir -p logs splits/_overfit
SPLIT_PATH="splits/_overfit/single_${OVERFIT_PROTEIN}.csv"

# Build a one-protein split from the train CSV (idempotent; falls back
# to the first row if the requested name isn't present).
python - <<PYEOF
import pandas as pd
df = pd.read_csv('splits/mdCATH_train.csv', index_col='name')
name = '${OVERFIT_PROTEIN}'
if name in df.index:
    df.loc[[name]].to_csv('${SPLIT_PATH}')
    print(f'Overfit on {name} (seqlen={len(df.loc[name, "seqres"])})')
else:
    fallback = df.index[0]
    df.iloc[:1].to_csv('${SPLIT_PATH}')
    print(f'WARNING: {name!r} not in train CSV, using {fallback!r} instead')
PYEOF

# Smaller crop, tiny effective batch (clusters_per_batch=1, samples_per_cluster=4
# → 4 transition pairs per gradient step), no label smoothing so the model
# can drive the loss as low as the data permits.
srun python -m scripts.train \
  --train_split "${SPLIT_PATH}" --val_split "${SPLIT_PATH}" \
  --data_dir /home/mila/d/danyal.rehman/scratch/mdcath/md_cath_processed \
  --model_dir "${WORKDIR_ROOT}" \
  --batch_size 1 --crop 128 --val_repeat 1 --epochs 20000 \
  --mdcath --ckpt_freq 1000 --wandb \
  --run_name "${RUN_NAME}" --wandb_name "${WANDB_NAME}" \
  --msm_num_states 10 --clusters_per_batch 1 --samples_per_cluster 4 \
  --msm_lagtime 50 --data_temperature 450 \
  --num_workers 1 --lr 5e-4 \
  --ar --num_bins 16384 --auto_discretize_range --discretize_std_k 4.5 \
  --calibration_batches 4 --label_smoothing 0.0 --bf16
