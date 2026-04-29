#!/bin/bash
#SBATCH --job-name=mars-fm-overfit
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

# Flow-matching counterpart of overfit_ar.sh: train on a single protein with
# the stochastic MSM sampler still active (4 transition pairs per step). The
# train loss is flow-matching MSE, so its scale is *not* directly comparable
# to AR cross-entropy — what matters is whether it drops monotonically and
# plateaus near the irreducible noise floor of the conditional distribution.
RUN_NAME="mdcath_fm_overfit_v1"
WANDB_NAME="${RUN_NAME}"
OVERFIT_PROTEIN="${OVERFIT_PROTEIN:-12asA00}"

WORKDIR_ROOT="/home/mila/d/danyal.rehman/scratch/mdcath/workdir"
export WANDB_DIR="${WORKDIR_ROOT}"
export WANDB_CACHE_DIR="${WORKDIR_ROOT}/.wandb_cache"
export WANDB_CONFIG_DIR="${WORKDIR_ROOT}/.wandb_config"
export TORCH_HOME="${WORKDIR_ROOT}/.torch"
export HF_HOME="${WORKDIR_ROOT}/.hf"
mkdir -p "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}" "${TORCH_HOME}" "${HF_HOME}"

FRESH="${FRESH:-1}"
if [[ "${FRESH}" == "1" ]]; then
  rm -rf "${WORKDIR_ROOT}/${RUN_NAME}"
  echo "Wiped ${WORKDIR_ROOT}/${RUN_NAME} for a fresh overfit run."
fi
mkdir -p "${WORKDIR_ROOT}"

mkdir -p logs splits/_overfit
SPLIT_PATH="splits/_overfit/single_${OVERFIT_PROTEIN}.csv"

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

# Hyperparameters mirror overfit_ar.sh except for the AR-specific knobs
# (--num_bins, --auto_discretize_range, --label_smoothing, etc) which the
# flow path doesn't use, and --ar is omitted so train.py instantiates
# MarSModule (flow matching) instead of MarSARModule.
srun python -m scripts.train \
  --train_split "${SPLIT_PATH}" --val_split "${SPLIT_PATH}" \
  --data_dir /home/mila/d/danyal.rehman/scratch/mdcath/md_cath_processed \
  --model_dir "${WORKDIR_ROOT}" \
  --batch_size 1 --crop 128 --val_repeat 1 --epochs 20000 \
  --mdcath --ckpt_freq 1000 --wandb \
  --run_name "${RUN_NAME}" --wandb_name "${WANDB_NAME}" \
  --msm_num_states 10 --clusters_per_batch 1 --samples_per_cluster 4 \
  --msm_lagtime 50 --data_temperature 450 \
  --num_workers 1 --lr 5e-4
