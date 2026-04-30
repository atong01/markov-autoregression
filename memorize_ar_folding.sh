#!/bin/bash
#SBATCH --job-name=memorize-ar-folding
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --array=0-3%1
#SBATCH --account=rrg-bengioy-ad
#SBATCH --gpus=h100:1
#SBATCH --open-mode=append           
#SBATCH --partition=gpubase_bygpu_b1

# configure .env with DATA_PATH and WORKDIR_ROOT

set -euo pipefail

source .env
source .venv/bin/activate

# Memorization test: the dataset is forced to return the SAME (x_t, x_{t+tau})
# pair on every call (numpy RNG reseeded per-idx in MarSDataset*.__getitem__).
# With clusters_per_batch=1, samples_per_cluster=1 and a single protein, every
# step shows the model the same single transition. CE should drive towards 0.
# If it stalls anywhere materially above 0, the AR machinery has a real bug.
RUN_NAME="mdcath_ar_folding_memorize"
WANDB_NAME="${RUN_NAME}"
OVERFIT_PROTEIN="${OVERFIT_PROTEIN:-12asA00}"

export WANDB_DIR="${WORKDIR_ROOT}"
export WANDB_CACHE_DIR="${WORKDIR_ROOT}/.wandb_cache"
export WANDB_CONFIG_DIR="${WORKDIR_ROOT}/.wandb_config"
export TORCH_HOME="${WORKDIR_ROOT}/.torch"
export HF_HOME="${WORKDIR_ROOT}/.hf"
mkdir -p "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}" "${TORCH_HOME}" "${HF_HOME}"

FRESH="${FRESH:-1}"
if [[ "${FRESH}" == "1" ]]; then
  rm -rf "${WORKDIR_ROOT}/${RUN_NAME}"
  echo "Wiped ${WORKDIR_ROOT}/${RUN_NAME} for a fresh memorize run."
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
    print(f'Memorize on {name} (seqlen={len(df.loc[name, "seqres"])})')
else:
    fallback = df.index[0]
    df.iloc[:1].to_csv('${SPLIT_PATH}')
    print(f'WARNING: {name!r} not in train CSV, using {fallback!r} instead')
PYEOF

# clusters_per_batch=1, samples_per_cluster=1 -> one transition pair per call.
# --deterministic_dataset reseeds numpy at the top of __getitem__, so the
# same idx always yields the same cluster picks, same in-cluster frame picks,
# and same crop offset. With len(dataset)=1, every step sees an identical
# pair. Watch train/loss in particular: it uses current (non-EMA) weights and
# is the cleanest signal of whether the model can memorize.
srun python -m scripts.train \
  --train_split "${SPLIT_PATH}" --val_split "${SPLIT_PATH}" \
  --data_dir $DATA_PATH/md_cath_processed \
  --model_dir "${WORKDIR_ROOT}" \
  --batch_size 1 --crop 128 --val_repeat 1 --epochs 20000 \
  --mdcath --ckpt_freq 1000 --wandb \
  --run_name "${RUN_NAME}" --wandb_name "${WANDB_NAME}" \
  --msm_num_states 10 --clusters_per_batch 2 --samples_per_cluster 1 \
  --msm_lagtime 50 --data_temperature 450 \
  --num_workers 1 --lr 5e-4 \
  --deterministic_dataset \
  --ar --num_bins 8192 --auto_discretize_range --discretize_std_k 4.5 \
  --calibration_batches 4 --label_smoothing 0.0 --bf16 \
  --euclidean --enable_folding
