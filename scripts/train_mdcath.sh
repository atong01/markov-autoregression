#!/bin/bash
#SBATCH --job-name=mdcath
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/mdcath_%j.out
#SBATCH --error=logs/mdcath_%j.err

echo "==========================================="
echo "Job started: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU devices: $CUDA_VISIBLE_DEVICES"
echo "==========================================="

cd /mnt/labs/home/atong/markov-autoregression

eval "$(micromamba shell hook --shell bash)"
micromamba activate mars

python -m scripts.train \
    --train_split splits/mdCATH_train.csv \
    --val_split splits/mdCATH_val.csv \
    --data_dir /mnt/labs/data/tong/mdCATH/md_cath_processed \
    --model_dir ./workdir \
    --batch_size 8 \
    --crop 256 \
    --val_repeat 5 \
    --epochs 1000 \
    --mdcath \
    --ckpt_freq 25 \
    --wandb \
    --run_name mdcath \
    --msm_num_states 10 \
    --clusters_per_batch 2 \
    --samples_per_cluster 12 \
    --msm_lagtime 50 \
    --data_temperature 450

echo "==========================================="
echo "Job completed: $(date)"
echo "==========================================="
