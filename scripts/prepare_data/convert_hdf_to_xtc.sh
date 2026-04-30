#!/bin/bash
#SBATCH --job-name=convert_hdf_to_xtc
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --cpus-per-task=20
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --account=def-bengioy

INPUT_DIR=${1:?Usage: sbatch convert_hdf_to_xtc.slurm <input_dir> <out_dir>}
OUT_DIR=${2:?Usage: sbatch convert_hdf_to_xtc.slurm <input_dir> <out_dir>}

mkdir -p logs

python convert_hdf_to_xtc.py \
    "$INPUT_DIR" \
    "$OUT_DIR" \
    --num_workers "$SLURM_CPUS_PER_TASK"
