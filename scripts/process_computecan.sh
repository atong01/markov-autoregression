#!/bin/bash
#SBATCH --job-name=proc-and-cluster
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --cpus-per-task=20
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --account=def-bengioy

source .env
source .venv/bin/activate

python -m scripts.prepare_data.prep_sims_mdcath_xtc \
  --split splits/mdCATH.txt \
  --sim_dir $DATA_PATH/md_cath_sims \
  --outdir $DATA_PATH/md_cath_processed \
  --num_workers $SLURM_CPUS_PER_TASK \
  --temps 450


python -m scripts.msm_clusters.create_msm_states_mdcath_xtc \
  --data_dir $DATA_PATH/md_cath_sims \
  --cluster_data_dir $DATA_PATH/md_cath_processed \
  --input_file splits/mdCATH.txt --msm_num_states 10 --temp 450 \
  --num_workers $SLURM_CPUS_PER_TASK

