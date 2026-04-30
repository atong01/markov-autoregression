python -m scripts.prepare_data.prep_sims_mdcath_xtc \
  --split splits/mdCATH.txt \
  --sim_dir /mnt/labs/data/tong/mdCATH/converted \
  --outdir /mnt/labs/data/tong/mdCATH/md_cath_processed_v2 \
  --num_workers 128 \
  --temps 450
