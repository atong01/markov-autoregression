# Full atom14 (frame-parameterized) sampler trajectories.
python -m scripts.analysis.analyze_mdcath --pdbdir workdir_out --mdcath_processed_dir /mnt/labs/data/tong/mdCATH/md_cath_processed_v2 --mddir /mnt/labs/data/tong/mdCATH/converted --num_workers 250 --xtc --truncate 500 --temp 450 --msm_lag 50

# Cα-only sampler trajectories (e.g. workdir_out_ar from a CA-only model).
# Disables secondary-structure metrics (DSSP requires backbone N/CA/C/O)
# and ΔGfold (FNC uses minimum heavy-atom distances). The MSM falls back
# to Rg-only. Computable on Cα: pairwise RMSD (4), RMSF (5), gyration
# radius (3), MSM (2) — 14 of 28 metrics; SS (6) and dG_fold (2) are NaN.
python -m scripts.analysis.analyze_mdcath --pdbdir workdir_out_ar --mdcath_processed_dir /mnt/labs/data/tong/mdCATH/md_cath_processed_v2 --mddir /mnt/labs/data/tong/mdCATH/converted --num_workers 250 --xtc --truncate 500 --temp 450 --msm_lag 50 --ca_only

