python -m scripts.generate --mars_ckpt workdir/mdcath/epoch\=874-step\=470750.ckpt --data_dir /mnt/labs/data/tong/mdCATH/md_cath_processed --split splits/mdCATH_test.csv --out_dir workdir_out --mdcath --temp 450 --calls_mars 200 --tree --max_mars_samples 500 --tree_parallel_chunk 100 --skip_existing

