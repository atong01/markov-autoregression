
python -m scripts.generate --mars_ckpt \
epoch\=99-step\=144500.ckpt --data_dir \
/mnt/labs/data/tong/mdCATH/md_cath_processed_v2 --split \
splits/mdCATH_test.csv --out_dir workdir_out_ar --mdcath --temp 450 \
--calls_mars 200 --tree --max_mars_samples 500 --tree_parallel_chunk 100 \
--skip_existing --ar
