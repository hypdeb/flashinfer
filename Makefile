bench:
	python3 benchmarks/flashinfer_benchmark.py --routine BatchMLAPagedAttentionWrapper --batch_size 1024 --s_kv 8192 --num_qo_heads 128 --num_kv_heads 1 --head_dim_ckv 512 --head_dim_kpe 64 --page_size 64 --backends trtllm-native --q_dtype fp8_e4m3 --kv_dtype fp8_e4m3 --s_qo 1 --num_iters 500

