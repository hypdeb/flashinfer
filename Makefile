bench-mla-decode:
	python benchmarks/flashinfer_benchmark.py \
	--routine BatchMLAPagedAttentionWrapper \
	--batch_size 1024 \
	--s_kv 8192 \
	--num_qo_heads 128 \
	--num_kv_heads 1 \
	--head_dim_ckv 512 \
	--head_dim_kpe 64 \
	--page_size 64 \
	--backends fa2 cutlass trtllm-native \
	--q_dtype bfloat16 \
	--kv_dtype bfloat16 \
	--s_qo 1 \
	--num_iters 500

bench-mla-prefill:
	python benchmarks/flashinfer_benchmark.py \
	--routine BatchPrefillWithRaggedKVCacheWrapper \
	--batch_size 2 \
	--s_kv 8192 \
	--s_qo 8192 \
	--num_qo_heads 128 \
	--num_kv_heads 128 \
	--head_dim_qk 192 \
	--head_dim_vo 128 \
	--page_size 64 \
	--backends fa2 cutlass cudnn trtllm-native \
	--q_dtype bfloat16 \
	--kv_dtype bfloat16 \
	--num_iters 100

bench-mm-fp4:
	python benchmarks/flashinfer_benchmark.py \
	--routine mm_fp4 \
	--m 4 \
	--n 1024 \
	--k 7168 \
	--out_dtype bfloat16 \
	--backends cudnn \
	--use_128x4_sf_layout \
	--use_nvfp4 \
	--autotune

bench-fp4-moe:
	python benchmarks/flashinfer_benchmark.py \
	--routine trtllm_fp4_block_scale_moe \
	--num_tokens 1 \
	--hidden_size 7168 \
	--intermediate_size 4096 \
	--num_experts 128 \
	--top_k 4 \
	--n_group 1 \
	--topk_group 1 \
	--routing_method llama4 \
	--num_iters 1 \
	-vv

bench-ml3-fp8-moe:
	python benchmarks/flashinfer_benchmark.py \
	--routine trtllm_fp8_block_scale_moe \
	--num_tokens 8192 \
	--hidden_size 7168 \
	--intermediate_size 4096 \
	--num_experts 128 \
	--top_k 4 \
	--routing_method renormalize \
	--use_shuffled_weight \
	--num_iters 100 \
	-vv

bench-fp8-moe:
	python benchmarks/flashinfer_benchmark.py \
	--routine trtllm_fp8_block_scale_moe \
	--num_tokens 8192 \
	--hidden_size 7168 \
	--intermediate_size 2048 \
	--num_experts 256 \
	--top_k 8 \
	--n_group 8 \
	--topk_group 4 \
	--routed_scaling_factor 2.5 \
	--use_routing_bias \
	--routing_method deepseek_v3 \
	--use_shuffled_weight \
	-vv

bench-moe-list:
	python benchmarks/flashinfer_benchmark.py \
	--testlist ml3_moe_bench.txt \
	--output_path ml3_moe_out.csv

bench-cutlass-moe-list:
	python benchmarks/flashinfer_benchmark.py \
	--testlist ml3_moe_cutlass.txt \
	--output_path ml3_moe_cutlass_out.csv

bench-old:
	python benchmarks/bench_trtllm_gen_mla.py

profile-and-stats:
	nsys profile -o report.nsys-rep -f true python benchmarks/flashinfer_benchmark.py --routine trtllm_fp4_block_scale_moe --num_tokens 1024 --hidden_size --intermediate_size 7168 --num_experts 128 --top_k 4 --routing_method renormalize --use_shuffled_weight --num_iters 1000 -vv --use_cuda_events
	nsys stats --report cuda_gpu_kern_sum --format column:colmax=0 report.nsys-rep | grep bmm