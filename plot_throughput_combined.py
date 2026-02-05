#!/usr/bin/env python3
"""Plot memory bandwidth comparison between CUTLASS and TRT-LLM backends."""

import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV files
df_cutlass = pd.read_csv("ml3_moe_cutlass_out.csv")
df_trtllm = pd.read_csv("ml3_moe_out.csv")

# Filter CUTLASS data by cutlass_variant
df_cutlass_fp8 = df_cutlass[df_cutlass["cutlass_variant"] == "fp8"].copy()
df_cutlass_nvfp4 = df_cutlass[df_cutlass["cutlass_variant"] == "nvfp4"].copy()

# Filter TRT-LLM data by routine
df_trtllm_fp8 = df_trtllm[df_trtllm["routine"] == "trtllm_fp8_block_scale_moe"].copy()
df_trtllm_fp4 = df_trtllm[df_trtllm["routine"] == "trtllm_fp4_block_scale_moe"].copy()

# Extract num_tokens (column index 29)
num_tokens_cutlass_fp8 = df_cutlass_fp8.iloc[:, 29].astype(int)
num_tokens_cutlass_nvfp4 = df_cutlass_nvfp4.iloc[:, 29].astype(int)
num_tokens_trtllm_fp8 = df_trtllm_fp8.iloc[:, 29].astype(int)
num_tokens_trtllm_fp4 = df_trtllm_fp4.iloc[:, 29].astype(int)

# Filter to only include num_tokens <= 1024
mask_cutlass_fp8 = num_tokens_cutlass_fp8 <= 1024
mask_cutlass_nvfp4 = num_tokens_cutlass_nvfp4 <= 1024
mask_trtllm_fp8 = num_tokens_trtllm_fp8 <= 1024
mask_trtllm_fp4 = num_tokens_trtllm_fp4 <= 1024

# Create figure
fig, ax = plt.subplots(figsize=(12, 7))

# Colors for backends
color_cutlass = (118/255, 185/255, 0/255)  # green
color_trtllm = (93/255, 22/255, 130/255)   # purple

# CUTLASS FP8 - solid line with circles
ax.plot(num_tokens_cutlass_fp8[mask_cutlass_fp8], 
        df_cutlass_fp8["tb_per_sec"][mask_cutlass_fp8], 
        color=color_cutlass, marker="o", linestyle="-", 
        label="CUTLASS FP8")

# CUTLASS NVFP4 - dashed line with triangles
ax.plot(num_tokens_cutlass_nvfp4[mask_cutlass_nvfp4], 
        df_cutlass_nvfp4["tb_per_sec"][mask_cutlass_nvfp4], 
        color=color_cutlass, marker="^", linestyle="--", 
        label="CUTLASS NVFP4")

# TRT-LLM FP8 - solid line with squares
ax.plot(num_tokens_trtllm_fp8[mask_trtllm_fp8], 
        df_trtllm_fp8["tb_per_sec"][mask_trtllm_fp8], 
        color=color_trtllm, marker="s", linestyle="-", 
        label="TensorRT-LLM FP8")

# TRT-LLM FP4 - dashed line with diamonds
ax.plot(num_tokens_trtllm_fp4[mask_trtllm_fp4], 
        df_trtllm_fp4["tb_per_sec"][mask_trtllm_fp4], 
        color=color_trtllm, marker="D", linestyle="--", 
        label="TensorRT-LLM FP4")

ax.set_xlabel("num_tokens")
ax.set_ylabel("TB/sec (Memory Throughput)")
ax.set_xscale("log", base=2)
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)

plt.title("Memory Bandwidth: CUTLASS vs TensorRT-LLM (num_tokens <= 1024)")
plt.tight_layout()
plt.savefig("throughput_combined_plot.png", dpi=150)
plt.show()

print("Plot saved to throughput_combined_plot.png")
