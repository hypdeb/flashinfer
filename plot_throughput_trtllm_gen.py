#!/usr/bin/env python3
"""Plot compute and memory throughput from benchmark results."""

import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv("ml3_moe_out.csv")

# Filter data by routine
df_fp8 = df[df["routine"] == "trtllm_fp8_block_scale_moe"].copy()
df_fp4 = df[df["routine"] == "trtllm_fp4_block_scale_moe"].copy()

# Extract relevant columns (num_tokens appears multiple times, use iloc[:, 29])
num_tokens_fp8 = df_fp8.iloc[:, 29].astype(int)
num_tokens_fp4 = df_fp4.iloc[:, 29].astype(int)

# Create figure with dual y-axes
fig, ax1 = plt.subplots(figsize=(12, 7))

# Colors for the two metrics
color_tflops = (118/255, 185/255, 0/255)  # RGB 118/185/0 (green)
color_tb = (93/255, 22/255, 130/255)  # RGB 93/22/130 (purple)

# Plot TFLOPS on left y-axis
ax1.set_xlabel("num_tokens")
ax1.set_ylabel("TFLOPS (Compute Throughput)", color=color_tflops)
# FP8 - solid line with circles
line1 = ax1.plot(num_tokens_fp8, df_fp8["tflops"], color=color_tflops, 
                  marker="o", linestyle="-", label="TFLOPS (FP8)")
# FP4 - dashed line with triangles
line2 = ax1.plot(num_tokens_fp4, df_fp4["tflops"], color=color_tflops, 
                  marker="^", linestyle="--", label="TFLOPS (FP4)")
ax1.tick_params(axis="y", labelcolor=color_tflops)
ax1.set_xscale("log", base=2)

# Create second y-axis for memory throughput
ax2 = ax1.twinx()
ax2.set_ylabel("TB/sec (Memory Throughput)", color=color_tb)
# FP8 - solid line with squares
line3 = ax2.plot(num_tokens_fp8, df_fp8["tb_per_sec"], color=color_tb, 
                  marker="s", linestyle="-", label="TB/sec (FP8)")
# FP4 - dashed line with diamonds
line4 = ax2.plot(num_tokens_fp4, df_fp4["tb_per_sec"], color=color_tb, 
                  marker="D", linestyle="--", label="TB/sec (FP4)")
ax2.tick_params(axis="y", labelcolor=color_tb)

# Add title and legend
plt.title("Compute and Memory Throughput vs num_tokens")
lines = line1 + line2 + line3 + line4
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="upper left")

# Add grid
ax1.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("throughput_plot.png", dpi=150)
plt.show()

print("Plot saved to throughput_plot.png")
