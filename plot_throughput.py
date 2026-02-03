#!/usr/bin/env python3
"""Plot compute and memory throughput from benchmark results."""

import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv("ml3_out.csv")

# Extract relevant columns
# num_tokens appears multiple times in the CSV, use the first one (column index 29)
num_tokens = df.iloc[:, 29].astype(int)
tflops = df["tflops"]
tb_per_sec = df["tb_per_sec"]

# Create figure with dual y-axes
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot tflops on left y-axis
color1 = (118/255, 185/255, 0/255)  # RGB 118/185/0 (green)
ax1.set_xlabel("num_tokens")
ax1.set_ylabel("TFLOPS (Compute Throughput)", color=color1)
line1 = ax1.plot(num_tokens, tflops, color=color1, marker="o", label="TFLOPS")
ax1.tick_params(axis="y", labelcolor=color1)
ax1.set_xscale("log", base=2)

# Create second y-axis for memory throughput
ax2 = ax1.twinx()
color2 = (93/255, 22/255, 130/255)  # RGB 93/22/130 (purple)
ax2.set_ylabel("TB/sec (Memory Throughput)", color=color2)
line2 = ax2.plot(num_tokens, tb_per_sec, color=color2, marker="s", label="TB/sec")
ax2.tick_params(axis="y", labelcolor=color2)

# Add title and legend
plt.title("Compute and Memory Throughput vs num_tokens")
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="upper left")

# Add grid
ax1.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("throughput_plot.png", dpi=150)
plt.show()

print("Plot saved to throughput_plot.png")
