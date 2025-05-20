import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Step 1: Load and process CSVs
# -----------------------------
# Define input folders
ours_dir = "ours_metric"
paper1_dir = "paper1_metric"
paper8_dir = "tap_paper8_metric"

# List of metric columns to use
metrics = ["Average Rank", "Average Perplexity", "Average Bad Word Ratio"]

# Function to extract numeric values
def clean_metrics(df):
    for col in metrics:
        if col in df.columns:
            df[col] = df[col].astype(str).str.extract(r"([0-9.]+)").astype(float)
    return df

# Load files with label
def load_and_label(folder, label, file_set):
    records = []
    for fname in file_set:
        if fname.endswith(".csv"):
            path = os.path.join(folder, fname)
            df = pd.read_csv(path)
            df = clean_metrics(df)
            df["Source"] = label
            records.append(df)
    return records

# Find common files
common_files = set(os.listdir(ours_dir)) & set(os.listdir(paper1_dir)) & set(os.listdir(paper8_dir))

# Load all sources
ours_records = load_and_label(ours_dir, "ours", common_files)
paper1_records = load_and_label(paper1_dir, "paper1", common_files)
paper8_records = load_and_label(paper8_dir, "paper8", common_files)

# Combine into one dataframe
combined = pd.concat(ours_records + paper1_records + paper8_records, ignore_index=True)

# Aggregate
summary = combined.groupby(["Model", "Source"])[metrics].mean().reset_index()

# Pivot
pivoted = summary.pivot(index="Model", columns="Source", values=metrics)
pivoted.columns = [f"{metric}_{source}" for metric, source in pivoted.columns]
pivoted.reset_index(inplace=True)

# Save summary
pivoted.to_csv("model_comparison_summary_with_paper8.csv", index=False)
print(pivoted)

# -----------------------------
# Step 2: Plotting
# -----------------------------
# Rename columns for plotting
pivoted_filtered = pivoted.rename(columns={
    'Average Rank_ours': 'Rank_Ours',
    'Average Rank_paper1': 'Rank_STS',
    'Average Rank_paper8': 'Rank_TAP',
    'Average Perplexity_ours': 'Perplexity_Ours',
    'Average Perplexity_paper1': 'Perplexity_STS',
    'Average Perplexity_paper8': 'Perplexity_TAP',
    'Average Bad Word Ratio_ours': 'BadWord_Ours',
    'Average Bad Word Ratio_paper1': 'BadWord_STS',
    'Average Bad Word Ratio_paper8': 'BadWord_TAP'
})

# Color palette
colors = ['#4C72B0', '#DD8452', '#55A868']  # Ours, STS, TAP
x = np.arange(len(pivoted_filtered))
bar_width = 0.25

# --- Figure 1: Average Rank ↓ ---
fig1, ax1 = plt.subplots(figsize=(6.5, 4))
ax1.bar(x - bar_width, pivoted_filtered['Rank_Ours'], bar_width, label="Ours", color=colors[0])
ax1.bar(x, pivoted_filtered['Rank_STS'], bar_width, label="STS", color=colors[1])
ax1.bar(x + bar_width, pivoted_filtered['Rank_TAP'], bar_width, label="TAP", color=colors[2])
ax1.set_title("Average Rank ↓")
ax1.set_ylabel("Score")
ax1.set_xticks(x)
ax1.set_xticklabels(pivoted_filtered['Model'], rotation=45, ha='right')
for j in range(len(pivoted_filtered)):
    ax1.text(x[j] - bar_width, pivoted_filtered['Rank_Ours'][j], f"{pivoted_filtered['Rank_Ours'][j]:.2f}", ha='center', va='bottom', fontsize=8)
    ax1.text(x[j], pivoted_filtered['Rank_STS'][j], f"{pivoted_filtered['Rank_STS'][j]:.2f}", ha='center', va='bottom', fontsize=8)
    ax1.text(x[j] + bar_width, pivoted_filtered['Rank_TAP'][j], f"{pivoted_filtered['Rank_TAP'][j]:.2f}", ha='center', va='bottom', fontsize=8)
ax1.legend()
fig1.tight_layout()
fig1.savefig("comparison_rank_with_tap.pdf")

# --- Figure 2: Perplexity ↓ (log scale) ---
fig2, ax2 = plt.subplots(figsize=(6.5, 4))
ax2.bar(x - bar_width, pivoted_filtered['Perplexity_Ours'], bar_width, label="Ours", color=colors[0])
ax2.bar(x, pivoted_filtered['Perplexity_STS'], bar_width, label="STS", color=colors[1])
ax2.bar(x + bar_width, pivoted_filtered['Perplexity_TAP'], bar_width, label="TAP", color=colors[2])
ax2.set_title("Perplexity ↓ (log scale)")
ax2.set_ylabel("Log-Scale Perplexity")
ax2.set_yscale('log')
ax2.set_xticks(x)
ax2.set_xticklabels(pivoted_filtered['Model'], rotation=45, ha='right')
for j in range(len(pivoted_filtered)):
    ax2.text(x[j] - bar_width, pivoted_filtered['Perplexity_Ours'][j], f"{pivoted_filtered['Perplexity_Ours'][j]:.0f}", ha='center', va='bottom', fontsize=8)
    ax2.text(x[j], pivoted_filtered['Perplexity_STS'][j], f"{pivoted_filtered['Perplexity_STS'][j]:.0f}", ha='center', va='bottom', fontsize=8)
    ax2.text(x[j] + bar_width, pivoted_filtered['Perplexity_TAP'][j], f"{pivoted_filtered['Perplexity_TAP'][j]:.0f}", ha='center', va='bottom', fontsize=8)
ax2.legend()
fig2.tight_layout()
fig2.savefig("comparison_perplexity_log_with_tap.pdf")

# --- Figure 3: Bad Word Ratio ↓ ---
fig3, ax3 = plt.subplots(figsize=(6.5, 4))
ax3.bar(x - bar_width, pivoted_filtered['BadWord_Ours'], bar_width, label="Ours", color=colors[0])
ax3.bar(x, pivoted_filtered['BadWord_STS'], bar_width, label="STS", color=colors[1])
ax3.bar(x + bar_width, pivoted_filtered['BadWord_TAP'], bar_width, label="TAP", color=colors[2])
ax3.set_title("Bad Word Ratio ↓")
ax3.set_ylabel("Ratio")
ax3.set_xticks(x)
ax3.set_xticklabels(pivoted_filtered['Model'], rotation=45, ha='right')
for j in range(len(pivoted_filtered)):
    ax3.text(x[j] - bar_width, pivoted_filtered['BadWord_Ours'][j], f"{pivoted_filtered['BadWord_Ours'][j]:.2f}", ha='center', va='bottom', fontsize=8)
    ax3.text(x[j], pivoted_filtered['BadWord_STS'][j], f"{pivoted_filtered['BadWord_STS'][j]:.2f}", ha='center', va='bottom', fontsize=8)
    ax3.text(x[j] + bar_width, pivoted_filtered['BadWord_TAP'][j], f"{pivoted_filtered['BadWord_TAP'][j]:.2f}", ha='center', va='bottom', fontsize=8)
ax3.legend()
fig3.tight_layout()
fig3.savefig("comparison_badword_with_tap.pdf")
