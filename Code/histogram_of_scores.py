import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
# -------------------------
parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
dataset_folder = os.path.join(parent_folder, "Dataset")

labeled_df=pd.read_csv(os.path.join(dataset_folder, "labeled_data.csv"))
# Load data
original_scores = labeled_df["MidtermClass"].values


# Define fixed-width bins from 0 to 20 in steps of 2
custom_bins = np.arange(0, 22, 2)  # [0, 2, 4, ..., 20]
# Plot histogram with bin edges
plt.figure(figsize=(10, 5))
counts, bins, _ = plt.hist(original_scores, bins=custom_bins, edgecolor='black', color='skyblue', alpha=0.7)

# Annotate bin edges
for b in bins:
    plt.axvline(x=b, color='gray', linestyle='--', linewidth=0.8)

bin_labels = [f"{int(bins[i])}-{int(bins[i+1])}" for i in range(len(bins)-1)]
bin_centers = 0.5 * (bins[:-1] + bins[1:])
plt.xticks(bin_centers, bin_labels)
# Title and labels
plt.title("Histogram of Midterm Scores with Bin Edges")
plt.xlabel("Midterm Score")
plt.ylabel("Number of Students")
plt.grid(True)
plt.tight_layout()
plt.show()

# Quantile bin statistics
quantile_labels = pd.qcut(original_scores, q=10, duplicates="drop")
quantile_counts = pd.value_counts(quantile_labels, sort=False)

print("🎯 Quantile bins and number of students in each:")
print(quantile_counts)
