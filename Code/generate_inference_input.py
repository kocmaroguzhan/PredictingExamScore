import pandas as pd
import os
import ast
import numpy as np

parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
dataset_folder = os.path.join(parent_folder, "Dataset")
config_path = os.path.join(parent_folder, "config.txt")
clustering_size = 0

with open(config_path, "r") as f:
    for line in f:
        if line.startswith("clustering_size"):
            clustering_size = int(line.strip().split("=")[1])

print("✅ clustering_size =", clustering_size)

# Load CSVs
features_df = pd.read_csv(os.path.join(dataset_folder, "merged_features.csv"))
test_df = pd.read_csv(os.path.join(dataset_folder, "processed_test_student.csv"))

inference_df = pd.merge(test_df, features_df, on="UID", how="left")

# Fill NaNs in feature columns with fallback clustering ID
for col in inference_df.columns:
    if col != "UID":
        inference_df[col] = inference_df[col].fillna(clustering_size).astype(int)

# Save to CSV
output_path = os.path.join(dataset_folder, "inference_data.csv")
inference_df.to_csv(output_path, index=False)

print(f"✅ Inference data saved to: {output_path}")