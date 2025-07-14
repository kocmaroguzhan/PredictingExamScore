import pandas as pd
import os
import ast
import json
parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
dataset_folder = os.path.join(parent_folder, "Dataset")
config_path = os.path.join(parent_folder, "config.txt")
model_embedding_size = 0
# Read config value
with open(config_path, "r") as f:
    for line in f:
        if line.startswith("model_embedding_size"):
            model_embedding_size = int(line.strip().split("=")[1])
print("✅ model_embedding_size =", model_embedding_size)
# Load CSVs
features_df = pd.read_csv(os.path.join(dataset_folder, "merged_embeddings.csv"))
label_df = pd.read_csv(os.path.join(dataset_folder, "processed_train_student.csv"))

labeled_df = pd.merge(features_df, label_df, on="UID", how="right")
# Fill NaN values in cluster feature columns with fallback cluster ID
for col in labeled_df.columns:
    if col == "UID":
        continue
    elif col.endswith("_embedding"):
        labeled_df[col] = labeled_df[col].apply(
            lambda x: json.dumps([0.0] * model_embedding_size) if pd.isna(x) else x
        )
    elif col.endswith("_valid"):
        labeled_df[col] = labeled_df[col].fillna(0).astype(int)
# Save to CSV
output_path = os.path.join(dataset_folder, "labeled_data.csv")
labeled_df.to_csv(output_path, index=False)

print(f"✅ Labeled data saved to: {output_path}")