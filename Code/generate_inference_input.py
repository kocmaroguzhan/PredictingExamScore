import pandas as pd
import os
import ast
import numpy as np

def fill_embedding(x,embedding_size):
    try:
        return x if isinstance(ast.literal_eval(x), list) else str([0.0] * embedding_size)
    except:
        return str([0.0] * embedding_size)
parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
dataset_folder = os.path.join(parent_folder, "Dataset")
config_path = os.path.join(dataset_folder, "config.txt")
augmentation_count=0
model_embedding_size = 0
number_of_augmented_code_saving_threshold=0
# Read config value
with open(config_path, "r") as f:
    for line in f:
        if line.startswith("augmentation_count"):
            augmentation_count = int(line.strip().split("=")[1])
        if line.startswith("model_embedding_size"):
            model_embedding_size = int(line.strip().split("=")[1])
        if line.startswith("number_of_augmented_code_saving_threshold"):
            number_of_augmented_code_saving_threshold = int(line.strip().split("=")[1])

print("✅ augmentation_count =", augmentation_count)
print("✅ model_embedding_size =", model_embedding_size)
print("✅ number_of_augmented_code_saving_threshold =", number_of_augmented_code_saving_threshold)

# Load CSVs
embedding_df = pd.read_csv(os.path.join(dataset_folder, "merged_embeddings.csv"))
test_df = pd.read_csv(os.path.join(dataset_folder, "processed_test_student.csv"))

inference_df = pd.merge(test_df, embedding_df, on="UID", how="left")
# Determine embedding size from first non-null feature column
embedding_original_cols = [col for col in inference_df.columns if "_original" in col and not col.endswith("_valid")]
embedding_augmented_cols = [col for col in inference_df.columns if "_augmented" in col and not col.endswith("_valid")]
validity_cols = [col for col in inference_df.columns if col.endswith("_valid")]

# Fill empty embeddings columns with zero-vectors
for col in embedding_original_cols + embedding_augmented_cols:
    inference_df[col] = inference_df[col].apply(lambda x: fill_embedding(x, model_embedding_size))
# Fill valid columns with 0
for col in validity_cols:
    inference_df[col] = inference_df[col].fillna(0).astype(int)
# Save to CSV
output_path = os.path.join(dataset_folder, "inference_data.csv")
inference_df.to_csv(output_path, index=False)

print(f"✅ Inference data saved to: {output_path}")