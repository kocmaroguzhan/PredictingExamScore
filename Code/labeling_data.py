import pandas as pd
import os
import ast

parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
dataset_folder = os.path.join(parent_folder, "Dataset")

# Load CSVs
feature_df = pd.read_csv(os.path.join(dataset_folder, "student_features_with_clusters.csv"))
label_df = pd.read_csv(os.path.join(dataset_folder, "processed_train_student.csv"))

labeled_df = pd.merge(feature_df, label_df, on="UID", how="right")

# Fill empty columns with 0
#for col in labeled_df.columns:
#    if col != "UID" :
#        labeled_df[col] =labeled_df[col].fillna(0).astype(int)
# Save to CSV
output_path = os.path.join(dataset_folder, "labeled_data.csv")
labeled_df.to_csv(output_path, index=False)

print(f"✅ Labeled data saved to: {output_path}")