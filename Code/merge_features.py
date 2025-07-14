import pandas as pd
import os
import json
def merge_student_features():
    parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dataset_folder = os.path.join(parent_folder, "Dataset")
    # Load CSVs
    quiz_df = pd.read_csv(os.path.join(dataset_folder, "quiz_embeddings.csv"))
    lab_df = pd.read_csv(os.path.join(dataset_folder, "lab_embeddings.csv"))


    # Merge on student ID
    quiz_df.rename(columns={"student_id": "UID"}, inplace=True)
    lab_df.rename(columns={"student_id": "UID"}, inplace=True)

    # Merge quiz and lab embeddings on UID (student ID) using an outer join.
    # This keeps all students who have either quiz or lab embeddings (or both).
    merged_df = pd.merge(quiz_df, lab_df, on="UID", how="outer")
    parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    config_path = os.path.join(parent_folder, "config.txt")
    model_embedding_size = 0
    # Read config value
    with open(config_path, "r") as f:
        for line in f:
            if line.startswith("model_embedding_size"):
                model_embedding_size = int(line.strip().split("=")[1])
    print("✅ model_embedding_size =", model_embedding_size)

    for col in merged_df.columns:
        if col == "UID":
            continue
        elif col.endswith("_embedding"):
            merged_df[col] = merged_df[col].apply(
                lambda x: json.dumps([0.0] * model_embedding_size) if pd.isna(x) else x
            )
        elif col.endswith("_valid"):
            merged_df[col] = merged_df[col].fillna(0).astype(int)
    # Save to CSV
    output_path = os.path.join(dataset_folder, "merged_embeddings.csv")
    merged_df.to_csv(output_path, index=False)

    print(f"✅ Merged features saved to: {output_path}")