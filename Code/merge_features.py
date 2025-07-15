import pandas as pd
import os
import ast 


def merge_student_features():
    parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dataset_folder = os.path.join(parent_folder, "Dataset")

    # Load CSVs
    quiz_df = pd.read_csv(os.path.join(dataset_folder, "all_quiz_embeddings.csv"))
    lab_df = pd.read_csv(os.path.join(dataset_folder, "all_lab_embeddings.csv"))



    # Merge quiz and lab embeddings on UID (student ID) using an outer join.
    # This keeps all students who have either quiz or lab embeddings (or both).
    merged_df = pd.merge(quiz_df, lab_df, on="UID", how="outer")
    ## Fill NaNs with zeros
    for col in merged_df.columns:
        if col != "UID" :
            merged_df[col] =merged_df[col].fillna(0).astype(int)
    # Save to CSV
    output_path = os.path.join(dataset_folder, "merged_features.csv")
    merged_df.to_csv(output_path, index=False)

    print(f"✅ Merged embeddings saved to: {output_path}")