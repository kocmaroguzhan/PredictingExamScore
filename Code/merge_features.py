import pandas as pd
import os
import ast 

def get_empty_code_embedding(empty_code_embedding_path):
    # Load the CSV
    df = pd.read_csv(empty_code_embedding_path)

    # Convert string back to list
    empty_code_embedding = ast.literal_eval(df["empty_code_embedding"].iloc[0])

    return empty_code_embedding
def merge_student_features():
    parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dataset_folder = os.path.join(parent_folder, "Dataset")
    empty_code_embedding_path=os.path.join(dataset_folder, "empty_code_embedding.csv")
    empty_code_embedding=get_empty_code_embedding(empty_code_embedding_path)
    # Load CSVs
    quiz_df = pd.read_csv(os.path.join(dataset_folder, "all_quiz_embeddings.csv"))
    lab_df = pd.read_csv(os.path.join(dataset_folder, "all_lab_embeddings.csv"))


    # Merge on student ID
    quiz_df.rename(columns={"student_id": "UID"}, inplace=True)
    lab_df.rename(columns={"student_id": "UID"}, inplace=True)
    quiz_df.rename(columns={"assessment_section": "quiz_assessment_section"}, inplace=True)
    lab_df.rename(columns={"assessment_section": "lab_assessment_section"}, inplace=True)

    # Merge quiz and lab embeddings on UID (student ID) using an outer join.
    # This keeps all students who have either quiz or lab embeddings (or both).
    merged_df = pd.merge(quiz_df, lab_df, on="UID", how="outer")

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
    
    ## Fill NaNs with empty code embeddings
    for col in merged_df.columns:
        if col != "UID" and ("_valid" not in col) and ("assessment_section" not in col):
            merged_df[col] = merged_df[col].apply(
                lambda x: x if isinstance(x, list) and len(x) == model_embedding_size
                else eval(x) if isinstance(x, str) else empty_code_embedding
            )
    # Fill NaNs in valid_flag columns with 1
    for col in merged_df.columns:
        if col.endswith("_valid"):
            merged_df[col] = merged_df[col].fillna(1).astype(int)
    # Save to CSV
    output_path = os.path.join(dataset_folder, "merged_embeddings.csv")
    merged_df.to_csv(output_path, index=False)

    print(f"✅ Merged embeddings saved to: {output_path}")