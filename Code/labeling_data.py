import pandas as pd
import os
import ast

parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
dataset_folder = os.path.join(parent_folder, "Dataset")
empty_code_embedding_path=os.path.join(dataset_folder, "empty_code_embedding.csv")


def get_empty_code_embedding():
    # Load the CSV
    df = pd.read_csv(empty_code_embedding_path)

    # Convert string back to list
    empty_code_embedding = ast.literal_eval(df["empty_code_embedding"].iloc[0])

    return empty_code_embedding
empty_code_embedding=get_empty_code_embedding()
def fill_embedding(x):
    try:
        return x if isinstance(ast.literal_eval(x), list) else empty_code_embedding
    except:
        return empty_code_embedding

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
label_df = pd.read_csv(os.path.join(dataset_folder, "processed_train_student.csv"))

labeled_df = pd.merge(embedding_df, label_df, on="UID", how="right")
# Determine embedding size from first non-null feature column
embedding_original_cols = [col for col in labeled_df.columns if "_original" in col and not col.endswith("_valid")]
embedding_augmented_cols = [col for col in labeled_df.columns if "_augmented" in col and not col.endswith("_valid")]
validity_cols = [col for col in labeled_df.columns if col.endswith("_valid")]
quiz_section_cols="quiz_assessment_section"
lab_section_cols="lab_assessment_section"
# Fill empty embeddings columns with zero-vectors
for col in embedding_original_cols + embedding_augmented_cols:
    labeled_df[col] = labeled_df[col].apply(lambda x: fill_embedding(x))

# Fill status columns with 1
for col in validity_cols:
    labeled_df[col] = labeled_df[col].fillna(1).astype(int)

labeled_df[quiz_section_cols] = labeled_df[quiz_section_cols].fillna("Quiz1.ube_1")#default value
labeled_df[lab_section_cols] = labeled_df[lab_section_cols].fillna("Lab1.ube_1")#default value
# Save to CSV
output_path = os.path.join(dataset_folder, "labeled_data.csv")
labeled_df.to_csv(output_path, index=False)

print(f"✅ Labeled data saved to: {output_path}")