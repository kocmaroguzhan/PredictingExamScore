import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import random
import re
from collections import defaultdict



# Robust method pattern that supports multiple modifiers
method_pattern = re.compile(
    r'\b(public|private|protected)\b'        # Access modifier
    r'(?:\s+\w+)*'                           # Optional modifiers
    r'\s+[\w<>\[\]]+'                        # Return type
    r'\s+\w+'                                # Method name
    r'\s*\([^)]*\)\s*'                       # Parameter list
)


# Function to count .java files per student
def count_java_code_submissions(data, target_student_id):
    total_count = 0 
    for entry in data["answers"]:
        student_id = str(entry["id"])
        if student_id == str(target_student_id):  # Match only the target student
            java_codes = [k for k in entry.keys() if k != "id" and k.endswith(".java")]
            total_count += len(java_codes)
    return total_count


# ----------------------------
# Main Feature Extracting Pipeline
# ----------------------------


def count_effective_lines(full_code):
    # Remove block comments (/* ... */)
    code_no_block_comments = re.sub(r"/\*.*?\*/", "", full_code, flags=re.DOTALL)

    # Remove single-line comments (// ...)
    code_no_line_comments = re.sub(r"//.*?$", "", code_no_block_comments, flags=re.MULTILINE)

    # Split into lines
    lines = code_no_line_comments.split('\n')

    effective_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and stripped not in ['{', '}']:
            effective_lines.append(line)

    return len(effective_lines)

def save_student_features(student_features,output_path):
    rows = []
    for student_id, feature_dict in student_features.items():
        row = {"UID": student_id}
        for feature_id, selected_feature_dict in feature_dict.items():
            for key, value in selected_feature_dict.items():
                row[f"{feature_id}_{key}"] = value
        rows.append(row)
    df=pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
def generate_all_quiz_embeddings():
    # Paths
    parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    feature_folder = os.path.join(parent_folder, "Quizzes")
    dataset_folder = os.path.join(parent_folder, "Dataset")
    os.makedirs(dataset_folder, exist_ok=True)
    
    student_features = {}
    
    for file_name in os.listdir(feature_folder):
        if not file_name.endswith(".json"):
            continue
        feature_path = os.path.join(feature_folder, file_name)
        print(f"📄 Processing {file_name}...")
        feature_id = os.path.splitext(file_name)[0].split(".")[0]
        with open(feature_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for student in tqdm(data["answers"], desc=f"Features for {feature_id}"):
            student_id = student.get("id", "unknown")
            code_sections = [v for k, v in student.items() if k.endswith(".java")]
            full_code = "\n".join(code_sections)
            # Count words in code
            word_count = len(re.findall(r'\b\w+\b', full_code))
            if student_id not in student_features:
                student_features[student_id] = {}

            if feature_id not in student_features[student_id]:
                student_features[student_id][feature_id] = {
                    "java_code_counts": 0,
                    "is_attended": 0,
                    "num_of_words_in_code":0,
                    "num_of_effective_lines_in_code":0
                }
            student_features[student_id][feature_id]["java_code_counts"] = count_java_code_submissions(data,student_id)
            student_features[student_id][feature_id]["is_attended"]=1
            student_features[student_id][feature_id]["num_of_words_in_code"]=word_count
            student_features[student_id][feature_id]["num_of_effective_lines_in_code"]=count_effective_lines(full_code)

        
    # Fill missing data with zeros
    all_feature_ids = set()

    # Step 1: Gather all feature IDs encountered
    for file_name in os.listdir(feature_folder):
        if file_name.endswith(".json"):
            feature_id = os.path.splitext(file_name)[0].split(".")[0]
            all_feature_ids.add(feature_id)

    # Step 2: Ensure every student has every feature initialized
    for student_id in student_features:
        for feature_id in all_feature_ids:
            if feature_id not in student_features[student_id]:
                student_features[student_id][feature_id] = {
                    "java_code_counts": 0,
                    "is_attended": 0,
                    "num_of_words_in_code": 0,
                    "num_of_effective_lines_in_code": 0
                }
    output_csv_path = os.path.join(dataset_folder, "all_quiz_embeddings.csv")
    save_student_features(student_features,output_csv_path)
