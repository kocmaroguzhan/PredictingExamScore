import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from transformers import RobertaTokenizer, RobertaModel
import csv 
import re
from sklearn.cluster import KMeans
import numpy as np
import hdbscan
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_distances

# Robust method pattern that supports multiple modifiers
method_pattern = re.compile(
    r'\b(public|private|protected)\b'        # Access modifier
    r'(?:\s+\w+)*'                           # Optional modifiers
    r'\s+[\w<>\[\]]+'                        # Return type
    r'\s+\w+'                                # Method name
    r'\s*\([^)]*\)\s*'                       # Parameter list
)



# ----------------------------
# Long Code Embedding Function
# ----------------------------

def get_embedding_for_long_code(code, tokenizer, model, device, chunk_size=512):
    tokenized = tokenizer.encode_plus(
        code,
        return_tensors="pt",
        add_special_tokens=True,
        return_attention_mask=True,
        truncation=False,
        max_length=99999  # large number to avoid truncation here
    )

    input_ids = tokenized['input_ids'].squeeze(0)
    attention_mask = tokenized['attention_mask'].squeeze(0)

    total_tokens = input_ids.size(0)
    stride = chunk_size // 4  # e.g., 128 if chunk_size is 512
    embeddings = []

    for start in range(0, total_tokens, stride):
        end = min(start + chunk_size, total_tokens)
        chunk_ids = input_ids[start:end].unsqueeze(0)
        chunk_mask = attention_mask[start:end].unsqueeze(0)

        with torch.no_grad():
            output = model(input_ids=chunk_ids.to(device), attention_mask=chunk_mask.to(device))
            chunk_embedding = output.last_hidden_state[:, 0, :].squeeze().cpu().numpy() #CLS token
            embeddings.append(chunk_embedding)

        if end == total_tokens:
            break

    if not embeddings:
        return np.zeros(model.config.hidden_size)

    final_embedding = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(final_embedding)
    if norm == 0:
        return final_embedding
    return final_embedding / norm


# ----------------------------
# Main Embedding Pipeline
# ----------------------------

def generate_all_data_embeddings():
    # Paths
    parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_folder = os.path.join(parent_folder, "Labs")
    dataset_folder = os.path.join(parent_folder, "Dataset")
    students_code_folder=os.path.join(dataset_folder, "Students_Code")
    os.makedirs(dataset_folder, exist_ok=True)
    os.makedirs(students_code_folder, exist_ok=True)
    config_path = os.path.join(parent_folder, "config.txt")
    model_embedding_size = 0
    number_of_code_saving_threshold=0
    # Read config value
    with open(config_path, "r") as f:
        for line in f:
            if line.startswith("model_embedding_size"):
                model_embedding_size = int(line.strip().split("=")[1])
            if line.startswith("number_of_code_saving_threshold"):
                number_of_code_saving_threshold = int(line.strip().split("=")[1])

    print("✅ model_embedding_size =", model_embedding_size)
    print("✅ number_of_code_saving_threshold =", number_of_code_saving_threshold)
    # Load CodeBERT
    model_name = "microsoft/codebert-base"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaModel.from_pretrained(model_name)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    student_embeddings = {}
    for file_name in os.listdir(data_folder):
        if not file_name.endswith(".json"):
            continue

        data_path = os.path.join(data_folder, file_name)
        data_id = os.path.splitext(file_name)[0].split(".")[0]

        print(f"📄 Processing {file_name}...")
        number_of_code_saved_code=0
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for student in tqdm(data["answers"], desc=f"Embedding {data_id}"):
            student_id = student.get("id", "unknown")
            code_sections = [v for k, v in student.items() if k.endswith(".java")]
            full_code = "\n".join(code_sections)
            if number_of_code_saved_code < number_of_code_saving_threshold:
                number_of_code_saved_code=number_of_code_saved_code+1
                output_txt_path = os.path.join(students_code_folder, f"{student_id}_{data_id}_original.txt")
                with open(output_txt_path, "w", encoding="utf-8") as txt_file:
                    txt_file.write(full_code)
                print(f"📝 Saved original code to {output_txt_path}")
            # Get original embedding
            original_embedding = get_embedding_for_long_code(full_code, tokenizer, model, device)
            if student_id not in student_embeddings:
                    student_embeddings[student_id] = {}
            if data_id not in student_embeddings[student_id]:
                student_embeddings[student_id][data_id] = {
                    "original": {}
                }
            # Save original
            student_embeddings[student_id][data_id]["original"] = {
                "embedding": original_embedding.tolist(),
                "valid": True
            }
           
    # Fill missing data with zeros
    all_data_ids = set()
    for data_scores in student_embeddings.values():
        all_data_ids.update(data_scores.keys())

    
    for student_id in student_embeddings:
        for data_id in all_data_ids:
            if data_id not in student_embeddings[student_id]:
                student_embeddings[student_id][data_id] = {
                     "original": {
                        "embedding": [0.0] * model_embedding_size,
                        "valid": False
                    }
                }
    return student_embeddings



   

def save_student_embeddings_to_csv(student_embeddings, filename="lab_embeddings.csv"):
    """
    Saves student embeddings in wide format:
    Each row has student_id, and for each data_id: embedding and valid flag.

    Args:
        student_embeddings (dict): student_id → data_id → "original" → {"embedding", "valid"}
        filename (str): Output CSV filename
    """
    # Determine all unique data_ids
    all_data_ids = sorted({
        data_id for student_data in student_embeddings.values() for data_id in student_data
    })

    # Build headers: student_id, quiz1_embedding, quiz1_valid, ...
    fieldnames = ["student_id"]
    for data_id in all_data_ids:
        fieldnames.append(f"{data_id}_embedding")
        fieldnames.append(f"{data_id}_valid")

    # Prepare output path
    parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dataset_folder = os.path.join(parent_folder, "Dataset")
    os.makedirs(dataset_folder, exist_ok=True)
    output_path = os.path.join(dataset_folder, filename)

    # Write to CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for student_id, data_dict in student_embeddings.items():
            row = {"student_id": student_id}
            for data_id in all_data_ids:
                entry = data_dict.get(data_id, {}).get("original", {})
                emb = entry.get("embedding", [])
                valid = entry.get("valid", False)
                row[f"{data_id}_embedding"] = json.dumps(emb)
                row[f"{data_id}_valid"] = valid
            writer.writerow(row)

    print(f"✅ Embeddings saved to: {output_path}")
 
def generate_lab_features():

    student_embeddings=generate_all_data_embeddings()
    save_student_embeddings_to_csv(student_embeddings)
