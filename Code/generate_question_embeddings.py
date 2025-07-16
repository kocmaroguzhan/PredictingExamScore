import json
import torch
import numpy as np
import pandas as pd

import os
import ast
from transformers import RobertaTokenizer, RobertaModel



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
    stride = chunk_size // 2  # e.g., 256 if chunk_size is 512-- %50 overlap
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



def generate_all_question_embeddings():
    # Paths
    parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    labs_folder = os.path.join(parent_folder, "Labs")
    quiz_folder= os.path.join(parent_folder, "Quizzes")
    dataset_folder = os.path.join(parent_folder, "Dataset")
    os.makedirs(dataset_folder, exist_ok=True)
    # Load CodeBERT
    model_name = "microsoft/codebert-base"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaModel.from_pretrained(model_name)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Store all question embeddings
    question_embeddings = []
    for file_name in os.listdir(labs_folder):
        if not file_name.endswith(".json"):
            continue

        lab_path = os.path.join(labs_folder, file_name)
        lab_id = os.path.splitext(file_name)[0]

        print(f"📄 Processing {file_name}...")

        with open(lab_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        raw_question = data.get("questions", [])
        question_text=""
        if isinstance(raw_question, list):
            question_text = " ".join(raw_question).strip().replace("\n", " ")
        else:
            question_text = raw_question.strip().replace("\n", " ")
       
        question_embedding  = get_embedding_for_long_code(question_text, tokenizer, model, device)
        question_embeddings.append({
            "lab_id": lab_id,
            "embedding": question_embedding.tolist()
        })
    for file_name in os.listdir(quiz_folder):
        if not file_name.endswith(".json"):
            continue

        quiz_path = os.path.join(quiz_folder, file_name)
        quiz_id = os.path.splitext(file_name)[0]

        print(f"📄 Processing {file_name}...")

        with open(quiz_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        raw_question = data.get("questions", [])
        question_text=""
        if isinstance(raw_question, list):
            question_text = " ".join(raw_question).strip().replace("\n", " ")
        else:
            question_text = raw_question.strip().replace("\n", " ")
       
        question_embedding  = get_embedding_for_long_code(question_text, tokenizer, model, device)
        question_embeddings.append({
            "quiz_id": quiz_id,
            "embedding": question_embedding.tolist()
        })
    # Merge into a single row with column names as IDs
    single_row = {}
    for item in question_embeddings:
        key = item.get("lab_id") or item.get("quiz_id")
        single_row[key] = item["embedding"]

    # Convert to DataFrame with a single row
    df = pd.DataFrame([single_row])

    # Save to CSV (store lists as strings)
    output_csv = os.path.join(dataset_folder, "question_embeddings.csv")
    df.to_csv(output_csv, index=False)

    print(f"✅ Saved wide-format question embeddings to: {output_csv}")


if __name__ == "__main__":
    generate_all_question_embeddings()