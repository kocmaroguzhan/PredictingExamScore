import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import json
from transformers import RobertaTokenizer, RobertaModel




def get_embedding(code, tokenizer, model, device):
    tokenized = tokenizer.encode_plus(
        code,
        return_tensors="pt",
        add_special_tokens=True,
        return_attention_mask=True,
        truncation=True,
        max_length=512  # optional, safe cutoff
    )

    input_ids = tokenized['input_ids'].to(device)
    attention_mask = tokenized['attention_mask'].to(device)

    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = output.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

    # ℓ2 normalisation
    norm = np.linalg.norm(cls_embedding)
    return cls_embedding if norm == 0 else cls_embedding / norm



def generate_empty_code_embeddings():
    # Load CodeBERT
    model_name = "microsoft/codebert-base"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaModel.from_pretrained(model_name)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)    
    empty_code = ""
    empty_code_embedding = get_embedding(empty_code, tokenizer, model, device)
    parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dataset_folder = os.path.join(parent_folder, "Dataset")
    output_path = os.path.join(dataset_folder, "empty_code_embedding.csv")
    # Save to CSV
    df = pd.DataFrame({
    "empty_code_embedding": [empty_code_embedding.tolist()]  # Convert NumPy array to list, then wrap in a list to make one row
    })
    df.to_csv(output_path, index=False)

    print(f"✅ Empty code embedding saved to {output_path}")

# === Entry point ===
if __name__ == "__main__":
    generate_empty_code_embeddings()

            