# Core libraries
import os
import ast
import numpy as np
import pandas as pd
from collections import defaultdict
from pprint import pprint  # Pretty-print for better readability
# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Flatten embedding and validity rows
def flatten_embeddings_with_valid_flag(row, emb_cols, valid_cols):
    emb_vectors = [np.array(row[col]) for col in emb_cols]
    valid_flags = np.array([row[col] for col in valid_cols], dtype=np.float32)  # ensure it's a 1D array
    return np.concatenate(emb_vectors + [valid_flags])

parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
dataset_folder = os.path.join(parent_folder, "Dataset")
inference_df=pd.read_csv(os.path.join(dataset_folder, "inference_data.csv"))
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
# Identify embedding columns
embedding_original_cols = [col for col in inference_df.columns if "_original" in col and not col.endswith("_valid")]
embedding_augmented_cols = [col for col in inference_df.columns if "_augmented" in col and not col.endswith("_valid")]
# Get original_valid columns
original_valid_cols = [col for col in inference_df.columns if "_original_valid" in col]
# Convert stringified vectors into lists
for col in embedding_original_cols + embedding_augmented_cols:
    inference_df[col] = inference_df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Original dataset
original_df = inference_df.copy()
original_df["features"] = inference_df.apply(
    lambda row: flatten_embeddings_with_valid_flag(row, embedding_original_cols, original_valid_cols), axis=1)

features = []
uids = []

# Add original features
for _, row in inference_df.iterrows():
    vec = flatten_embeddings_with_valid_flag(row, embedding_original_cols,original_valid_cols)
    features.append(vec)
    uids.append(row["UID"])


# Add augmented features
for _, row in inference_df.iterrows():
    for i in range(augmentation_count):
        aug_cols = [col.replace("original", f"augmented_{i}") for col in embedding_original_cols]
        valid_cols = [col.replace("original", f"augmented_{i}") + "_valid" for col in embedding_original_cols]
        
        # Proceed only if all required columns are present
        if all(col in row and isinstance(row[col], list) for col in aug_cols + valid_cols):
            vec = flatten_embeddings_with_valid_flag(row, aug_cols, valid_cols)
            features.append(vec)
            uids.append(row["UID"])



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# === Prepare input for model ===
X = torch.tensor(np.array(features), dtype=torch.float32).to(device)


input_dim = X.shape[1]

class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            #nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 64),
            #nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 16),
            #nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),

            #nn.Linear(16, 1),
            #nn.BatchNorm1d(32),
            #nn.ReLU(),
            #nn.Dropout(0.1),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.model(x)
# Load model and move to device
model = MLP(input_dim).to(device)
model.load_state_dict(torch.load("best_mlp_model.pt", map_location=device))
model.eval()


# Inference
with torch.no_grad():
    predictions = model(X).squeeze().cpu().numpy()

uid_scores = defaultdict(list)
for uid, pred in zip(uids, predictions):
    uid_scores[uid].append(pred)
pprint(dict(uid_scores))
# === Take average score per UID ===
final_predictions = {uid: int(round(np.mean(scores))) for uid, scores in uid_scores.items()}

# === Save to DataFrame ===
result_df = pd.DataFrame({
    "UID": list(final_predictions.keys()),
    "MidtermClass": list(final_predictions.values())
})

# === Save to CSV ===
output_path = os.path.join(dataset_folder, "predicted_scores.csv")
result_df.to_csv(output_path, index=False)
print(f"✅ Saved predicted scores to: {output_path}")