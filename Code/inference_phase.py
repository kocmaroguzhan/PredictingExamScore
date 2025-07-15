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
import joblib
def get_empty_code_embedding(empty_code_embedding_path):
    # Load the CSV
    df = pd.read_csv(empty_code_embedding_path)

    # Convert string back to list
    empty_code_embedding = ast.literal_eval(df["empty_code_embedding"].iloc[0])

    return empty_code_embedding

# Flatten embedding and validity rows
def flatten_embeddings_with_valid_flag(row, emb_cols, valid_cols):
    emb_vectors = [np.array(row[col]) for col in emb_cols]
    valid_flags = np.array([row[col] for col in valid_cols], dtype=np.float32)
    return np.concatenate(emb_vectors + [valid_flags])

def flatten_embeddings(row, emb_cols):
    emb_vectors = [np.array(row[col]) for col in emb_cols]
    return np.concatenate(emb_vectors )
parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
dataset_folder = os.path.join(parent_folder, "Dataset")
code_folder = os.path.join(parent_folder, "Code")
inference_df=pd.read_csv(os.path.join(dataset_folder, "inference_data.csv"))
config_path = os.path.join(dataset_folder, "config.txt")
with open(config_path, "r") as f:
    for line in f:
        if line.startswith("model_embedding_size"):
            model_embedding_size = int(line.strip().split("=")[1])
        if line.startswith("num_of_embeddings"):
            num_of_embeddings = int(line.strip().split("=")[1])
        if line.startswith("augmentation_count"):
            augmentation_count = int(line.strip().split("=")[1])
embedding_dim = model_embedding_size * num_of_embeddings
prediction_index_path = os.path.join(dataset_folder, "prediction_index.txt")
prediction_index=0
empty_code_embedding_path=os.path.join(dataset_folder, "empty_code_embedding.csv")
empty_code_embedding=get_empty_code_embedding(empty_code_embedding_path)

# Read config value
with open(prediction_index_path, "r") as f:
    for line in f:
        if line.startswith("prediction_index"):
            prediction_index= int(line.strip().split("=")[1])
            
next_prediction_index=prediction_index+1
print("✅ current prediction_index =", prediction_index)
print("✅ new prediction_index =", next_prediction_index)
# Read merged data

task_infos = {
    "quiz1": {"emb_cols": [], "valid_cols": [], "features": [], "uids": []},
    "quiz2": {"emb_cols": [], "valid_cols": [], "features": [], "uids": []},
    "lab1":  {"emb_cols": [], "valid_cols": [], "features": [], "uids": []},
    "lab2":  {"emb_cols": [], "valid_cols": [], "features": [], "uids": []},
}

# === Parse column names ===
for task in task_infos.keys():
    emb_cols = [col for col in inference_df.columns if f"{task.capitalize()}_" in col and "_original" in col and not col.endswith("_valid")]
    valid_cols = [col for col in inference_df.columns if f"{task.capitalize()}_" in col and "_original_valid" in col]
    task_infos[task]["emb_cols"] = emb_cols
    task_infos[task]["valid_cols"] = valid_cols
    for col in emb_cols:
        inference_df[col] = inference_df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Convert merged_df to UID lookup
merged_uid_lookup = inference_df.set_index("UID")

# === Extract original and augmented features per task ===
for task, info in task_infos.items():
    emb_cols = info["emb_cols"]
    valid_cols = info["valid_cols"]
    features = []
    uids = []

    for uid in inference_df["UID"]:
        if uid in merged_uid_lookup.index:
            row = merged_uid_lookup.loc[uid]
            # Original
            features.append(flatten_embeddings(row, emb_cols))
            uids.append(uid)
            # Augmented
            for i in range(augmentation_count):
                aug_emb_cols = [col.replace("original", f"augmented_{i}") for col in emb_cols]
                aug_valid_cols = [col.replace("original", f"augmented_{i}") + "_valid" for col in emb_cols]
                if all(col in row and isinstance(row[col], list) for col in aug_emb_cols + aug_valid_cols):
                    features.append(flatten_embeddings(row, aug_emb_cols))
                    uids.append(uid)
        else:
            # If UID is missing in merged_df, add empty code embedding feature vector
            features.append(empty_code_embedding)
            uids.append(uid)

    task_infos[task]["features"] = features
    task_infos[task]["uids"] = uids

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# === Prepare input for model ===
X_quiz_1 = task_infos["quiz1"]["features"]
scaler_path=os.path.join(code_folder, "x_scaler_quiz_1.pkl")
x_scaler_quiz_1 = joblib.load(scaler_path)
X_scaled_quiz_1 = x_scaler_quiz_1.transform(X_quiz_1)  # X must have same feature count as used in fitting
X_quiz_1_tensor = torch.tensor(np.array(X_scaled_quiz_1), dtype=torch.float32).to(device)

class MLP_Quiz_1(nn.Module):
    def __init__(self, input_dim):
        super(MLP_Quiz_1, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            #nn.Dropout(0.1),

            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.model(x)
# Load model and move to device
model_quiz_1 = MLP_Quiz_1(X_quiz_1_tensor.shape[1]).to(device)
model_quiz_1.load_state_dict(torch.load("best_full_model_quiz_1.pt", map_location=device))
model_quiz_1.eval()
# Inference
with torch.no_grad():
    predictions = model_quiz_1(X_quiz_1_tensor).squeeze().cpu().numpy()

scaler_path=os.path.join(code_folder, "y_scaler_quiz_1.pkl")
y_scaler_quiz_1 = joblib.load(scaler_path)
predictions = y_scaler_quiz_1.inverse_transform(predictions.reshape(-1, 1)).flatten()
uid_scores = defaultdict(list)
for uid, pred in zip(uids, predictions):
    uid_scores[uid].append(pred)

#pprint(dict(uid_scores))
# === Take average score per UID ===
final_predictions_quiz_1 = {uid: np.mean(scores) for uid, scores in uid_scores.items()}
print("Quiz 1 predictions")
print(final_predictions_quiz_1)

# === Prepare input for model ===
X_quiz_2 = task_infos["quiz2"]["features"]
scaler_path=os.path.join(code_folder, "x_scaler_quiz_2.pkl")
x_scaler_quiz_2 = joblib.load(scaler_path)
X_scaled_quiz_2 = x_scaler_quiz_2.transform(X_quiz_2)  # X must have same feature count as used in fitting
X_quiz_2_tensor = torch.tensor(np.array(X_scaled_quiz_2), dtype=torch.float32).to(device)

class MLP_Quiz_2(nn.Module):
    def __init__(self, input_dim):
        super(MLP_Quiz_2, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            #nn.Dropout(0.1),

            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.model(x)

model_quiz_2 = MLP_Quiz_2(X_quiz_2_tensor.shape[1]).to(device)
model_quiz_2.load_state_dict(torch.load("best_full_model_quiz_2.pt", map_location=device))
model_quiz_2.eval()
# Inference
with torch.no_grad():
    predictions = model_quiz_2(X_quiz_2_tensor).squeeze().cpu().numpy()

scaler_path=os.path.join(code_folder, "y_scaler_quiz_2.pkl")
y_scaler_quiz_2 = joblib.load(scaler_path)
predictions = y_scaler_quiz_2.inverse_transform(predictions.reshape(-1, 1)).flatten()
uid_scores = defaultdict(list)
for uid, pred in zip(uids, predictions):
    uid_scores[uid].append(pred)

#pprint(dict(uid_scores))
# === Take average score per UID ===
final_predictions_quiz_2 = {uid: np.mean(scores) for uid, scores in uid_scores.items()}
print("Quiz 2 predictions")
print(final_predictions_quiz_2)


# === Prepare input for model ===
X_lab_1 = task_infos["lab1"]["features"]
scaler_path=os.path.join(code_folder, "x_scaler_lab_1.pkl")
x_scaler_lab_1 = joblib.load(scaler_path)
X_scaled_lab_1 = x_scaler_lab_1.transform(X_lab_1)  # X must have same feature count as used in fitting
X_lab_1_tensor = torch.tensor(np.array(X_scaled_lab_1), dtype=torch.float32).to(device)


class MLP_Lab_1(nn.Module):
    def __init__(self, input_dim):
        super(MLP_Lab_1, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            #nn.Dropout(0.1),

            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.model(x)

model_lab_1 = MLP_Lab_1(X_lab_1_tensor.shape[1]).to(device)
model_lab_1.load_state_dict(torch.load("best_full_model_lab_1.pt", map_location=device))
model_lab_1.eval()
# Inference
with torch.no_grad():
    predictions = model_lab_1(X_lab_1_tensor).squeeze().cpu().numpy()

scaler_path=os.path.join(code_folder, "y_scaler_lab_1.pkl")
y_scaler_lab_1 = joblib.load(scaler_path)
predictions = y_scaler_lab_1.inverse_transform(predictions.reshape(-1, 1)).flatten()
uid_scores = defaultdict(list)
for uid, pred in zip(uids, predictions):
    uid_scores[uid].append(pred)

#pprint(dict(uid_scores))
# === Take average score per UID ===
final_predictions_lab_1 = {uid: np.mean(scores) for uid, scores in uid_scores.items()}
print("Lab 1 predictions")
print(final_predictions_lab_1)

# === Prepare input for model ===
X_lab_2 = task_infos["lab2"]["features"]
scaler_path=os.path.join(code_folder, "x_scaler_lab_2.pkl")
x_scaler_lab_2 = joblib.load(scaler_path)
X_scaled_lab_2 = x_scaler_lab_2.transform(X_lab_2)  # X must have same feature count as used in fitting
X_lab_2_tensor = torch.tensor(np.array(X_scaled_lab_2), dtype=torch.float32).to(device)


class MLP_Lab_2(nn.Module):
    def __init__(self, input_dim):
        super(MLP_Lab_2, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.model(x)
model_lab_2 = MLP_Lab_2(X_lab_2_tensor.shape[1]).to(device)
model_lab_2.load_state_dict(torch.load("best_full_model_lab_2.pt", map_location=device))
model_lab_2.eval()
# Inference
with torch.no_grad():
    predictions = model_lab_1(X_lab_2_tensor).squeeze().cpu().numpy()

scaler_path=os.path.join(code_folder, "y_scaler_lab_2.pkl")
y_scaler_lab_2 = joblib.load(scaler_path)
predictions = y_scaler_lab_2.inverse_transform(predictions.reshape(-1, 1)).flatten()
uid_scores = defaultdict(list)
for uid, pred in zip(uids, predictions):
    uid_scores[uid].append(pred)

#pprint(dict(uid_scores))
# === Take average score per UID ===
final_predictions_lab_2 = {uid: (np.mean(scores)) for uid, scores in uid_scores.items()}
print("Lab 2 predictions")
print(final_predictions_lab_2)


quiz1_r2_score_path = os.path.join(dataset_folder, "final_cv_quiz1_results.txt")
quiz2_r2_score_path = os.path.join(dataset_folder, "final_cv_quiz2_results.txt")
lab1_r2_score_path = os.path.join(dataset_folder, "final_cv_lab1_results.txt")
lab2_r2_score_path = os.path.join(dataset_folder, "final_cv_lab2_results.txt")

# Read r2 scores
quiz1_r2_score=0
with open(quiz1_r2_score_path, "r") as f:
    for line in f:
        if line.startswith("Average R2"):
            quiz1_r2_score = float(line.strip().split(":")[1])
quiz2_r2_score=0
with open(quiz2_r2_score_path, "r") as f:
    for line in f:
        if line.startswith("Average R2"):
            quiz2_r2_score = float(line.strip().split(":")[1])
lab1_r2_score=0
with open(lab1_r2_score_path, "r") as f:
    for line in f:
        if line.startswith("Average R2"):
            lab1_r2_score = float(line.strip().split(":")[1])
lab2_r2_score=0
with open(lab2_r2_score_path, "r") as f:
    for line in f:
        if line.startswith("Average R2"):
            lab2_r2_score = float(line.strip().split(":")[1])


print("✅ quiz1_r2_score =", quiz1_r2_score)
print("✅ quiz2_r2_score =", quiz2_r2_score)
print("✅ lab1_r2_score =", lab1_r2_score)
print("✅ lab2_r2_score =", lab2_r2_score)

sum_r2_score=quiz1_r2_score+quiz2_r2_score+lab1_r2_score+lab2_r2_score
quiz1_weight=quiz1_r2_score/sum_r2_score
quiz2_weight=quiz2_r2_score/sum_r2_score
lab1_weight=lab1_r2_score/sum_r2_score
lab2_weight=lab2_r2_score/sum_r2_score

# Final weighted prediction
final_scores = {}

uids_all = inference_df["UID"].tolist()

for uid in uids_all:
    score_components = []
    weight_components = []

    if uid in final_predictions_quiz_1:
        score_components.append(final_predictions_quiz_1[uid] * quiz1_weight)
        weight_components.append(quiz1_weight)

    if uid in final_predictions_quiz_2:
        score_components.append(final_predictions_quiz_2[uid] * quiz2_weight)
        weight_components.append(quiz2_weight)

    if uid in final_predictions_lab_1:
        score_components.append(final_predictions_lab_1[uid] * lab1_weight)
        weight_components.append(lab1_weight)

    if uid in final_predictions_lab_2:
        score_components.append(final_predictions_lab_2[uid] * lab2_weight)
        weight_components.append(lab2_weight)

    if weight_components:
        final_score = sum(score_components) / sum(weight_components)
    else:
        final_score = 1  # fallback default score if nothing predicted

    final_scores[uid] = int(round(final_score))

# Print or save
print("\n🎯 Final Weighted Predictions:")
print(final_scores)

# Optional: save to CSV
final_pred_df = pd.DataFrame({"UID": list(final_scores.keys()), "MidtermClass": list(final_scores.values())})
final_pred_df.to_csv(os.path.join(dataset_folder, f"final_predictions_{next_prediction_index}.csv"), index=False)
print(f"\n✅ Saved to: final_predictions_{next_prediction_index}.csv")

"""
# === Save to DataFrame ===
result_df = pd.DataFrame({
    "UID": list(final_predictions.keys()),
    "MidtermClass": list(final_predictions.values())
})

# === Save to CSV ===
output_path = os.path.join(dataset_folder, f"prediction_{next_prediction_index}_predicted_scores.csv")
result_df.to_csv(output_path, index=False)
print(f"✅ Saved predicted scores to: {output_path}")

prediction_index=next_prediction_index
with open(prediction_index_path, "w") as f:
    f.write(f"prediction_index={prediction_index}\n")
print(f"✅ Prediction index is updated as {prediction_index} and written to : {prediction_index_path}")
"""