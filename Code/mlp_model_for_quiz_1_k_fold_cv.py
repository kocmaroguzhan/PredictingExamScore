# Re-import required libraries due to environment reset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold
import json
import numpy as np
import pandas as pd
import os
import ast

# ========== Configurable Parameters ==========
batch_size = 256
embedding_dim = 0  # Will be read from config
alpha_start = 0.1
alpha_stop = 1.0
alpha_step_size = 0.1
patience = 20
epochs = 100 
early_stop_r2_threshold = 0.9

# ========== Data Augmentation Functions ==========
def flatten_embeddings_with_valid_flag(row, emb_cols, valid_cols):
    emb_vectors = [np.array(row[col]) for col in emb_cols]
    valid_flags = np.array([row[col] for col in valid_cols], dtype=np.float32)
    return np.concatenate(emb_vectors + [valid_flags])

def flatten_embeddings(row, emb_cols):
    emb_vectors = [np.array(row[col]) for col in emb_cols]
    return np.concatenate(emb_vectors)


def mixup_data_augmentation(X, y, embedding_dim, alpha=0.4):
    if alpha <= 0:
        return X, y
    lam = np.random.beta(alpha, alpha)
    indices = np.random.permutation(len(X))
    X_emb = X[:, :embedding_dim]
    X_valid = X[:, embedding_dim:]
    X_mixed_emb = lam * X_emb + (1 - lam) * X_emb[indices]
    X_mixed_valid = np.minimum(X_valid, X_valid[indices])
    X_mixed = np.concatenate([X_mixed_emb, X_mixed_valid], axis=1)
    y_mixed = lam * y + (1 - lam) * y[indices]
    return X_mixed, y_mixed

def generate_mixup_data_variants(X, y, embedding_dim, alpha_start, alpha_stop, alpha_step_size):
    mixup_data_results = []
    for alpha in np.arange(alpha_start, alpha_stop + alpha_step_size / 2, alpha_step_size):
        X_mixed, y_mixed = mixup_data_augmentation(X, y, embedding_dim, alpha=alpha)
        mixup_data_results.append((X_mixed, y_mixed))
    return mixup_data_results

def combine_data_array_list(data_array_list):
    X_all = np.concatenate([pair[0] for pair in data_array_list], axis=0)
    y_all = np.concatenate([pair[1] for pair in data_array_list], axis=0)
    return X_all, y_all

def generate_noisy_samples(X_clean, embedding_dim, noise_std=0.01):
    X_noisy = X_clean.copy()
    noise = np.random.normal(0.0, noise_std, size=(X_clean.shape[0], embedding_dim))
    X_noisy[:, :embedding_dim] += noise
    norms = np.linalg.norm(X_noisy[:, :embedding_dim], axis=1, keepdims=True)
    X_noisy[:, :embedding_dim] /= np.clip(norms, 1e-8, None)
    return X_noisy

def merge_original_and_noise_samples(X_clean, X_noisy, y_clean):
    X_combined = np.concatenate([X_clean, X_noisy], axis=0)
    y_combined = np.concatenate([y_clean, y_clean], axis=0)
    return X_combined, y_combined

# Combine mix up  features
def merge_samples(X_sample_0,X_sample_1,y_sample_0,y_sample_1):
    X_combined = np.concatenate([X_sample_0, X_sample_1], axis=0)
    y_combined = np.concatenate([y_sample_0, y_sample_1], axis=0)
    return (X_combined,y_combined)

import torch.nn as nn
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
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


# ========== Load Config and Dataset ==========
parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
dataset_folder = os.path.join(parent_folder, "Dataset")
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

labeled_df = pd.read_csv(os.path.join(dataset_folder, "labeled_data.csv"))
embedding_original_cols = [col for col in labeled_df.columns if "_original" in col and "Quiz1" in col and not col.endswith("_valid")]
original_valid_cols = [col for col in labeled_df.columns if "_original_valid" in col and "Quiz1" in col]

for col in embedding_original_cols:
    labeled_df[col] = labeled_df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

original_df = labeled_df.copy()
#original_df["features"] = labeled_df.apply(
#    lambda row: flatten_embeddings_with_valid_flag(row, embedding_original_cols, original_valid_cols), axis=1)


original_df["features"] = labeled_df.apply(
    lambda row: flatten_embeddings(row, embedding_original_cols), axis=1)

augmented_rows = []
for _, row in labeled_df.iterrows():
    for i in range(augmentation_count):
        aug_cols = [col.replace("original", f"augmented_{i}") for col in embedding_original_cols]
        #valid_cols = [col.replace("original", f"augmented_{i}") + "_valid" for col in embedding_original_cols]
        if all(col in labeled_df.columns for col in aug_cols):
            vecs = [np.array(ast.literal_eval(row[col]) if isinstance(row[col], str) else row[col]) for col in aug_cols]
            #valid_flags = [int(row[col]) for col in valid_cols]
            features = np.concatenate(vecs)
            augmented_rows.append({"UID": row["UID"], "MidtermClass": row["MidtermClass"], "features": features})
augmented_df = pd.DataFrame(augmented_rows)

original_X = np.stack(original_df["features"].values)
original_y = original_df["MidtermClass"].values
original_uids = original_df["UID"].values

kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold = 1
r2_scores = []
mse_scores = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

oof_df_list=[]#Out of fold list which is validation set for current fold

for train_idx, val_idx in kf.split(original_X):
    print(f"\n====== Fold {fold} ======")
    X_train_original, X_val_original = original_X[train_idx], original_X[val_idx]
    y_train_original, y_val_original = original_y[train_idx], original_y[val_idx]
    uid_train_original = original_uids[train_idx]

    uid_train_set = set(uid_train_original)
    filtered_augmented_df = augmented_df[augmented_df["UID"].isin(uid_train_set)]
    X_train_aug = np.stack(filtered_augmented_df["features"].values)
    y_train_aug = filtered_augmented_df["MidtermClass"].values

    mixup_data = generate_mixup_data_variants(X_train_original, y_train_original, embedding_dim, alpha_start, alpha_stop, alpha_step_size)
    X_mixup_final, y_mixup_final = combine_data_array_list(mixup_data)
    X_mixup_noisy = generate_noisy_samples(X_mixup_final, embedding_dim)
    X_mixup_combined, y_mixup_combined = merge_original_and_noise_samples(X_mixup_final, X_mixup_noisy, y_mixup_final)

    X_train_aug_noisy = generate_noisy_samples(X_train_aug, embedding_dim)
    X_train_aug_combined, y_train_aug_combined = merge_original_and_noise_samples(X_train_aug, X_train_aug_noisy, y_train_aug)
    X_train_final,y_train_final=merge_samples(X_mixup_combined,X_train_aug_combined,y_mixup_combined,y_train_aug_combined)
    #X_train_final, y_train_final = shuffle(X_train_final, y_train_final, random_state=46)
    #X_train_final=X_train_aug_combined
    #y_train_final=y_train_aug_combined
    # Normalize input features (fit only on training data of the current fold)
    scaler = StandardScaler()
    X_train_final = scaler.fit_transform(X_train_final)       # fit + transform on train
    X_val_original = scaler.transform(X_val_original)         # transform only on val

    # Fit on training targets
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train_final.reshape(-1, 1)).flatten()
    y_val_scaled = y_scaler.transform(y_val_original.reshape(-1, 1)).flatten()
    X_train_tensor = torch.tensor(X_train_final, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).view(-1, 1).to(device)
    X_val_tensor = torch.tensor(X_val_original, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32).view(-1, 1).to(device)
    print("Training Feature matrix shape:", X_train_final.shape)
    print("Number of samples for training:", X_train_final.shape[0])
    print("Feature dimension (per sample):", X_train_final.shape[1])
    print("Number of samples for validation:", y_val_original.shape[0])

    print("📊 y_train_final:")
    print("  - Min:", np.min(y_train_final))
    print("  - Max:", np.max(y_train_final))

    print("📊 y_val_original:")
    print("  - Min:", np.min(y_val_original))
    print("  - Max:", np.max(y_val_original))

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False)

    model = MLP(X_train_tensor.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-4)
    criterion = nn.MSELoss()
    best_val_r2 = float("-inf")
    best_model_state = None
    trigger_count = 0
    val_r2_history = []

    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                val_preds.append(outputs.cpu().numpy())
                val_targets.append(batch_y.cpu().numpy())

        val_preds = np.vstack(val_preds)
        val_targets = np.vstack(val_targets)

        # 🔁 Inverse-transform both
        val_preds_original = y_scaler.inverse_transform(val_preds)
        val_targets_original = y_scaler.inverse_transform(val_targets)
        # ✅ Now compute metrics on original scale
        val_mse = mean_squared_error(val_targets_original, val_preds_original)
        val_r2 = r2_score(val_targets_original, val_preds_original)
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_model_state = model.state_dict()
        val_r2_history.append(val_r2)

        if epoch >= 10:
            if val_r2 > early_stop_r2_threshold:
                trigger_count += 1
                if trigger_count >= patience:
                    print(f"⛔ Early stopping at epoch {epoch+1}")
                    break
            else:
                trigger_count = 0

    print(f"✅ Fold {fold} Best R²: {best_val_r2:.4f}")
    r2_scores.append(best_val_r2)
    mse_scores.append(val_mse)
    fold += 1

    model.load_state_dict(best_model_state)
    model.eval()

    val_preds, val_targets = [], []
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)  # this is the final prediction
            val_preds.append(outputs.cpu().numpy())
            val_targets.append(batch_y.cpu().numpy())

    val_preds = np.vstack(val_preds)
    val_targets = np.vstack(val_targets)

    # Inverse-transform if you standardized the targets
    val_preds_original = y_scaler.inverse_transform(val_preds)
    val_targets_original = y_scaler.inverse_transform(val_targets)

    # Save to dataframe
    uids_val = original_uids[val_idx]
    fold_df = pd.DataFrame()
    fold_df["UID"] = uids_val
    fold_df["True_MidtermClass"] = val_targets_original.flatten()
    fold_df["Quiz1_Pred"] = val_preds_original.flatten()  # Saving final prediction

    oof_df_list.append(fold_df)


average_r2=np.mean(r2_scores)
print("\n📊 Final Cross-Validation Results:")
print("Average R²:", average_r2)
print("Average MSE:", np.mean(mse_scores))


results_path = os.path.join(dataset_folder, "final_cv_quiz1_results.txt")
with open(results_path, "w") as f:
    f.write(f"Average R2: {average_r2:.4f}\n")


oof_all_df = pd.concat(oof_df_list)
output_csv_path = os.path.join(dataset_folder, "oof_preds_quiz1.csv")
oof_all_df.to_csv(output_csv_path, index=False)
