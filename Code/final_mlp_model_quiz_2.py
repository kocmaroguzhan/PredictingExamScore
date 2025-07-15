
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import os
import ast
# ========== Final Training on Full Dataset ==========
print("\n🚀 Training Final Model on Full Dataset...")


# ========== Configurable Parameters ==========
batch_size = 256
embedding_dim = 0  # Will be read from config
alpha_start = 0.1
alpha_stop = 1.0
alpha_step_size = 0.1
epochs = 100

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
        valid_cols = [col.replace("original", f"augmented_{i}") + "_valid" for col in embedding_original_cols]
        if all(col in labeled_df.columns for col in aug_cols + valid_cols):
            vecs = [np.array(ast.literal_eval(row[col]) if isinstance(row[col], str) else row[col]) for col in aug_cols]
            valid_flags = [int(row[col]) for col in valid_cols]
            features = np.concatenate(vecs)
            augmented_rows.append({"UID": row["UID"], "MidtermClass": row["MidtermClass"], "features": features})
augmented_df = pd.DataFrame(augmented_rows)

original_X = np.stack(original_df["features"].values)
original_y = original_df["MidtermClass"].values
original_uids = original_df["UID"].values

# Prepare all training data
uid_train_set = set(original_uids)
augmented_df = augmented_df[augmented_df["UID"].isin(uid_train_set)]
X_train_aug = np.stack(augmented_df["features"].values)
y_train_aug = augmented_df["MidtermClass"].values

# Mixup on original
mixup_data = generate_mixup_data_variants(original_X, original_y, embedding_dim, alpha_start, alpha_stop, alpha_step_size)
X_mixup_final, y_mixup_final = combine_data_array_list(mixup_data)
X_mixup_noisy = generate_noisy_samples(X_mixup_final, embedding_dim)
X_mixup_combined, y_mixup_combined = merge_original_and_noise_samples(X_mixup_final, X_mixup_noisy, y_mixup_final)

# Noise on augmented
X_train_aug_noisy = generate_noisy_samples(X_train_aug, embedding_dim)
X_train_aug_combined, y_train_aug_combined = merge_original_and_noise_samples(X_train_aug, X_train_aug_noisy, y_train_aug)

# Merge all
X_train_final, y_train_final = merge_samples(X_mixup_combined, X_train_aug_combined, y_mixup_combined, y_train_aug_combined)

# Normalize input features
x_scaler = StandardScaler()
X_train_final = x_scaler.fit_transform(X_train_final)

# Normalize target
y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train_final.reshape(-1, 1)).flatten()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Convert to tensor
X_train_tensor = torch.tensor(X_train_final, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).view(-1, 1).to(device)

train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)

# Train final model
model = MLP(X_train_tensor.shape[1]).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-4)
criterion = nn.MSELoss()

best_loss = float("inf")
best_model_state = None

print("🧠 Starting training on full data...")

for epoch in range(epochs):
    train_preds, train_targets = [], []
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        # Collect scaled predictions and targets
        train_preds.append(outputs.detach().cpu().numpy())
        train_targets.append(batch_y.detach().cpu().numpy())

    # Stack all batches
    train_preds = np.vstack(train_preds)
    train_targets = np.vstack(train_targets)

    # 🔁 Inverse-transform both
    train_preds_original = y_scaler.inverse_transform(train_preds)
    train_targets_original = y_scaler.inverse_transform(train_targets)

    # 🎯 Compute true MSE
    epoch_loss = mean_squared_error(train_targets_original, train_preds_original)


    if epoch_loss < best_loss:
        best_loss = epoch_loss
        best_model_state = model.state_dict()

    if epoch % 10 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch+1}/{epochs} - MSE Loss: {epoch_loss:.4f}")

# Save model and scalers
torch.save(best_model_state, "best_full_model_quiz_2.pt")
import joblib
joblib.dump(x_scaler, "x_scaler_quiz_2.pkl")
joblib.dump(y_scaler, "y_scaler_quiz_2.pkl")

print(f"✅ Final model saved as 'best_full_model.pt'")
print(f"📦 Scalers saved as 'x_scaler.pkl' and 'y_scaler.pkl'")
