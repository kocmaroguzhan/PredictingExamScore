# Re-import required libraries due to environment reset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import pandas as pd
import os
import ast


# Flatten embedding and validity rows
def flatten_embeddings_with_valid_flag(row, emb_cols, valid_cols):
    emb_vectors = [np.array(row[col]) for col in emb_cols]
    valid_flags = np.array([row[col] for col in valid_cols], dtype=np.float32)  # ensure it's a 1D array
    return np.concatenate(emb_vectors + [valid_flags])


import numpy as np

def generate_noisy_samples(
        X_clean: np.ndarray,
        embedding_dim: int,
        noise_std: float = 0.01
    ) -> np.ndarray:
    """
    Add Gaussian noise to the *embedding portion* of X_clean.

    Parameters
    ----------
    X_clean : np.ndarray
        Shape (n_samples, total_feature_dim).
        Columns 0‥embedding_dim-1 hold concatenated embeddings,
        columns embedding_dim‥ end hold validity flags (ints 0/1).
    embedding_dim : int
        How many columns at the start of X_clean are embeddings.
    noise_std : float, default 0.01
        Standard deviation of the additive Gaussian noise.

    Returns
    -------
    X_noisy : np.ndarray
        Same shape as X_clean.  The first `embedding_dim` columns are
        noisy + re-normalised to unit length per sample, the rest are
        unchanged validity flags.
    """
    X_noisy = X_clean.copy()
    mean=0.0
    # --- 1. Add noise only to the embedding slice --------------------------
    noise = np.random.normal(loc=mean, scale=noise_std,
                             size=(X_clean.shape[0], embedding_dim))
    X_noisy[:, :embedding_dim] += noise

    # --- 2. Re-normalise each embedding vector to unit length -------------
    norms = np.linalg.norm(X_noisy[:, :embedding_dim], axis=1, keepdims=True)
    X_noisy[:, :embedding_dim] /= np.clip(norms, 1e-8, None) ##np.clip to prevent by division by zero

    # validity-flag columns remain unchanged
    return X_noisy




# Combine clean + noisy features
def merge_original_and_noise_samples(X_train,X_train_noisy,y_train):
    X_train_combined = np.concatenate([X_train, X_train_noisy], axis=0)
    y_train_combined = np.concatenate([y_train, y_train], axis=0)
    return (X_train_combined,y_train_combined)

# -------------------------
# Load and Prepare Data
# -------------------------
parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
dataset_folder = os.path.join(parent_folder, "Dataset")
config_path = os.path.join(dataset_folder, "config.txt")
augmentation_count=0
model_embedding_size = 0
number_of_augmented_code_saving_threshold=0
num_of_embeddings=0
# Read config value
with open(config_path, "r") as f:
    for line in f:
        if line.startswith("augmentation_count"):
            augmentation_count = int(line.strip().split("=")[1])
        if line.startswith("model_embedding_size"):
            model_embedding_size = int(line.strip().split("=")[1])
        if line.startswith("number_of_augmented_code_saving_threshold"):
            number_of_augmented_code_saving_threshold = int(line.strip().split("=")[1])
        if line.startswith("num_of_embeddings"):
            num_of_embeddings = int(line.strip().split("=")[1])

print("✅ augmentation_count =", augmentation_count)
print("✅ model_embedding_size =", model_embedding_size)
print("✅ number_of_augmented_code_saving_threshold =", number_of_augmented_code_saving_threshold)
print("✅ num_of_embeddings =", num_of_embeddings)
labeled_df=pd.read_csv(os.path.join(dataset_folder, "labeled_data.csv"))
# Identify embedding columns
embedding_original_cols = [col for col in labeled_df.columns if "_original" in col and not col.endswith("_valid")]
embedding_augmented_cols = [col for col in labeled_df.columns if "_augmented" in col and not col.endswith("_valid")]
# Get original_valid columns
original_valid_cols = [col for col in labeled_df.columns if "_original_valid" in col]
# Convert stringified vectors into lists
for col in embedding_original_cols + embedding_augmented_cols:
    labeled_df[col] = labeled_df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)


# Original dataset
original_df = labeled_df.copy()
original_df["features"] = labeled_df.apply(
    lambda row: flatten_embeddings_with_valid_flag(row, embedding_original_cols, original_valid_cols), axis=1)


#Augmented dataset
augmented_rows = []

for _, row in labeled_df.iterrows():
    for i in range(augmentation_count):  
        aug_cols = [col.replace("original", f"augmented_{i}") for col in embedding_original_cols]
        valid_cols = [col.replace("original", f"augmented_{i}") + "_valid" for col in embedding_original_cols]
        
        # Proceed only if all required columns are present
        if all(col in labeled_df.columns for col in aug_cols + valid_cols):
            # Get embedding vectors
            vecs = [
                np.array(ast.literal_eval(row[col]) if isinstance(row[col], str) else row[col])
                for col in aug_cols
            ]
            # Get validity flags
            valid_flags = [int(row[col]) for col in valid_cols]
            
            # Concatenate embeddings and validity flags
            features = np.concatenate(vecs + [np.array(valid_flags)])
            
            augmented_rows.append({
                "UID": row["UID"],
                "MidtermClass": row["MidtermClass"],
                "features": features
            })

augmented_df = pd.DataFrame(augmented_rows)

# Prepare matrices
original_X = np.stack(original_df["features"].values)
original_y = original_df["MidtermClass"].values

# Prepare training pool (only augmented samples)
X_aug = np.stack(augmented_df["features"].values)
y_aug = augmented_df["MidtermClass"].values

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"💻 Using device: {device}")
# Validation set → original only
X_val = original_X
y_val = original_y

# Training set → augmented only
X_train = X_aug
y_train = y_aug

embedding_dim=model_embedding_size*num_of_embeddings
#Add noise to training samples
X_train_noisy = generate_noisy_samples(X_train,embedding_dim=embedding_dim)
X_train, y_train = merge_original_and_noise_samples(X_train, X_train_noisy, y_train)

print("Training Feature matrix shape:", X_train.shape)
print("Number of samples for training:", X_train.shape[0])
print("Feature dimension (per sample):", X_train.shape[1])
print("Number of samples for validation:", y_val.shape[0])

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

batch_size = 32  # You can tune this
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
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
input_dim = X_train.shape[1]
model = MLP(input_dim).to(device)  # ✅ Send model to GPU
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
epochs = 500
val_r2_history=[]
# Early stopping parameters
early_stop_r2_threshold=0.8
trigger_count=0
patience=20
best_val_r2 = float("-inf")
best_model_state = None
##Training loop
for epoch in range(epochs):
    model.train()
    train_preds, train_targets = [], []
    for batch_X, batch_y in train_loader:
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Store predictions and targets for training metrics
        train_preds.append(outputs.detach().cpu().numpy())
        train_targets.append(batch_y.detach().cpu().numpy())
    # ===========================
    # Validation phase
    # ===========================
    model.eval()
    val_preds, val_targets = [], []

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            val_preds.append(outputs.cpu().numpy())
            val_targets.append(batch_y.cpu().numpy())

    # ===========================
    # Compute Metrics
    # ===========================
    train_preds = np.vstack(train_preds)
    train_targets = np.vstack(train_targets)
    val_preds = np.vstack(val_preds)
    val_targets = np.vstack(val_targets)

    train_mse = mean_squared_error(train_targets, train_preds)
    train_r2 = r2_score(train_targets, train_preds)
    val_mse = mean_squared_error(val_targets, val_preds)
    val_r2 = r2_score(val_targets, val_preds)

    print(f"📆 Epoch {epoch+1}/{epochs}")
    print(f"🧠 Train MSE: {train_mse:.4f} | R²: {train_r2:.4f}")
    print(f"🎯 Val   MSE: {val_mse:.4f} | R²: {val_r2:.4f}")
    print("-" * 50)
    if val_r2 > best_val_r2:
        best_val_r2 = val_r2
        best_model_state = model.state_dict()  # Save current model weights
        print(f"💾 New best model saved at epoch {epoch+1} with R² = {val_r2:.4f}")
    val_r2_history.append(val_r2)
    # Early stopping check
    if epoch >= 10:
        r2_now = val_r2_history[-1] ##last r2 score on validation
        if r2_now > early_stop_r2_threshold:
            trigger_count += 1
            print(f"⚠️ Epoch {epoch+1}: Validation R² is higher than {early_stop_r2_threshold} ({trigger_count}/{patience})")
            if trigger_count >= patience:
                print(f"⛔ Early stopping at epoch {epoch+1}:  Validation R² is higher than {early_stop_r2_threshold} for {patience} epochs.")
                break
        else:
            trigger_count = 0  # reset r2 score is dropped

            
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f"✅ Best model loaded with R² = {best_val_r2:.4f}")    
    torch.save(best_model_state, "best_mlp_model.pt")
    print("📁 Best model saved to 'best_mlp_model.pt'")

