import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import ast
import os 

batch_size=4
epochs=100
early_stop_r2_threshold=0.9
patience=10
parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
dataset_folder = os.path.join(parent_folder, "Dataset")
quiz1_predictions_path = os.path.join(dataset_folder, "oof_preds_quiz1.csv")
quiz2_predictions_path = os.path.join(dataset_folder, "oof_preds_quiz2.csv")
lab1_predictions_path = os.path.join(dataset_folder, "oof_preds_lab1.csv")
lab2_predictions_path = os.path.join(dataset_folder, "oof_preds_lab2.csv")
# === Load and merge OOF prediction files ===
quiz1 = pd.read_csv(quiz1_predictions_path)
quiz2 = pd.read_csv(quiz2_predictions_path)
lab1 = pd.read_csv(lab1_predictions_path)
lab2 = pd.read_csv(lab2_predictions_path)

# Drop duplicate label columns before merging to avoid conflicts
quiz2 = quiz2.drop(columns=["True_MidtermClass"])
lab1 = lab1.drop(columns=["True_MidtermClass"])
lab2 = lab2.drop(columns=["True_MidtermClass"])
# === Merge on UID ===
merged = quiz1.merge(quiz2, on="UID") \
              .merge(lab1, on="UID") \
              .merge(lab2, on="UID")

# === Rename for clarity ===
merged.rename(columns={
    "True_MidtermClass": "Target"
}, inplace=True)

# Parse embeddings and convert them to NumPy arrays
for col in ["Quiz1_Pred", "Quiz2_Pred", "Lab1_Pred", "Lab2_Pred"]:
    merged[col] = merged[col].apply(lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else np.array(x))

# Columns that contain the stringified embeddings
embedding_cols = ["Quiz1_Pred", "Quiz2_Pred", "Lab1_Pred", "Lab2_Pred"]
# Stack shape: (num_samples, 4, embedding_dim)
X = np.concatenate([np.stack(merged[col].to_numpy()) for col in ["Quiz1_Pred", "Quiz2_Pred", "Lab1_Pred", "Lab2_Pred"]], axis=1)
y = merged["Target"].values


r2_scores = []
mse_scores = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
# === Train MLPRegressor ===

class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim,16),
            nn.BatchNorm1d(16),     # <- BatchNorm added
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(16,8),
            nn.BatchNorm1d(8),     # <- BatchNorm added
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.model(x)

fold=1
for train_idx, val_idx in kf.split(X):
    print(f"\n====== Fold {fold} ======")
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)       # fit + transform on train
    X_val_scaled = x_scaler.transform(X_val)         # transform only on val

    # Fit on training targets
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).flatten()
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).view(-1, 1).to(device)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32).view(-1, 1).to(device)
    print("Training Feature matrix shape:", X_train_scaled.shape)
    print("Number of samples for training:", X_train_scaled.shape[0])
    print("Feature dimension (per sample):", X_train_scaled.shape[1])
    print("Number of samples for validation:", y_val_scaled.shape[0])

    print("📊 y_train_final:")
    print("  - Min:", np.min(y_train))
    print("  - Max:", np.max(y_train))

    print("📊 y_val_original:")
    print("  - Min:", np.min(y_val))
    print("  - Max:", np.max(y_val))

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False)

    model = MLP(X_train_tensor.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
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



print("\n📊 Final Cross-Validation Results:")
print("Average R²:", np.mean(r2_scores))
print("Average MSE:", np.mean(mse_scores))