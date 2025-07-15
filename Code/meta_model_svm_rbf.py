import pandas as pd
import numpy as np
import os
import ast
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler


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


# === Parse stringified embeddings in each column ===
for col in ["Quiz1_Pred", "Quiz2_Pred", "Lab1_Pred", "Lab2_Pred"]:
    merged[col] = merged[col].apply(lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else np.array(x))

# === Concatenate the embeddings instead of averaging ===
X = np.concatenate(
    [np.stack(merged[col].to_numpy()) for col in ["Quiz1_Pred", "Quiz2_Pred", "Lab1_Pred", "Lab2_Pred"]],
    axis=1
)
y = merged["True_MidtermClass"].values

# === Standardize features ===
scaler = StandardScaler()
X = scaler.fit_transform(X)

# === Initialize SVR and KFold ===
svr = SVR(kernel="rbf", C=1.0, epsilon=0.2)  # You can tune C and epsilon
kf = KFold(n_splits=5, shuffle=True, random_state=42)

r2_scores = []
mse_scores = []

# === Cross-validation loop ===
for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)       # fit + transform on train
    X_val_scaled = x_scaler.transform(X_val)         # transform only on val

    # Fit on training targets
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).flatten()

    svr.fit(X_train_scaled, y_train_scaled)
    y_pred = svr.predict(X_val_scaled)

    r2 = r2_score(y_val_scaled, y_pred)
    mse = mean_squared_error(y_val_scaled, y_pred)

    r2_scores.append(r2)
    mse_scores.append(mse)

    print(f"✅ Fold {fold}: R² = {r2:.4f}, MSE = {mse:.4f}")

# === Final results ===
print("\n📊 Final SVR Results:")
print("Average R²:", np.mean(r2_scores))
print("Average MSE:", np.mean(mse_scores))
