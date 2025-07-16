# unified_mlp_kfold_cv.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import ast
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error

# ✅ Set seed for reproducibility
np.random.seed(42)
method="merge"
# =====================
# CONFIGURATION
# =====================
assesment_names=["Quiz1","Quiz2","Lab1","Lab2"]
parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
dataset_folder = os.path.join(parent_folder, "Dataset")
labeled_data_file = os.path.join(dataset_folder, "labeled_data.csv")
question_embeddings_data_file=os.path.join(dataset_folder, "question_embeddings.csv")
batch_size_quiz_1 = 128
batch_size_quiz_2 = 256
batch_size_lab_1 = 256
batch_size_lab_2 = 256
batch_size_dict={
                "Quiz1":batch_size_quiz_1,
                "Quiz2":batch_size_quiz_2,
                "Lab1":batch_size_lab_1,
                "Lab2":batch_size_lab_2
                }
lr_quiz_1 = 0.01
lr_quiz_2 = 0.1
lr_lab_1 = 0.1
lr_lab_2 = 0.1

lr_dict={
                "Quiz1":lr_quiz_1,
                "Quiz2":lr_quiz_2,
                "Lab1":lr_lab_1,
                "Lab2":lr_lab_2
                }
embedding_dim = 0  # Will be read from config
alpha_start = 0.1
alpha_stop = 1.0
alpha_step_size = 0.1
patience = 20
epochs = 200
early_stop_r2_threshold = 0.9
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


# =====================
#FUNCTION DECLARATIONS

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


# Combine mix up  features
def merge_samples(X_sample_0,X_sample_1,y_sample_0,y_sample_1):
    X_combined = np.concatenate([X_sample_0, X_sample_1], axis=0)
    y_combined = np.concatenate([y_sample_0, y_sample_1], axis=0)
    return (X_combined,y_combined)

def generate_augmented_df(labeled_df, base_cols, feature_name):
    rows = []
    for _, row in labeled_df.iterrows():
        for i in range(augmentation_count):
            aug_cols = [col.replace("original", f"augmented_{i}") for col in base_cols]
            if all(col in labeled_df.columns for col in aug_cols):
                vecs = [np.array(ast.literal_eval(row[col]) if isinstance(row[col], str) else row[col]) for col in aug_cols]
                features = np.concatenate(vecs)
                rows.append({
                    "UID": row["UID"],
                    "MidtermClass": row["MidtermClass"],
                    feature_name: features
                })
    return pd.DataFrame(rows)

def apply_flatten_embeddings(df, target_df, col_list, feature_name):
    target_df[feature_name] = df.apply(lambda row: flatten_embeddings(row, col_list), axis=1)

def merge_question_answer_embeddings(original_df, question_embeddings_df, assesment_name):
    """
    Merges the corresponding question embeddings to the student's answer features.

    Parameters:
    - original_df: DataFrame with student records and assessment section info
    - question_embeddings_df: Single-row DataFrame with question embedding columns
    - assesment_name: One of ["Quiz1", "Quiz2", "Lab1", "Lab2"]

    Returns:
    - merged_features: np.ndarray with [answer features | question embedding]
    """

    if assesment_name in ["Quiz1", "Quiz2"]:
        section_col = "quiz_assessment_section"
    else:
        section_col = "lab_assessment_section"

    student_answer_features = np.stack(original_df[f"{assesment_name}_features"].values)
    section_names = original_df[section_col].values

    question_embeddings = []
    for section in section_names:
        value = question_embeddings_df.at[0, section]  # single-row lookup
        if isinstance(value, str):
            emb = np.array(ast.literal_eval(value))
        else:
            emb = np.array(value)
        question_embeddings.append(emb)

    question_emb_matrix = np.vstack(question_embeddings)
    return np.concatenate([question_emb_matrix,student_answer_features ], axis=1)

def add_section_info(df, original_df, assesment_name):
    section_col = "quiz_assessment_section" if "Quiz" in assesment_name else "lab_assessment_section"
    df[section_col] = df["UID"].map(original_df.set_index("UID")[section_col])
    return df
class MLP_quiz_1(nn.Module):
    def __init__(self, input_dim):
        super(MLP_quiz_1, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(0.2),

            #nn.Linear(256, 256),
            #nn.BatchNorm1d(256),
            #nn.ReLU(),
            #nn.Dropout(0.1),

            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.model(x)

class MLP_quiz_2(nn.Module):
    def __init__(self, input_dim):
        super(MLP_quiz_2, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            #nn.Dropout(0.1),

            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.model(x)

class MLP_lab_1(nn.Module):
    def __init__(self, input_dim):
        super(MLP_lab_1, self).__init__()
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

class MLP_lab_2(nn.Module):
    def __init__(self, input_dim):
        super(MLP_lab_2, self).__init__()
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

model_dict={
        "Quiz1": MLP_quiz_1,
        "Quiz2": MLP_quiz_2,
        "Lab1": MLP_lab_1,
        "Lab2": MLP_lab_2
}
# =====================
# LOAD DATA
# =====================
print("📥 Loading data...")
labeled_df = pd.read_csv(labeled_data_file)
question_embeddings_df= pd.read_csv(question_embeddings_data_file)
embedding_original_cols = [col for col in labeled_df.columns if "_original" in col and not col.endswith("_valid")]
#original_valid_cols = [col for col in labeled_df.columns if "_original_valid" in col ]
for col in embedding_original_cols:
    labeled_df[col] = labeled_df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

original_df = labeled_df.copy()

# =====================
# TRAIN + EVALUATE
# =====================

kf = KFold(n_splits=3, shuffle=True, random_state=42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


for assesment_name in assesment_names:

    # Get columns and features
    embedding_assesment_original_cols = [
        col for col in labeled_df.columns
        if "_original" in col and assesment_name in col and not col.endswith("_valid")
    ]
    # Apply flatten
    apply_flatten_embeddings(labeled_df, original_df, embedding_assesment_original_cols, f"{assesment_name}_features")
    # Generate augmented DataFrame
    augmented_df = generate_augmented_df(labeled_df, embedding_assesment_original_cols, f"{assesment_name}_features")
    # Add question section to augmented_df
    augmented_df = add_section_info(augmented_df, original_df, assesment_name)
    question_answer_concat_original_X = merge_question_answer_embeddings(original_df, question_embeddings_df, assesment_name)
    question_answer_concat_aug_X = merge_question_answer_embeddings(augmented_df, question_embeddings_df, assesment_name)
    original_y = original_df["MidtermClass"].values
    original_uids = original_df["UID"].values
    augmented_uid_array = augmented_df["UID"].values
    augmented_y_array = augmented_df["MidtermClass"].values
    fold = 1
    r2_scores = []
    mse_scores = []

    oof_df_list=[]#Out of fold list which is validation set for current fold

    for train_idx, val_idx in kf.split(question_answer_concat_original_X):
        print(f"\n====== Fold {fold} ======")
        X_train_original, X_val_original = question_answer_concat_original_X[train_idx], question_answer_concat_original_X[val_idx]
        y_train_original, y_val_original = original_y[train_idx], original_y[val_idx]
        X_train_original_noisy = generate_noisy_samples(X_train_original, embedding_dim,noise_std=0.0001)
        X_train_original_combined, y_train_original_combined = merge_samples(X_train_original, X_train_original_noisy, y_train_original,y_train_original)
        uid_train_original = original_uids[train_idx]

        # Create a mask for training fold UIDs
        train_aug_mask = np.isin(augmented_uid_array, uid_train_original)
        # Apply filtering
        X_train_aug = question_answer_concat_aug_X[train_aug_mask]
        y_train_aug = augmented_y_array[train_aug_mask]

        mixup_data = generate_mixup_data_variants(X_train_original, y_train_original, embedding_dim, alpha_start, alpha_stop, alpha_step_size)
        X_mixup_final, y_mixup_final = combine_data_array_list(mixup_data)
        X_mixup_noisy = generate_noisy_samples(X_mixup_final, embedding_dim)
        X_mixup_combined, y_mixup_combined = merge_samples(X_mixup_final, X_mixup_noisy, y_mixup_final,y_mixup_final)

        X_train_aug_noisy = generate_noisy_samples(X_train_aug, embedding_dim)
        X_train_aug_combined, y_train_aug_combined = merge_samples(X_train_aug, X_train_aug_noisy, y_train_aug,y_train_aug)
        #X_train_final,y_train_final=merge_samples(X_mixup_combined,X_train_aug_combined,y_mixup_combined,y_train_aug_combined)
        #X_train_final,y_train_final=merge_samples(X_train_original,X_train_aug,y_train_original,y_train_aug)
        #X_train_final, y_train_final = shuffle(X_train_final, y_train_final, random_state=46)
        #X_train_final=X_train_original
        #y_train_final=y_train_original
        X_train_final,y_train_final=merge_samples(X_train_original,X_train_aug,y_train_original,y_train_aug)
        # Normalize input features (fit only on training data of the current fold)
        x_scaler = StandardScaler()
        X_train_scaled = x_scaler.fit_transform(X_train_final)       # fit + transform on train
        X_val_scaled = x_scaler.transform(X_val_original)         # transform only on val

        # Fit on training targets
        y_scaler = StandardScaler()
        y_train_scaled = y_scaler.fit_transform(y_train_final.reshape(-1, 1)).flatten()
        y_val_scaled = y_scaler.transform(y_val_original.reshape(-1, 1)).flatten()
        
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).view(-1, 1).to(device)
        X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32).view(-1, 1).to(device)
        print("Training Feature matrix shape:", X_train_scaled.shape)
        print("Number of samples for training:", X_train_scaled.shape[0])
        print("Feature dimension (per sample):", X_train_scaled.shape[1])
        print("Number of samples for validation:", y_val_scaled.shape[0])

        print("📊 y_train_final:")
        print("  - Min:", np.min(y_train_final))
        print("  - Max:", np.max(y_train_final))

        print("📊 y_val_original:")
        print("  - Min:", np.min(y_val_original))
        print("  - Max:", np.max(y_val_original))

        train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size = batch_size_dict[assesment_name], shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size = batch_size_dict[assesment_name], shuffle=False)

        model = model_dict[assesment_name](X_train_tensor.shape[1]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr_dict[assesment_name], weight_decay=1e-2)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
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
            # 🔍 Print every 10 epochs or first epoch
            if (epoch + 1) % 10 == 0 or epoch == 0:
                model.eval()
                train_preds = []
                train_targets = []
                with torch.no_grad():
                    for batch_X, batch_y in train_loader:
                        pred = model(batch_X)
                        train_preds.append(pred.cpu().numpy())
                        train_targets.append(batch_y.cpu().numpy())

                train_preds = np.vstack(train_preds)
                train_targets = np.vstack(train_targets)

                train_preds_original = y_scaler.inverse_transform(train_preds)
                train_targets_original = y_scaler.inverse_transform(train_targets)

                train_mse = mean_squared_error(train_targets_original, train_preds_original)
                train_r2 = r2_score(train_targets_original, train_preds_original)
                current_lr = scheduler.get_last_lr()[0]
                print(f"📉 Epoch {epoch+1:03d} | Train MSE: {train_mse:.4f} | Train R²: {train_r2:.4f} | Val MSE: {val_mse:.4f} | Val R²: {val_r2:.4f}")
                print(f"Epoch {epoch+1:03d} | LR: {current_lr:.6f}")
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
            scheduler.step()
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
        fold_df[f"{assesment_name}_Pred"] = val_preds_original.flatten()  # Saving final prediction

        oof_df_list.append(fold_df)


    average_r2=np.mean(r2_scores)
    print(f"\n📊 Final Cross-Validation Results for : {assesment_name}")
    print("Average R²:", average_r2)
    print("Average MSE:", np.mean(mse_scores))

    results_path = os.path.join(dataset_folder, f"{method}_final_cv_{assesment_name}_results.txt")
    with open(results_path, "w") as f:
        f.write(f"Average R2: {average_r2:.4f}\n")
        
    oof_all_df = pd.concat(oof_df_list)
    output_csv_path = os.path.join(dataset_folder, f"{method}_oof_preds_{assesment_name}.csv")
    oof_all_df.to_csv(output_csv_path, index=False)