# unified_mlp_kfold_cv.py

import os
import ast
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error

# ✅ Set seed for reproducibility
np.random.seed(42)
method="multiply"
# =====================
# CONFIGURATION
# =====================
assesment_names=["Quiz1","Quiz2","Lab1","Lab2"]
parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
dataset_folder = os.path.join(parent_folder, "Dataset")
labeled_data_file = os.path.join(dataset_folder, "labeled_data.csv")
question_embeddings_data_file=os.path.join(dataset_folder, "question_embeddings.csv")
batch_size_quiz_1 = 4
batch_size_quiz_2 = 256
batch_size_lab_1 = 256
batch_size_lab_2 = 256
batch_size_dict={
                "Quiz1":batch_size_quiz_1,
                "Quiz2":batch_size_quiz_2,
                "Lab1":batch_size_lab_1,
                "Lab2":batch_size_lab_2
                }
lr_quiz_1 = 0.005
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

def substract_question_answer_embeddings(original_df, question_embeddings_df, assesment_name):
    """
    Returns the element-wise difference between question embeddings and student answer embeddings.

    Parameters:
    - original_df: DataFrame with student records and assessment section info
    - question_embeddings_df: Single-row DataFrame with question embedding columns
    - assesment_name: One of ["Quiz1", "Quiz2", "Lab1", "Lab2"]

    Returns:
    - diff_features: np.ndarray where each row is (question_embedding - student_answer_embedding)
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

    # ✅ Return A - B (question - answer)
    return question_emb_matrix - student_answer_features

def add_section_info(df, original_df, assesment_name):
    section_col = "quiz_assessment_section" if "Quiz" in assesment_name else "lab_assessment_section"
    df[section_col] = df["UID"].map(original_df.set_index("UID")[section_col])
    return df

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

from sklearn.pipeline import Pipeline

kf = KFold(n_splits=3, shuffle=True, random_state=42)

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
    augmented_df = add_section_info(augmented_df, original_df, assesment_name)

    # Difference of question and answer embeddings
    question_answer_diff_original_X = substract_question_answer_embeddings(original_df, question_embeddings_df, assesment_name)

    original_y = original_df["MidtermClass"].values
    original_uids = original_df["UID"].values

    fold = 1
    r2_scores = []
    mse_scores = []
    oof_df_list = []

    for train_idx, val_idx in kf.split(question_answer_diff_original_X):
        print(f"\n====== RidgeCV + PCA Fold {fold} ======")

        # Split features and labels
        X_train, X_val = question_answer_diff_original_X[train_idx], question_answer_diff_original_X[val_idx]
        y_train, y_val = original_y[train_idx], original_y[val_idx]
        uid_val = original_uids[val_idx]

        # Standardize targets
        y_scaler = StandardScaler()
        y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).flatten()

        # Pipeline: StandardScaler → PCA → RidgeCV
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.95)),  # Retain 95% variance
            ("ridge", RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 150.0], scoring="neg_mean_squared_error", cv=3))
        ])

        # Fit pipeline on training data
        pipeline.fit(X_train, y_train_scaled)

        # Predict on validation data
        val_preds_scaled = pipeline.predict(X_val)
        val_preds_original = y_scaler.inverse_transform(val_preds_scaled.reshape(-1, 1)).flatten()

        # Evaluate
        val_mse = mean_squared_error(y_val, val_preds_original)
        val_r2 = r2_score(y_val, val_preds_original)
        best_alpha = pipeline.named_steps["ridge"].alpha_

        print(f"🎯 RidgeCV + PCA Val MSE: {val_mse:.4f} | R²: {val_r2:.4f} | Alpha: {best_alpha}")

        # Save results
        r2_scores.append(val_r2)
        mse_scores.append(val_mse)

        fold_df = pd.DataFrame({
            "UID": uid_val,
            "True_MidtermClass": y_val,
            f"{assesment_name}_RidgeCV_PCA_Pred": val_preds_original
        })
        oof_df_list.append(fold_df)

        fold += 1

    # Summary
    print(f"\n📊 RidgeCV + PCA Final CV Results for {assesment_name}")
    print("Average R²:", np.mean(r2_scores))
    print("Average MSE:", np.mean(mse_scores))

    # Save predictions
    #oof_all_df = pd.concat(oof_df_list)
    #ridge_output_path = os.path.join(dataset_folder, f"ridgecv_pca_oof_preds_{assesment_name}.csv")
    #oof_all_df.to_csv(ridge_output_path, index=False)
