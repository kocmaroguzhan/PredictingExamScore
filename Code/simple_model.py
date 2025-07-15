import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import os 

parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
dataset_folder = os.path.join(parent_folder, "Dataset")
quiz1_predictions_path = os.path.join(dataset_folder, "oof_preds_quiz1.csv")
quiz2_predictions_path = os.path.join(dataset_folder, "oof_preds_quiz2.csv")
lab1_predictions_path = os.path.join(dataset_folder, "oof_preds_lab1.csv")
lab2_predictions_path = os.path.join(dataset_folder, "oof_preds_lab2.csv")

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

# === Compute mean prediction across all models ===
prediction_cols = ["Quiz1_Pred", "Quiz2_Pred", "Lab1_Pred", "Lab2_Pred"]

# If predictions are stored as strings (e.g., "[1.23]"), convert them to float
for col in prediction_cols:
    merged[col] = merged[col].apply(lambda x: float(eval(x)) if isinstance(x, str) else x)


merged["Final_Prediction"] = (
    merged["Quiz1_Pred"] * quiz1_weight +
    merged["Quiz2_Pred"] * quiz2_weight +
    merged["Lab1_Pred"]  * lab1_weight +
    merged["Lab2_Pred"]  * lab2_weight
)


# Evaluate performance

r2 = r2_score(merged["Target"], merged["Final_Prediction"])
mse = mean_squared_error(merged["Target"], merged["Final_Prediction"])

print(f"✅ Final Averaged Model:")
print(f"   - R² Score: {r2:.4f}")
print(f"   - MSE: {mse:.4f}")