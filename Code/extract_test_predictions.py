import pandas as pd
import os

# --- paths -------------------------------------------------
    
parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
dataset_folder = os.path.join(parent_folder, "Dataset")                   
pred_path   = os.path.join(dataset_folder, "exam_prediction.csv")
test_path   = os.path.join(dataset_folder, "processed_test_student.csv")
output_path = os.path.join(dataset_folder, "exam_prediction_test_only.csv")

# --- load files -------------------------------------------
pred_df  = pd.read_csv(pred_path)      # expects columns: UID, MidtermClass
test_df  = pd.read_csv(test_path)      # expects at least a column: UID


# --- merge and fill missing predictions ---------------------
merged = test_df[["UID"]].merge(pred_df, on="UID", how="left")

# Fill missing predictions with MidtermClass = 1
merged["MidtermClass"] = merged["MidtermClass"].fillna(1).astype(int)

# --- save --------------------------------------------------
merged.to_csv(output_path, index=False)
print(f"✅ Predictions (with fallback=1) saved to: {output_path}")