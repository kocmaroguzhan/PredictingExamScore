import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import ast
import os
def load_embeddings(csv_path):
    df = pd.read_csv(csv_path)
    embedding_cols = [col for col in df.columns if col.endswith("_embedding")]
    validity_cols = [col.replace("_embedding", "_valid") for col in embedding_cols]
    for col in embedding_cols:
        df[col] = df[col].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
    return df, embedding_cols, validity_cols


def compute_similarity_scores(df, embedding_cols, target_uid):
    target_row = df[df["UID"] == target_uid].iloc[0]
    all_scores = []

    for idx, row in df.iterrows():
        if row["UID"] == target_uid:
            continue  # Skip comparing to self

        sim_scores = {
            "Target_UID": target_uid,
            "Compared_UID": row["UID"]
        }

        for emb_col in embedding_cols:
            valid_col = emb_col.replace("_embedding", "_valid")
            vec1 = target_row[emb_col]
            vec2 = row[emb_col]
            valid1 = target_row[valid_col]
            valid2 = row[valid_col]

            if valid1 == 1 and valid2 == 1 and np.linalg.norm(vec1) > 0 and np.linalg.norm(vec2) > 0:
                sim = cosine_similarity([vec1], [vec2])[0][0]
            else:
                sim = 0.0

            sim_scores[f"{emb_col}_similarity"] = sim

        all_scores.append(sim_scores)

    # Convert to DataFrame
    result_df = pd.DataFrame(all_scores)

    return result_df


def sort_similarity_independently(result_df, embedding_cols):
    """
    Sorts the result DataFrame independently for each embedding similarity column.

    Args:
        result_df (pd.DataFrame): DataFrame containing columns like 'Target_UID', 'Compared_UID', and embedding similarity scores.
        embedding_cols (list of str): List of embedding column names (without '_similarity' suffix).

    Returns:
        pd.DataFrame: Sorted DataFrame where each embedding's similarity is sorted independently.
    """
    sorted_dfs = []

    for emb_col in embedding_cols:
        sim_col = f"{emb_col}_similarity"
        
        # Sort by similarity for this embedding
        sorted_df = result_df.sort_values(by=sim_col, ascending=False).copy()
        
        # Mark which embedding this sorting is based on
        sorted_df["sorted_by"] = sim_col
        
        sorted_dfs.append(sorted_df)

    # Combine them vertically
    final_df = pd.concat(sorted_dfs, ignore_index=True)

    # Retain only relevant columns
    similarity_cols = [f"{col}_similarity" for col in embedding_cols]
    return final_df[["Target_UID", "Compared_UID", "sorted_by"] + similarity_cols]


def compute_weighted_class_score_for_top_k(sorted_df, labeled_df, embedding_cols, top_k=3):
    """
    For each embedding similarity column, selects top-k rows and computes similarity × MidtermClass score.

    Args:
        sorted_df (pd.DataFrame): Output of sort_similarity_independently().
        labeled_df (pd.DataFrame): Original DataFrame with UID and MidtermClass.
        embedding_cols (list of str): List of embedding names (without '_similarity').
        top_k (int): Number of top similar students to keep per embedding.

    Returns:
        pd.DataFrame: Filtered and scored DataFrame.
    """
    similarity_cols = [f"{col}_similarity" for col in embedding_cols]

    # Create a UID → MidtermClass map
    uid_to_class = labeled_df.set_index("UID")["MidtermClass"].to_dict()

    # Add MidtermClass info to each Compared_UID
    sorted_df = sorted_df.copy()
    sorted_df["Compared_UID_Class"] = sorted_df["Compared_UID"].map(uid_to_class)

    # Compute weighted class score per similarity column
    for sim_col in similarity_cols:
        weight_col = sim_col.replace("_similarity", "_weighted_class_score")
        sorted_df[weight_col] = sorted_df[sim_col] * sorted_df["Compared_UID_Class"]
    
    # Select top-k per Target_UID and embedding sort column
    top_k_df = (
        sorted_df
        .sort_values(
            by=["Target_UID", "sorted_by"] + similarity_cols,
            ascending=[True, True] + [False] * len(similarity_cols)
        )
        .groupby(["Target_UID", "sorted_by"])
        .head(top_k)
        .reset_index(drop=True)
    )
    
    return top_k_df

def predict_midtermclass_from_weighted_scores(top_k_df, embedding_cols):
    """
    Predicts MidtermClass for each student based on top-k similarity-weighted scores.

    Args:
        top_k_df (pd.DataFrame): Output from compute_weighted_class_score_for_top_k().
        embedding_cols (list of str): List of embedding names (without '_similarity').

    Returns:
        pd.DataFrame: DataFrame with columns Target_UID, embedding_name, predicted_midterm_class.
    """
    results = []

    for emb_col in embedding_cols:
        weighted_col = f"{emb_col}_weighted_class_score"

        # Filter rows that were sorted based on this embedding
        emb_df = top_k_df[top_k_df["sorted_by"] == f"{emb_col}_similarity"]

        # Get similarity column name
        similarity_col = f"{emb_col}_similarity"

        # Grouped sum of weighted class scores
        weighted_sum = emb_df.groupby("Target_UID")[weighted_col].sum()

        # Grouped sum of similarity weights
        similarity_sum = emb_df.groupby("Target_UID")[similarity_col].sum()
                
        # Avoid division by zero: if similarity_sum is 0, predicted class = 0
        predicted_values = np.where(
            similarity_sum != 0,
            weighted_sum / similarity_sum,
            1.0 ##classes are from 1 to 20 so minimum class is 1
        )

        # Construct result DataFrame
        predicted_scores = pd.DataFrame({
            "Target_UID": similarity_sum.index,
            "predicted_midterm_class": predicted_values
        })
        predicted_scores["embedding"] = emb_col
        # Append to final results list
        results.append(predicted_scores)

    # Combine results
    exam_predictions = pd.concat(results, ignore_index=True)

    # Optional: Round predicted class for cleaner output
    exam_predictions["predicted_midterm_class"] = exam_predictions["predicted_midterm_class"]

    return exam_predictions

def aggregate_predictions_by_mean(exam_predictions):
    """
    Aggregates per-embedding predicted_midterm_class values into a single
    prediction for each Target_UID by taking the mean across embeddings.

    Parameters
    ----------
    exam_predictions : pd.DataFrame
        Output of `predict_midtermclass_from_weighted_scores`, containing
        columns:
          • Target_UID
          • predicted_midterm_class  (per embedding)
          • embedding                (name of the embedding)

    Returns
    -------
    pd.DataFrame
        One row per Target_UID with columns:
          • Target_UID
          • mean_predicted_midterm_class   (float mean)
          • final_predicted_midterm_class  (rounded int)
          • num_embeddings_used            (count of embeddings that contributed)
    """
    # --- compute mean across embeddings ---
    grouped = (
        exam_predictions
        .groupby("Target_UID")["predicted_midterm_class"]
        .agg(mean_predicted_midterm_class="mean",
             num_embeddings_used="count")
        .reset_index()
    )

    # --- round to nearest integer for a discrete class prediction ---
    grouped["final_predicted_midterm_class"] = grouped["mean_predicted_midterm_class"].round(0).astype(int)

    return grouped



def save_exam_predictions_to_csv(grouped_df, output_path):
    """
    Saves the final predicted midterm class per student to a CSV file.

    Parameters
    ----------
    grouped_df : pd.DataFrame
        Output from `aggregate_predictions_by_mean`, containing:
          • Target_UID
          • final_predicted_midterm_class

    output_filename : str
        Name of the CSV file to save (default: "predicted_midterm_classes.csv")
    """
    # Prepare output DataFrame
    output_df = grouped_df.rename(columns={
        "Target_UID": "UID",
        "final_predicted_midterm_class": "MidtermClass"
    })[["UID", "MidtermClass"]]

    # Save to CSV
    output_df.to_csv(output_path, index=False)
    print(f"✅ All predictions saved to '{output_path}'")
    
parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
dataset_folder = os.path.join(parent_folder, "Dataset")
output_path = os.path.join(dataset_folder, "exam_prediction.csv")
labeled_csv=os.path.join(dataset_folder, "labeled_data.csv")
merged_csv= os.path.join(dataset_folder, "merged_embeddings.csv")
labeled_df, labeled_embedding_cols, labeled_validity_cols=load_embeddings(labeled_csv)
merged_df, merged_embedding_cols, merged_validity_cols=load_embeddings(merged_csv)
# Store individual prediction results
all_predictions = []

for target_uid in merged_df["UID"].unique():
    result_df = compute_similarity_scores(merged_df, merged_embedding_cols, target_uid)
    sorted_df = sort_similarity_independently(result_df, merged_embedding_cols)
    top_k_df = compute_weighted_class_score_for_top_k(sorted_df, labeled_df, merged_embedding_cols,top_k=5)
    exam_predictions = predict_midtermclass_from_weighted_scores(top_k_df, merged_embedding_cols)
    exam_prediction = aggregate_predictions_by_mean(exam_predictions)
    # Add Target UID for identification
    exam_prediction["Target_UID"] = target_uid
    all_predictions.append(exam_prediction)

# Combine all predictions into one DataFrame
final_all_predictions = pd.concat(all_predictions, ignore_index=True)
save_exam_predictions_to_csv(final_all_predictions, output_path)