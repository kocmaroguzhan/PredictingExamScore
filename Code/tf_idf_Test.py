import os
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from tqdm import tqdm

# Setup paths
parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
feature_folder = os.path.join(parent_folder, "Quizzes")  # or "Labs"
dataset_folder = os.path.join(parent_folder, "Dataset")
os.makedirs(dataset_folder, exist_ok=True)

student_features = {}

# Process each quiz/lab JSON file
for file_name in os.listdir(feature_folder):
    if not file_name.endswith(".json"):
        continue

    feature_path = os.path.join(feature_folder, file_name)
    feature_id = os.path.splitext(file_name)[0].split(".")[0]
    print(f"📄 Processing {file_name}...")

    with open(feature_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract valid student code
    raw_student_codes = {
        s['id']: "\n".join([v for k, v in s.items() if k.endswith(".java")]).strip()
        for s in data.get("answers", [])
    }

    # Filter out empty code submissions
    student_codes = {sid: code for sid, code in raw_student_codes.items() if code}
    if len(student_codes) <= 1:
        print(f"⚠️ Skipping {feature_id}: Not enough valid submissions")
        continue

    student_ids = list(student_codes.keys())
    code_texts = list(student_codes.values())

    # TF-IDF encoding
    vectorizer = TfidfVectorizer(token_pattern=r'\b\w+\b', norm='l2')
    tfidf_matrix = vectorizer.fit_transform(code_texts)

    # Cosine similarity between submissions
    cosine_sim = cosine_similarity(tfidf_matrix)
    np.fill_diagonal(cosine_sim, 0)
    avg_sim = cosine_sim.sum(axis=1) / (cosine_sim.shape[1] - 1)
    avg_sim_dict = dict(zip(student_ids, avg_sim))

    # DBSCAN clustering
    dbscan = DBSCAN(eps=0.01, min_samples=2, metric="cosine")
    cluster_labels = dbscan.fit_predict(tfidf_matrix.toarray())
    cluster_dict = dict(zip(student_ids, cluster_labels))

    # Store features
    for student_id in raw_student_codes:
        if student_id not in student_features:
            student_features[student_id] = {}

        # is_attended
        student_features[student_id][f"{feature_id}_is_attended"] = int(student_id in student_codes)

        # avg similarity if exists
        if student_id in avg_sim_dict:
            student_features[student_id][f"{feature_id}_avg_similarity"] = avg_sim_dict[student_id]

        # cluster label (or -1 for outlier)
        if student_id in cluster_dict:
            student_features[student_id][f"{feature_id}_cluster"] = cluster_dict[student_id]

        # outlier flag
        if student_id in cluster_dict:
            student_features[student_id][f"{feature_id}_is_outlier"] = int(cluster_dict[student_id] == -1)

# Save to CSV
df = pd.DataFrame.from_dict(student_features, orient="index").reset_index()
df.rename(columns={"index": "UID"}, inplace=True)
output_path = os.path.join(dataset_folder, "student_features_with_clusters.csv")
df.to_csv(output_path, index=False)
print(f"\n✅ Features with clustering saved to: {output_path}")
