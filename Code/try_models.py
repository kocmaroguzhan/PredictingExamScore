import pandas as pd
import glob
import ast
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import sys

try:
    print('Script started', flush=True)
    # 1. Find all oof_preds_lab*.csv and oof_preds_quiz*.csv files
    lab_files = sorted(glob.glob('Dataset/oof_preds_lab*.csv'))
    quiz_files = sorted(glob.glob('Dataset/oof_preds_quiz*.csv'))
    oof_files = lab_files + quiz_files

    print(f"Found {len(lab_files)} lab files: {[os.path.basename(f) for f in lab_files]}", flush=True)
    print(f"Found {len(quiz_files)} quiz files: {[os.path.basename(f) for f in quiz_files]}", flush=True)

    # 2. Load and merge features
    features = []
    for file in oof_files:
        df = pd.read_csv(file)
        # Parse the vector string into 8 float columns
        vec_col = df.columns[-1]
        def parse_vec(x):
            try:
                return list(map(float, ast.literal_eval(x)))
            except Exception:
                return [float('nan')]*8
        vecs = df[vec_col].apply(parse_vec)
        base = os.path.basename(file).replace('.csv','')
        vec_df = pd.DataFrame(vecs.tolist(), columns=pd.Index([f"{base}_f{i}" for i in range(1, 9)]))
        out = pd.concat([df[['UID']], vec_df], axis=1)
        out = out.rename(columns={'UID': 'student_id'})
        features.append(out)

    # Merge all features on student_id
    from functools import reduce
    X_all = reduce(lambda left, right: pd.merge(left, right, on='student_id'), features)

    print(f"Total features per student: {X_all.shape[1] - 1}", flush=True)  # -1 for student_id

    # 3. Load target
    y = pd.read_csv('Dataset/processed_train_student.csv')[['UID', 'MidtermClass']]
    y = y.rename(columns={'UID': 'student_id', 'MidtermClass': 'midterm_score'})

    # 4. Merge features and target
    data = pd.merge(X_all, y, on='student_id')
    X = data.drop(['student_id', 'midterm_score'], axis=1)
    y = data['midterm_score']

    print(f"Training data shape: {X.shape}", flush=True)
    print(f"Number of students: {len(data)}", flush=True)

    # 5. Try different models
    models = {
        'RandomForest': RandomForestRegressor(random_state=42),
        'GBM': GradientBoostingRegressor(random_state=42),
        'Ridge': Ridge()
    }

    print('About to evaluate models', flush=True)
    print("\nModel Performance (RMSE and R2):", flush=True)
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
        rmse = (-scores.mean())**0.5
        r2_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        r2 = r2_scores.mean()
        print(f"{name}: RMSE={rmse:.4f}, R2={r2:.4f}", flush=True)
except Exception as e:
    print(f"Exception occurred: {e}", flush=True) 