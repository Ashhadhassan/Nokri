"""
Trains the LightGBM Re-Ranker on Supervised Pairs (Good Fit, Potential Fit, No Fit).
Extracts 5 multi-signal ranking features and trains a regression/ranking model.
Saves model to models/lgbm_ranker.txt.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.ranking.features import extract_pairwise_features
from src.ranking.tfidf_ranker import TFIDFJobRanker
from src.nlp.skill_extractor import extract_skills_from_text
from sentence_transformers import SentenceTransformer

LABEL_MAP = {
    "No Fit": 0,
    "Potential Fit": 1,
    "Good Fit": 2
}

def train_lgbm(
    train_csv: str = "data/raw/hf_fit_train.csv",
    output_model: str = "models/lgbm_ranker.txt",
    sample_size: int = 4000
):
    print("=" * 60)
    print("TRAINING LIGHTGBM RE-RANKER")
    print("=" * 60)
    t0 = time.time()

    print(f"Loading {sample_size} training pairs from {train_csv}...")
    df = pd.read_csv(train_csv, nrows=sample_size)
    df["target"] = df["label"].map(LABEL_MAP).fillna(0).astype(int)

    print("Loading feature extraction models (TF-IDF & MiniLM)...")
    tfidf = TFIDFJobRanker.load()
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"Extracting 5-dimensional feature vectors for {len(df)} pairs...")
    resume_texts = df["resume_text"].tolist()
    job_texts = df["job_description_text"].tolist()

    # Batch encode embeddings
    res_vecs = embedder.encode(resume_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    job_vecs = embedder.encode(job_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    sem_sims = np.sum(res_vecs * job_vecs, axis=1)

    features = []
    for i in range(len(df)):
        r_text = resume_texts[i]
        j_text = job_texts[i]
        s_sim = float(sem_sims[i])
        t_sim = float(tfidf.score_pair(r_text, j_text) / 100.0)
        r_skills = extract_skills_from_text(r_text)
        j_skills = extract_skills_from_text(j_text)
        
        feat = extract_pairwise_features(
            semantic_sim=s_sim,
            tfidf_sim=t_sim,
            resume_skills=r_skills,
            job_skills=j_skills,
            resume_text=r_text,
            job_title="Target Role"
        )
        features.append(feat)

    X = np.array(features, dtype=np.float32)
    y = df["target"].values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.9,
        "verbosity": -1,
        "random_state": 42
    }

    print("Training LightGBM booster...")
    booster = lgb.train(
        params,
        train_data,
        num_boost_round=150,
        valid_sets=[val_data]
    )

    os.makedirs(os.path.dirname(output_model), exist_ok=True)
    booster.save_model(output_model)
    elapsed = time.time() - t0

    print(f"\n[SUCCESS] LightGBM Re-Ranker trained in {elapsed:.2f}s and saved to {output_model}!")
    print("Feature Importances:")
    feat_names = ["semantic_sim", "tfidf_sim", "skill_overlap_ratio", "missing_skills_count", "title_match"]
    for name, imp in zip(feat_names, booster.feature_importance()):
        print(f"  - {name}: {imp}")

if __name__ == "__main__":
    train_lgbm()
