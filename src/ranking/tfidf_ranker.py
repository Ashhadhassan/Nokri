"""
TF-IDF Baseline Matcher and Ranker.
Implements BaseJobRanker contract using Scikit-Learn TfidfVectorizer and Cosine Similarity.
Provides deterministic, fast baseline matching and Top-K retrieval across the 50,000 jobs corpus.
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.ranking.base import BaseJobRanker
from src.schema import JOB_ID_COL, JOB_TITLE_COL, JOB_DESC_COL, JOB_SKILLS_COL, JOB_COMPANY_COL, JOB_LOCATION_COL

MODEL_ARTIFACT_PATH = "models/tfidf_ranker.pkl"

class TFIDFJobRanker(BaseJobRanker):
    def __init__(
        self,
        max_features: int = 25000,
        ngram_range: tuple = (1, 2),
        stop_words: str = "english"
    ):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words=stop_words,
            sublinear_tf=True
        )
        self.job_matrix = None
        self.jobs_metadata: List[Dict[str, Any]] = []
        self.is_fitted = False

    def _prepare_job_text(self, row: dict) -> str:
        title = row.get("title", "")
        role = row.get("role", "")
        desc = row.get("description", "")
        skills = " ".join(row.get("skills", []))
        return f"{title} {role} {skills} {desc}"

    def fit(self, jobs_df: Any) -> None:
        """Fits TF-IDF vectorizer on 50,000 jobs corpus and indexes matrix."""
        if isinstance(jobs_df, str):
            jobs_df = pd.read_parquet(jobs_df)

        print(f"Fitting TF-IDF on {len(jobs_df)} jobs corpus...")
        corpus = [self._prepare_job_text(row) for row in jobs_df.to_dict(orient="records")]
        
        self.job_matrix = self.vectorizer.fit_transform(corpus)
        self.jobs_metadata = jobs_df[[
            "job_id", "title", "role", "company", "location", "skills"
        ]].to_dict(orient="records")
        self.is_fitted = True
        print(f"TF-IDF fitted. Vocabulary size: {len(self.vectorizer.vocabulary_)}, Matrix shape: {self.job_matrix.shape}")

    def score_pair(self, resume_text: str, job_description: str) -> float:
        """Calculates 0-100 cosine match score between a resume and single job."""
        if not self.is_fitted:
            raise RuntimeError("TFIDFJobRanker is not fitted yet.")

        res_vec = self.vectorizer.transform([resume_text])
        job_vec = self.vectorizer.transform([job_description])
        sim = cosine_similarity(res_vec, job_vec)[0][0]
        # Scale to 0-100 range
        score = float(np.clip(sim * 100.0, 0.0, 100.0))
        return round(score, 2)

    def rank_jobs(
        self,
        resume_text: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Retrieves and ranks the top_k best matching jobs for the given resume."""
        if not self.is_fitted or self.job_matrix is None:
            raise RuntimeError("TFIDFJobRanker is not fitted yet.")

        res_vec = self.vectorizer.transform([resume_text])
        # Vectorized cosine similarity: dot product of normalized vectors
        scores = (self.job_matrix * res_vec.T).toarray().flatten()

        # Fast argpartition for top_k indices
        if top_k >= len(scores):
            top_indices = np.argsort(-scores)
        else:
            partitioned = np.argpartition(-scores, top_k)[:top_k]
            top_indices = partitioned[np.argsort(-scores[partitioned])]

        results = []
        for rank_idx, idx in enumerate(top_indices[:top_k], start=1):
            meta = self.jobs_metadata[idx]
            score_val = round(float(np.clip(scores[idx] * 100.0, 0.0, 100.0)), 2)
            results.append({
                "rank": rank_idx,
                "job_id": meta["job_id"],
                "title": meta["title"],
                "company": meta["company"],
                "location": meta["location"],
                "score": score_val,
                "skills": meta.get("skills", [])
            })

        return results

    def save(self, filepath: str = MODEL_ARTIFACT_PATH) -> None:
        """Serializes fitted ranker to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        print(f"TF-IDF model artifact saved to {filepath}")

    @classmethod
    def load(cls, filepath: str = MODEL_ARTIFACT_PATH) -> "TFIDFJobRanker":
        """Loads fitted ranker from disk."""
        with open(filepath, "rb") as f:
            model = pickle.load(f)
        return model
