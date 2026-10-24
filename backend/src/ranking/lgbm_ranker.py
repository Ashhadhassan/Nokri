"""
Two-Stage Learned Re-Ranker using LightGBM (LambdaRank).
Implements BaseJobRanker contract:
- Stage 1: Candidate Generation via FAISSDenseRetriever (Top 50)
- Stage 2: Re-Ranking via LightGBM using multi-signal engineered features
"""

import os
import lightgbm as lgb
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

try:
    from src.ranking.base import BaseJobRanker
    from src.ranking.faiss_retriever import FAISSDenseRetriever
    from src.ranking.tfidf_ranker import TFIDFJobRanker
    from src.ranking.features import extract_pairwise_features
    from src.nlp.skill_extractor import extract_skills_from_text
except ImportError:
    from backend.src.ranking.base import BaseJobRanker
    from backend.src.ranking.faiss_retriever import FAISSDenseRetriever
    from backend.src.ranking.tfidf_ranker import TFIDFJobRanker
    from backend.src.ranking.features import extract_pairwise_features
    from backend.src.nlp.skill_extractor import extract_skills_from_text

def get_default_lgbm_path():
    candidates = [
        os.path.join("models", "lgbm_ranker.txt"),
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "models", "lgbm_ranker.txt"),
        os.path.join(os.path.dirname(__file__), "..", "..", "models", "lgbm_ranker.txt"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return os.path.abspath(candidates[0])

LGBM_MODEL_PATH = get_default_lgbm_path()

class TwoStageLGBMRanker(BaseJobRanker):
    def __init__(
        self,
        retriever: Optional[FAISSDenseRetriever] = None,
        tfidf: Optional[TFIDFJobRanker] = None,
        model_path: str = LGBM_MODEL_PATH
    ):
        self.retriever = retriever or FAISSDenseRetriever()
        self.tfidf = tfidf or TFIDFJobRanker.load()
        self.model_path = model_path
        self.booster: Optional[lgb.Booster] = None
        self.is_fitted = False

        if os.path.exists(model_path):
            self.load()

    def fit(self, training_data: Any) -> None:
        """Fits or updates LightGBM model on pairwise feature dataset."""
        if isinstance(training_data, str) and os.path.exists(training_data):
            self.booster = lgb.Booster(model_file=training_data)
            self.is_fitted = True
            return

    def score_pair(self, resume_text: str, job_description: str) -> float:
        """Scores a single pair using the two-stage model."""
        sem_sim = self.retriever.score_pair(resume_text, job_description) / 100.0
        tfidf_sim = self.tfidf.score_pair(resume_text, job_description) / 100.0
        
        res_skills = extract_skills_from_text(resume_text)
        job_skills = extract_skills_from_text(job_description)
        
        feat = extract_pairwise_features(
            semantic_sim=sem_sim,
            tfidf_sim=tfidf_sim,
            resume_skills=res_skills,
            job_skills=job_skills,
            resume_text=resume_text,
            job_title="Target Job"
        ).reshape(1, -1)

        if self.booster is not None:
            raw_score = float(self.booster.predict(feat)[0])
            # Sigmoid normalization to 0-100 range
            score = 100.0 / (1.0 + np.exp(-raw_score * 2.0))
        else:
            # Fallback ensemble
            score = (sem_sim * 0.6 + tfidf_sim * 0.2 + (len(set(res_skills).intersection(set(job_skills))) / max(1, len(job_skills))) * 0.2) * 100.0

        return round(float(np.clip(score, 0.0, 100.0)), 2)

    def rank_jobs(
        self,
        resume_text: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Stage 1: Retrieve top 50 candidates via FAISS
        Stage 2: Re-rank using LightGBM model
        """
        # Candidate generation (retrieve top 50 candidates)
        candidates = self.retriever.rank_jobs(resume_text, top_k=50, filters=filters)
        if not candidates:
            return []

        res_skills = extract_skills_from_text(resume_text)
        
        # Build feature matrix for candidate jobs
        features_list = []
        for cand in candidates:
            sem_sim = cand["score"] / 100.0
            tfidf_sim = self.tfidf.score_pair(resume_text, cand.get("description", cand["title"])) / 100.0
            job_skills = cand.get("skills", [])
            
            feat = extract_pairwise_features(
                semantic_sim=sem_sim,
                tfidf_sim=tfidf_sim,
                resume_skills=res_skills,
                job_skills=job_skills,
                resume_text=resume_text,
                job_title=cand["title"]
            )
            features_list.append(feat)

        X = np.array(features_list)
        
        if self.booster is not None:
            pred_scores = self.booster.predict(X)
        else:
            # Weighted ensemble if booster not yet trained
            pred_scores = X[:, 0] * 0.55 + X[:, 1] * 0.20 + X[:, 2] * 0.25

        # Sort candidates by re-ranker score
        sorted_indices = np.argsort(-pred_scores)
        
        final_ranked = []
        for rank_idx, idx in enumerate(sorted_indices[:top_k], start=1):
            cand = candidates[idx]
            calibrated_score = round(float(np.clip((pred_scores[idx] + 1.0) * 50.0 if self.booster else pred_scores[idx] * 100.0, 0.0, 100.0)), 2)
            final_ranked.append({
                "rank": rank_idx,
                "job_id": cand["job_id"],
                "title": cand["title"],
                "company": cand["company"],
                "location": cand["location"],
                "score": calibrated_score,
                "skills": cand.get("skills", []),
                "description": cand.get("description", "")
            })

        return final_ranked

    def save(self) -> None:
        if self.booster:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            self.booster.save_model(self.model_path)

    def load(self) -> None:
        self.booster = lgb.Booster(model_file=self.model_path)
        self.is_fitted = True
