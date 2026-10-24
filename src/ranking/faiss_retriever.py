"""
Dense Semantic Retriever using Sentence-Transformers (all-MiniLM-L6-v2) and FAISS.
Implements BaseJobRanker contract for sub-second semantic retrieval across 50,000 jobs.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
import faiss
from sentence_transformers import SentenceTransformer

from src.ranking.base import BaseJobRanker

MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_PATH = "models/faiss_index.bin"
META_PATH = "models/faiss_metadata.parquet"

class FAISSDenseRetriever(BaseJobRanker):
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        index_path: str = INDEX_PATH,
        meta_path: str = META_PATH
    ):
        self.model_name = model_name
        self.index_path = index_path
        self.meta_path = meta_path
        self._model = None
        self.index = None
        self.metadata: List[Dict[str, Any]] = []
        self.dimension = 384  # MiniLM dimension
        self.is_fitted = False

        if os.path.exists(index_path) and os.path.exists(meta_path):
            self.load()

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def fit(self, jobs_df: Any) -> None:
        """Encodes all jobs and constructs FAISS IndexFlatIP (cosine similarity)."""
        if isinstance(jobs_df, str):
            jobs_df = pd.read_parquet(jobs_df)

        print(f"Building FAISS Index for {len(jobs_df)} jobs using {self.model_name}...")
        texts = []
        meta_records = []
        for row in jobs_df.to_dict(orient="records"):
            t = f"{row.get('title', '')} - {row.get('role', '')}. " + " ".join(row.get('skills', [])) + " " + row.get('description', '')[:300]
            texts.append(t)
            meta_records.append({
                "job_id": row["job_id"],
                "title": row["title"],
                "role": row.get("role", ""),
                "company": row["company"],
                "location": row["location"],
                "skills": row.get("skills", []),
                "description": row.get("description", "")
            })

        # Batch encode with normalization for cosine similarity via inner product
        embeddings = self.model.encode(
            texts,
            batch_size=128,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        self.metadata = meta_records
        self.is_fitted = True

        self.save()
        print(f"FAISS index built with {self.index.ntotal} vectors.")

    def score_pair(self, resume_text: str, job_description: str) -> float:
        """Computes semantic cosine similarity (0-100) between resume and job."""
        vecs = self.model.encode([resume_text, job_description], normalize_embeddings=True)
        sim = float(np.dot(vecs[0], vecs[1]))
        score = float(np.clip(sim * 100.0, 0.0, 100.0))
        return round(score, 2)

    def rank_jobs(
        self,
        resume_text: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Dense vector search returning top_k jobs."""
        if self.index is None or not self.metadata:
            raise RuntimeError("FAISS Index is not loaded or fitted.")

        query_vec = self.model.encode([resume_text], normalize_embeddings=True)
        query_vec = np.ascontiguousarray(query_vec, dtype=np.float32)

        # Search top_k
        k = min(top_k * 2 if filters else top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vec, k)

        scores = scores[0]
        indices = indices[0]

        results = []
        for rank_idx, (idx, score_val) in enumerate(zip(indices, scores), start=1):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx]
            
            # Optional location filter
            if filters and "location" in filters and filters["location"]:
                loc_filter = filters["location"].lower()
                if loc_filter not in meta["location"].lower():
                    continue

            calibrated_score = round(float(np.clip(score_val * 100.0, 0.0, 100.0)), 2)
            results.append({
                "rank": len(results) + 1,
                "job_id": meta["job_id"],
                "title": meta["title"],
                "role": meta.get("role", ""),
                "company": meta["company"],
                "location": meta["location"],
                "score": calibrated_score,
                "skills": meta.get("skills", []),
                "description": meta.get("description", "")
            })
            if len(results) >= top_k:
                break

        return results

    def save(self) -> None:
        """Saves FAISS index and metadata to disk."""
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        pd.DataFrame(self.metadata).to_parquet(self.meta_path, index=False)

    def load(self) -> None:
        """Loads FAISS index and metadata from disk."""
        self.index = faiss.read_index(self.index_path)
        df = pd.read_parquet(self.meta_path)
        self.metadata = df.to_dict(orient="records")
        self.is_fitted = True
