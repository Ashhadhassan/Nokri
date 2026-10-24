"""
Feature Engineering for Two-Stage Re-Ranking.
Extracts ranking signals between resume and job candidates:
1. Semantic Similarity (Sentence-Transformers MiniLM)
2. Lexical Similarity (TF-IDF Cosine)
3. Skill Overlap Ratio (Jaccard Index)
4. Missing Skills Count
5. Title / Role Match Score
6. Experience Match Score
"""

import re
from typing import List, Set, Dict, Any
import numpy as np

def compute_jaccard_overlap(resume_skills: List[str], job_skills: List[str]) -> tuple:
    """Returns (overlap_ratio, missing_count)."""
    res_set = set(s.lower() for s in resume_skills)
    job_set = set(s.lower() for s in job_skills)
    
    if not job_set:
        return 0.5, 0
    
    intersection = res_set.intersection(job_set)
    overlap_ratio = len(intersection) / len(job_set)
    missing_count = len(job_set) - len(intersection)
    return float(overlap_ratio), int(missing_count)

def compute_title_overlap(resume_text: str, job_title: str) -> float:
    """Computes token overlap between resume text and job title."""
    if not job_title or not resume_text:
        return 0.0
    title_tokens = set(re.findall(r"\b[a-zA-Z]{3,}\b", job_title.lower()))
    if not title_tokens:
        return 0.0
    resume_tokens = set(re.findall(r"\b[a-zA-Z]{3,}\b", resume_text[:1000].lower()))
    common = title_tokens.intersection(resume_tokens)
    return float(len(common) / len(title_tokens))

def extract_pairwise_features(
    semantic_sim: float,
    tfidf_sim: float,
    resume_skills: List[str],
    job_skills: List[str],
    resume_text: str,
    job_title: str
) -> np.ndarray:
    """
    Constructs a 5-dimensional feature vector for the LightGBM re-ranker.
    Feature Order:
    0: semantic_similarity (0-1)
    1: tfidf_similarity (0-1)
    2: skill_overlap_ratio (0-1)
    3: missing_skills_count (int >= 0)
    4: title_match_score (0-1)
    """
    overlap_ratio, missing_count = compute_jaccard_overlap(resume_skills, job_skills)
    title_match = compute_title_overlap(resume_text, job_title)
    
    return np.array([
        float(semantic_sim),
        float(tfidf_sim),
        float(overlap_ratio),
        float(missing_count),
        float(title_match)
    ], dtype=np.float32)
