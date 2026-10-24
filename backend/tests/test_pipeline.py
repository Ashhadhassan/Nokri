"""
Comprehensive Unit and Integration Test Suite.
Tests schema validation, skill extractor, TF-IDF baseline, resume parser, explainer,
FAISS Dense Retriever, and Two-Stage LightGBM Ranker.
"""

import os
import pytest
import numpy as np

import sys

backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
workspace_dir = os.path.abspath(os.path.join(backend_dir, ".."))
for p in [workspace_dir, backend_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from src.schema import JobPosting, ParsedResume, MatchResult
    from src.nlp.skill_extractor import extract_skills_from_text
    from src.matching.explainer import explain_match
    from src.ranking.tfidf_ranker import TFIDFJobRanker
    from src.ranking.faiss_retriever import FAISSDenseRetriever
    from src.ranking.lgbm_ranker import TwoStageLGBMRanker
    from src.ranking.features import extract_pairwise_features
    from src.parser.resume_parser import parse_resume_text, parse_resume_file
except ImportError:
    from backend.src.schema import JobPosting, ParsedResume, MatchResult
    from backend.src.nlp.skill_extractor import extract_skills_from_text
    from backend.src.matching.explainer import explain_match
    from backend.src.ranking.tfidf_ranker import TFIDFJobRanker
    from backend.src.ranking.faiss_retriever import FAISSDenseRetriever
    from backend.src.ranking.lgbm_ranker import TwoStageLGBMRanker
    from backend.src.ranking.features import extract_pairwise_features
    from backend.src.parser.resume_parser import parse_resume_text, parse_resume_file

def test_skill_extractor():
    text = "Full Stack Engineer with expertise in Python, JavaScript, Docker, and AWS."
    skills = extract_skills_from_text(text)
    assert "Python" in skills
    assert "JavaScript" in skills
    assert "Docker" in skills
    assert "Amazon Web Services" in skills

def test_explainer():
    res_skills = ["Python", "Docker"]
    job_skills = ["Python", "Docker", "Kubernetes", "AWS"]
    result = explain_match(85.0, res_skills, job_skills, "DevOps Engineer")
    
    assert result.score == 85.0
    assert "Python" in result.matched_skills
    assert "Kubernetes" in result.missing_skills
    assert len(result.matched_skills) == 2
    assert len(result.missing_skills) == 2

def test_parser_text():
    text = "Data Scientist with experience in Machine Learning, Pandas, and SQL."
    parsed = parse_resume_text(text)
    assert len(parsed.raw_text) > 0
    assert "SQL" in parsed.skills

def test_parser_sample_pdf():
    sample_dir = "data/sample_resumes"
    if os.path.exists(sample_dir):
        pdfs = [os.path.join(sample_dir, f) for f in os.listdir(sample_dir) if f.endswith(".pdf")]
        if pdfs:
            parsed = parse_resume_file(pdfs[0])
            assert len(parsed.raw_text) > 50
            assert len(parsed.skills) > 0

def test_tfidf_ranker():
    model_path = "models/tfidf_ranker.pkl"
    if os.path.exists(model_path):
        ranker = TFIDFJobRanker.load(model_path)
        assert ranker.is_fitted
        score = ranker.score_pair("Python developer", "Looking for a Python software engineer")
        assert 0.0 <= score <= 100.0
        
        ranked = ranker.rank_jobs("Python developer", top_k=5)
        assert len(ranked) == 5
        assert ranked[0]["score"] >= ranked[-1]["score"]

def test_features_extraction():
    feat = extract_pairwise_features(
        semantic_sim=0.85,
        tfidf_sim=0.70,
        resume_skills=["Python", "Docker"],
        job_skills=["Python", "Docker", "Kubernetes"],
        resume_text="Senior Python developer with Docker expertise",
        job_title="Senior Python Engineer"
    )
    assert feat.shape == (5,)
    assert 0.0 <= feat[0] <= 1.0  # sem_sim
    assert feat[2] > 0.0         # overlap
    assert feat[3] == 1.0        # missing count (Kubernetes)

def test_faiss_retriever():
    if os.path.exists("models/faiss_index.bin"):
        retriever = FAISSDenseRetriever()
        assert retriever.is_fitted
        score = retriever.score_pair("Python developer", "Backend software engineer writing Python microservices")
        assert 0.0 <= score <= 100.0
        
        ranked = retriever.rank_jobs("Senior Software Engineer", top_k=3)
        assert len(ranked) == 3

def test_lgbm_ranker():
    if os.path.exists("models/lgbm_ranker.txt"):
        ranker = TwoStageLGBMRanker()
        assert ranker.is_fitted
        score = ranker.score_pair("Python engineer", "Looking for a Python software engineer")
        assert 0.0 <= score <= 100.0

        ranked = ranker.rank_jobs("Python software engineer", top_k=3)
        assert len(ranked) == 3
