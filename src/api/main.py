"""
FastAPI REST API Service for Resume-to-Job Matcher.
Exposes typed endpoints for parsing, matching, ranking, and user feedback.
Supports multi-model selection: TF-IDF, FAISS Dense Retriever, Two-Stage LightGBM Ranker.
"""

import os
import sys
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.schema import MatchResult, RankedJobItem, ParsedResume
from src.api.models import (
    HealthResponse, ParseRequest, MatchRequest, RankRequest, RankResponse,
    FeedbackRequest, FeedbackResponse
)
from src.ranking.tfidf_ranker import TFIDFJobRanker, MODEL_ARTIFACT_PATH
from src.ranking.faiss_retriever import FAISSDenseRetriever
from src.ranking.lgbm_ranker import TwoStageLGBMRanker
from src.parser.resume_parser import parse_resume_text, parse_resume_file
from src.nlp.skill_extractor import extract_skills_from_text
from src.matching.explainer import explain_match

# Model registry
rankers: Dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading ranking models into memory...")
    # 1. Load TF-IDF
    tfidf = TFIDFJobRanker.load()
    rankers["tfidf"] = tfidf
    print(f"TF-IDF loaded ({len(tfidf.jobs_metadata)} jobs).")

    # 2. Load FAISS Dense Retriever
    faiss_retriever = FAISSDenseRetriever()
    rankers["faiss"] = faiss_retriever
    print(f"FAISS index loaded ({faiss_retriever.index.ntotal} vectors).")

    # 3. Load Two-Stage LightGBM Ranker
    lgbm_ranker = TwoStageLGBMRanker(retriever=faiss_retriever, tfidf=tfidf)
    rankers["lgbm"] = lgbm_ranker
    print("LightGBM Re-Ranker loaded.")

    yield
    print("Shutting down API service.")

app = FastAPI(
    title="Resume-to-Job Matcher API",
    description="Production-grade Semantic Matcher, Job Retrieval, and Skill Gap Analysis Service",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_engine(model_name: Optional[str]):
    key = (model_name or "lgbm").lower()
    if key in rankers:
        return rankers[key]
    return rankers.get("lgbm") or rankers.get("tfidf")

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    loaded = len(rankers) > 0
    total = len(rankers["tfidf"].jobs_metadata) if "tfidf" in rankers else 0
    return HealthResponse(
        status="healthy",
        model_loaded=loaded,
        total_jobs_indexed=total
    )

@app.post("/parse", response_model=ParsedResume, tags=["Parsing"])
async def parse_resume(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None)
):
    """Parses an uploaded file (PDF/DOCX) or raw pasted text."""
    if file is not None:
        contents = await file.read()
        if len(contents) > 2 * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="File size exceeds 2 MB limit."
            )
        temp_dir = "data/temp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, file.filename)
        with open(temp_path, "wb") as f:
            f.write(contents)
        try:
            parsed = parse_resume_file(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        return parsed

    if text:
        return parse_resume_text(text)

    raise HTTPException(status_code=400, detail="Provide either a file or text payload.")

@app.post("/match", response_model=MatchResult, tags=["Matching"])
def match_pair(payload: MatchRequest):
    """Calculates 0-100 match score and skill gap between a resume and single job."""
    engine = get_engine(payload.model)
    if engine is None:
        raise HTTPException(status_code=503, detail="Ranking model not ready.")

    score = engine.score_pair(payload.resume_text, payload.job_description)
    resume_skills = extract_skills_from_text(payload.resume_text)
    job_skills = extract_skills_from_text(payload.job_description)

    return explain_match(
        score=score,
        resume_skills=resume_skills,
        job_skills=job_skills,
        job_title=payload.job_title
    )

@app.post("/rank", response_model=RankResponse, tags=["Ranking"])
def rank_top_jobs(payload: RankRequest):
    """Retrieves top-K jobs from the 50,000 indexed corpus for a given resume."""
    engine = get_engine(payload.model)
    if engine is None:
        raise HTTPException(status_code=503, detail="Ranking model not ready.")

    resume_skills = extract_skills_from_text(payload.resume_text)
    ranked_raw = engine.rank_jobs(payload.resume_text, top_k=payload.top_k)

    ranked_items = []
    for r in ranked_raw:
        job_skills = r.get("skills", [])
        explanation_obj = explain_match(
            score=r["score"],
            resume_skills=resume_skills,
            job_skills=job_skills,
            job_title=r["title"],
            company=r["company"]
        )
        ranked_items.append(RankedJobItem(
            rank=r["rank"],
            job_id=r["job_id"],
            title=r["title"],
            company=r["company"],
            location=r["location"],
            score=r["score"],
            matched_skills=explanation_obj.matched_skills,
            missing_skills=explanation_obj.missing_skills,
            explanation=explanation_obj.explanation
        ))

    total = len(rankers["tfidf"].jobs_metadata) if "tfidf" in rankers else 50000
    return RankResponse(
        total_indexed=total,
        returned_count=len(ranked_items),
        jobs=ranked_items
    )

@app.post("/feedback", response_model=FeedbackResponse, tags=["Feedback"])
def record_feedback(payload: FeedbackRequest):
    """Logs thumbs up/down feedback for model tracking."""
    log_dir = "data/logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "feedback.jsonl")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(payload.model_dump_json() + "\n")
    return FeedbackResponse(status="recorded")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000, reload=True)
