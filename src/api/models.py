"""
FastAPI Request and Response Models using Pydantic v2.
"""

from typing import List, Optional
from pydantic import BaseModel, Field
from src.schema import MatchResult, RankedJobItem, ParsedResume

class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool
    total_jobs_indexed: int
    version: str = "1.0.0"

class ParseRequest(BaseModel):
    text: str = Field(..., description="Raw text of the resume")

class MatchRequest(BaseModel):
    resume_text: str = Field(..., description="Resume text")
    job_description: str = Field(..., description="Target job description")
    job_title: Optional[str] = "Target Job"
    model: Optional[str] = Field(default="lgbm", description="Model engine: 'tfidf', 'faiss', or 'lgbm'")

class RankRequest(BaseModel):
    resume_text: str = Field(..., description="Resume text to find top matching jobs for")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of ranked jobs to return")
    model: Optional[str] = Field(default="lgbm", description="Model engine: 'tfidf', 'faiss', or 'lgbm'")

class RankResponse(BaseModel):
    total_indexed: int
    returned_count: int
    jobs: List[RankedJobItem]

class FeedbackRequest(BaseModel):
    job_id: str
    resume_id: Optional[str] = "anonymous"
    thumbs_up: bool
    notes: Optional[str] = None

class FeedbackResponse(BaseModel):
    status: str = "recorded"
