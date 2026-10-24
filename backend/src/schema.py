"""
Shared Data Schemas and Type Definitions for the Resume-to-Job Matcher system.
Ensures consistency across data pipelines, models, and API endpoints.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

# ---------------------------------------------------------
# Parquet Column Names Constants
# ---------------------------------------------------------
JOB_ID_COL = "job_id"
JOB_TITLE_COL = "title"
JOB_ROLE_COL = "role"
JOB_DESC_COL = "description"
JOB_SKILLS_COL = "skills"
JOB_EXP_COL = "experience"
JOB_LOCATION_COL = "location"
JOB_COMPANY_COL = "company"

RESUME_ID_COL = "resume_id"
RESUME_CAT_COL = "category"
RESUME_TEXT_COL = "raw_text"
RESUME_SKILLS_COL = "skills"
RESUME_SPLIT_COL = "split"


class JobPosting(BaseModel):
    """Schema for a cleaned job posting."""
    job_id: str
    title: str
    role: Optional[str] = ""
    description: str
    skills: List[str] = Field(default_factory=list)
    experience: Optional[str] = ""
    location: Optional[str] = "Remote"
    company: Optional[str] = "Unknown"


class ParsedResume(BaseModel):
    """Schema for a parsed and sanitized resume."""
    resume_id: str
    category: Optional[str] = "General"
    raw_text: str
    skills: List[str] = Field(default_factory=list)
    experience_years: Optional[float] = None
    education_level: Optional[str] = None
    split: Optional[str] = "train"  # 'train', 'val', or 'test'


class MatchResult(BaseModel):
    """Output schema for a 1-to-1 resume against job match."""
    score: float = Field(..., ge=0.0, le=100.0, description="Match score from 0 to 100")
    matched_skills: List[str] = Field(default_factory=list)
    missing_skills: List[str] = Field(default_factory=list)
    explanation: str = Field(..., description="Explainable justification of the match")
    title: Optional[str] = None
    company: Optional[str] = None


class RankedJobItem(BaseModel):
    """Schema for a single job inside top-K ranked results."""
    rank: int
    job_id: str
    title: str
    company: Optional[str] = "Unknown"
    location: Optional[str] = "Remote"
    score: float
    matched_skills: List[str] = Field(default_factory=list)
    missing_skills: List[str] = Field(default_factory=list)
    explanation: str
