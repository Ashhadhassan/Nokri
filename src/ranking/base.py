"""
Abstract Base Class defining the contract for all Job Rankers in the project.
All implementations (TF-IDF, FAISS Dense Retriever, LightGBM Re-Ranker) adhere to this.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseJobRanker(ABC):
    """Abstract interface for scoring and ranking jobs given a resume."""

    @abstractmethod
    def fit(self, jobs_df: Any) -> None:
        """
        Fit or index the ranking model on a corpus of cleaned jobs.
        :param jobs_df: pandas DataFrame or path to parquet containing cleaned jobs.
        """
        pass

    @abstractmethod
    def score_pair(self, resume_text: str, job_description: str) -> float:
        """
        Calculates a relevance/match score between a resume and a single job description.
        :param resume_text: Plain text of candidate resume.
        :param job_description: Plain text of job description.
        :return: Float score between 0.0 and 100.0.
        """
        pass

    @abstractmethod
    def rank_jobs(
        self,
        resume_text: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves and ranks the top_k most relevant jobs for a given resume.
        :param resume_text: Plain text of candidate resume.
        :param top_k: Number of jobs to return (default 10).
        :param filters: Optional metadata filters (e.g. {'location': 'Remote'}).
        :return: List of dicts with keys: ['job_id', 'title', 'company', 'location', 'score']
        """
        pass
