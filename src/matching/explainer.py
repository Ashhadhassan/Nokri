"""
Skill Gap Analysis and Explanation Engine.
Computes matched skills, missing skills, and plain-language justifications for match scores.
Truthful and strictly grounded in textual evidence.
"""

from typing import List, Dict, Set, Any
from src.schema import MatchResult

class SkillExplainer:
    def explain_match(
        self,
        score: float,
        resume_skills: List[str],
        job_skills: List[str],
        job_title: str = "Target Role",
        company: str = ""
    ) -> MatchResult:
        res_set: Set[str] = set(resume_skills)
        job_set: Set[str] = set(job_skills)

        # Matched and missing
        matched = sorted(list(res_set.intersection(job_set)))
        missing = sorted(list(job_set.difference(res_set)))

        # Natural language summary
        if not job_skills:
            explanation = f"General profile match for {job_title} based on semantic text similarity."
        elif len(matched) == 0:
            top_missing = ", ".join(missing[:4])
            explanation = f"Low skill overlap. Missing key required qualifications: {top_missing}."
        elif len(missing) == 0:
            top_matched = ", ".join(matched[:4])
            explanation = f"Excellent skill alignment! Candidate demonstrates core skills: {top_matched}."
        else:
            top_matched = ", ".join(matched[:4])
            top_missing = ", ".join(missing[:3])
            overlap_pct = (len(matched) / len(job_set)) * 100
            explanation = (
                f"Solid fit ({overlap_pct:.0f}% skill coverage). Demonstrates {top_matched}. "
                f"Could strengthen profile by acquiring or highlighting: {top_missing}."
            )

        return MatchResult(
            score=round(score, 1),
            matched_skills=matched,
            missing_skills=missing,
            explanation=explanation,
            title=job_title,
            company=company
        )

# Global helper
_explainer = SkillExplainer()

def explain_match(
    score: float,
    resume_skills: List[str],
    job_skills: List[str],
    job_title: str = "Target Role",
    company: str = ""
) -> MatchResult:
    return _explainer.explain_match(score, resume_skills, job_skills, job_title, company)
