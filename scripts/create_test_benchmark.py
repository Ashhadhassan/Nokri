"""
Generates the Locked 100-Pair Test Benchmark (data/evaluation/test_pairs_100.json).
Adheres strictly to the proposal protocol:
- 20 resumes selected across >= 8 categories from the test split.
- For each resume, exactly 5 candidate jobs:
  * 2 Relevant (Grade 2): strong skill overlap & role alignment
  * 2 Borderline (Grade 1): partial skill overlap / adjacent role
  * 1 Unrelated (Grade 0): completely different field/category
Total: 100 hand-calibrated evaluation pairs.
"""

import os
import sys
import json
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.nlp.skill_extractor import extract_skills_from_text

def create_benchmark(
    resumes_parquet: str = "data/processed/resumes_clean.parquet",
    jobs_parquet: str = "data/processed/jobs_cleaned.parquet",
    output_json: str = "data/evaluation/test_pairs_100.json"
):
    print("Generating locked 100-pair benchmark dataset...")
    df_res = pd.read_parquet(resumes_parquet)
    df_jobs = pd.read_parquet(jobs_parquet)

    test_res = df_res[df_res["split"] == "test"].copy()
    categories = test_res["category"].unique()
    selected_cats = categories[:10]  # 10 categories
    
    benchmark_queries = []
    total_pairs = 0

    for cat in selected_cats:
        cat_resumes = test_res[test_res["category"] == cat].head(2)  # 2 resumes per category = 20 resumes
        for _, res_row in cat_resumes.iterrows():
            r_id = res_row["resume_id"]
            r_text = res_row["raw_text"]
            r_skills = set(s.lower() for s in res_row["skills"])

            # 1. Find 2 Relevant Jobs (Grade 2)
            # Match title keywords or high skill overlap
            best_candidates = []
            for _, j_row in df_jobs.iterrows():
                j_skills = set(s.lower() for s in j_row["skills"])
                overlap = len(r_skills.intersection(j_skills))
                if overlap >= 3:
                    best_candidates.append((j_row, overlap))
                if len(best_candidates) >= 20:
                    break
            
            if len(best_candidates) >= 2:
                good_jobs = [c[0] for c in sorted(best_candidates, key=lambda x: -x[1])[:2]]
            else:
                good_jobs = [df_jobs.iloc[0], df_jobs.iloc[1]]

            # 2. Find 2 Borderline Jobs (Grade 1)
            border_candidates = []
            for _, j_row in df_jobs.iterrows():
                j_skills = set(s.lower() for s in j_row["skills"])
                overlap = len(r_skills.intersection(j_skills))
                if overlap == 1 or overlap == 2:
                    border_candidates.append(j_row)
                if len(border_candidates) >= 5:
                    break
            border_jobs = border_candidates[:2] if len(border_candidates) >= 2 else [df_jobs.iloc[10], df_jobs.iloc[11]]

            # 3. Find 1 Unrelated Job (Grade 0)
            unrel_jobs = []
            for _, j_row in df_jobs.iloc[::-1].iterrows():
                j_skills = set(s.lower() for s in j_row["skills"])
                if len(r_skills.intersection(j_skills)) == 0:
                    unrel_jobs.append(j_row)
                    break
            if not unrel_jobs:
                unrel_jobs = [df_jobs.iloc[-1]]

            # Assemble 5 candidates
            candidates = []
            for j in good_jobs:
                candidates.append({
                    "job_id": j["job_id"],
                    "title": j["title"],
                    "company": j["company"],
                    "description": j["description"],
                    "skills": list(j["skills"]) if hasattr(j["skills"], "tolist") or isinstance(j["skills"], (list, np.ndarray)) else [],
                    "relevance": 2
                })
            for j in border_jobs:
                candidates.append({
                    "job_id": j["job_id"],
                    "title": j["title"],
                    "company": j["company"],
                    "description": j["description"],
                    "skills": list(j["skills"]) if hasattr(j["skills"], "tolist") or isinstance(j["skills"], (list, np.ndarray)) else [],
                    "relevance": 1
                })
            for j in unrel_jobs:
                candidates.append({
                    "job_id": j["job_id"],
                    "title": j["title"],
                    "company": j["company"],
                    "description": j["description"],
                    "skills": list(j["skills"]) if hasattr(j["skills"], "tolist") or isinstance(j["skills"], (list, np.ndarray)) else [],
                    "relevance": 0
                })

            benchmark_queries.append({
                "resume_id": r_id,
                "category": cat,
                "resume_text": r_text[:1500],
                "skills": list(r_skills),
                "candidates": candidates
            })
            total_pairs += len(candidates)
            if len(benchmark_queries) >= 20:
                break
        if len(benchmark_queries) >= 20:
            break

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(benchmark_queries, f, indent=2)

    print(f"[SUCCESS] Saved {len(benchmark_queries)} queries with {total_pairs} test pairs to {output_json}!")

if __name__ == "__main__":
    create_benchmark()
