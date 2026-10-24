"""
Fits and serializes the TF-IDF Baseline model on data/processed/jobs_cleaned.parquet.
Measures pair scoring latency and Top-10 ranking latency.
"""

import os
import sys
import time
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.ranking.tfidf_ranker import TFIDFJobRanker

def run_baseline_pipeline(
    jobs_parquet: str = "data/processed/jobs_cleaned.parquet",
    resumes_parquet: str = "data/processed/resumes_clean.parquet",
    model_dest: str = "models/tfidf_ranker.pkl"
):
    print("=" * 60)
    print("TF-IDF BASELINE PIPELINE")
    print("=" * 60)

    # 1. Fit model
    t0 = time.time()
    ranker = TFIDFJobRanker(max_features=30000, ngram_range=(1, 2))
    ranker.fit(jobs_parquet)
    fit_time = time.time() - t0
    print(f"Fit completed in {fit_time:.2f} seconds.")

    # 2. Benchmark latency with sample resume
    res_df = pd.read_parquet(resumes_parquet)
    sample_resume = res_df.iloc[0]["raw_text"]
    sample_job_desc = ranker.jobs_metadata[0]["title"] + " " + " ".join(ranker.jobs_metadata[0]["skills"])

    # Test pair score
    t_pair_0 = time.time()
    pair_score = ranker.score_pair(sample_resume, sample_job_desc)
    t_pair = (time.time() - t_pair_0) * 1000
    print(f"\n[LATENCY TEST] 1-to-1 Pair Match Score: {pair_score}/100 ({t_pair:.2f} ms)")

    # Test top-10 ranking over 50,000 jobs
    t_rank_0 = time.time()
    top10 = ranker.rank_jobs(sample_resume, top_k=10)
    t_rank = (time.time() - t_rank_0) * 1000
    print(f"[LATENCY TEST] Top-10 Retrieval across 50,000 jobs: {t_rank:.2f} ms")

    print("\nTop 3 Ranked Jobs:")
    for j in top10[:3]:
        print(f"  #{j['rank']} [{j['score']}/100] {j['title']} @ {j['company']} ({j['location']})")

    # 3. Save model
    ranker.save(model_dest)
    print(f"\n[SUCCESS] Baseline TF-IDF model saved to {model_dest} ({os.path.getsize(model_dest)/(1024*1024):.2f} MB)")

if __name__ == "__main__":
    run_baseline_pipeline()
