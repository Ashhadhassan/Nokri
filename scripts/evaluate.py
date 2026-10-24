"""
Comparative Benchmark Evaluation Pipeline.
Evaluates all three architectures on the locked 100-pair benchmark (data/evaluation/test_pairs_100.json):
1. TF-IDF Baseline Matcher
2. FAISS Dense Semantic Retriever (Sentence-Transformers all-MiniLM-L6-v2)
3. Two-Stage LightGBM Re-Ranker
Computes NDCG@10, P@5, MRR, and Latency (p95).
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from tabulate import tabulate

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.ranking.tfidf_ranker import TFIDFJobRanker
from src.ranking.faiss_retriever import FAISSDenseRetriever
from src.ranking.lgbm_ranker import TwoStageLGBMRanker

def dcg_at_k(r: list, k: int = 10) -> float:
    r = np.asarray(r, dtype=float)[:k]
    if r.size:
        return float(np.sum(r / np.log2(np.arange(2, r.size + 2))))
    return 0.0

def ndcg_at_k(r: list, k: int = 10) -> float:
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.0
    return float(dcg_at_k(r, k) / dcg_max)

def precision_at_k(r: list, k: int = 5, relevance_thresh: int = 1) -> float:
    r = r[:k]
    if not r:
        return 0.0
    hits = sum(1 for x in r if x >= relevance_thresh)
    return float(hits / k)

def mrr(r: list, relevance_thresh: int = 1) -> float:
    for idx, val in enumerate(r, start=1):
        if val >= relevance_thresh:
            return 1.0 / idx
    return 0.0

def evaluate_models(
    benchmark_json: str = "data/evaluation/test_pairs_100.json"
):
    print("=" * 70)
    print("OFFICIAL BENCHMARK EVALUATION (LOCKED 100 TEST PAIRS)")
    print("=" * 70)

    with open(benchmark_json, "r", encoding="utf-8") as f:
        queries = json.load(f)

    print(f"Loaded {len(queries)} evaluation query sessions ({sum(len(q['candidates']) for q in queries)} pairs).")

    print("\nLoading models...")
    tfidf = TFIDFJobRanker.load()
    faiss_retriever = FAISSDenseRetriever()
    lgbm_ranker = TwoStageLGBMRanker(retriever=faiss_retriever, tfidf=tfidf)

    models = {
        "TF-IDF Baseline": tfidf,
        "Dense Semantic (FAISS)": faiss_retriever,
        "Two-Stage LightGBM Ranker": lgbm_ranker
    }

    results_table = []
    baseline_ndcg = None

    for name, model in models.items():
        ndcg_list = []
        p5_list = []
        mrr_list = []
        latencies = []

        for q in queries:
            resume_text = q["resume_text"]
            candidates = q["candidates"]

            t0 = time.time()
            scores = []
            for cand in candidates:
                job_desc = cand["title"] + " " + cand["description"]
                score = model.score_pair(resume_text, job_desc)
                scores.append(score)
            
            elapsed_ms = (time.time() - t0) * 1000
            latencies.append(elapsed_ms)

            # Sort true labels according to model scores
            true_labels = [c["relevance"] for c in candidates]
            ranked_labels = [rel for _, rel in sorted(zip(scores, true_labels), reverse=True)]

            ndcg_list.append(ndcg_at_k(ranked_labels, k=5))
            p5_list.append(precision_at_k(ranked_labels, k=5, relevance_thresh=1))
            mrr_list.append(mrr(ranked_labels, relevance_thresh=2))  # MRR for top grade 2

        avg_ndcg = float(np.mean(ndcg_list))
        avg_p5 = float(np.mean(p5_list))
        avg_mrr = float(np.mean(mrr_list))
        p95_lat = float(np.percentile(latencies, 95))

        if baseline_ndcg is None:
            baseline_ndcg = avg_ndcg
            gain_str = "Baseline Ref"
        else:
            pct_gain = ((avg_ndcg - baseline_ndcg) / baseline_ndcg) * 100.0
            gain_str = f"+{pct_gain:.1f}% vs Baseline"

        results_table.append({
            "Model Architecture": name,
            "NDCG@5": f"{avg_ndcg:.4f}",
            "NDCG Gain": gain_str,
            "P@5": f"{avg_p5:.4f}",
            "MRR (Grade 2)": f"{avg_mrr:.4f}",
            "Latency p95 (ms)": f"{p95_lat:.1f} ms"
        })

    summary_df = pd.DataFrame(results_table)
    table_str = tabulate(summary_df, headers="keys", tablefmt="github", showindex=False)
    print("\n" + table_str)

    # Save to markdown report in docs
    os.makedirs("docs", exist_ok=True)
    with open("docs/benchmark_results.md", "w", encoding="utf-8") as f:
        f.write("# Model Benchmark Evaluation Results\n\n")
        f.write(table_str + "\n")

    return summary_df

if __name__ == "__main__":
    evaluate_models()
