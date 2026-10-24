# Model Benchmark Evaluation Results

| Model Architecture        |   NDCG@5 | NDCG Gain          |   P@5 |   MRR (Grade 2) | Latency p95 (ms)   |
|---------------------------|----------|--------------------|-------|-----------------|--------------------|
| TF-IDF Baseline           |   0.8867 | Baseline Ref       |   0.8 |          0.7792 | 7.0 ms             |
| Dense Semantic (FAISS)    |   0.8224 | +-7.3% vs Baseline |   0.8 |          0.5083 | 639.9 ms           |
| Two-Stage LightGBM Ranker |   0.8323 | +-6.1% vs Baseline |   0.8 |          0.5583 | 422.0 ms           |
