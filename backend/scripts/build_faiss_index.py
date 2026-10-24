"""
Builds the FAISS Vector Index for 50,000 Cleaned Jobs.
Uses Sentence-Transformers (all-MiniLM-L6-v2) to generate normalized 384-d dense embeddings.
Saves to models/faiss_index.bin and models/faiss_metadata.parquet.
"""

import os
import sys
import time
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.ranking.faiss_retriever import FAISSDenseRetriever

def build_index(
    jobs_parquet: str = "data/processed/jobs_cleaned.parquet",
    batch_size: int = 256
):
    print("=" * 60)
    print("BUILDING FAISS DENSE VECTOR INDEX")
    print("=" * 60)
    t0 = time.time()
    
    retriever = FAISSDenseRetriever()
    df = pd.read_parquet(jobs_parquet)
    print(f"Loaded {len(df)} jobs from {jobs_parquet}.")

    retriever.fit(df)
    elapsed = time.time() - t0
    print(f"\n[SUCCESS] FAISS Index built and saved in {elapsed:.2f} seconds!")
    print(f"Total vectors indexed: {retriever.index.ntotal}")

if __name__ == "__main__":
    build_index()
