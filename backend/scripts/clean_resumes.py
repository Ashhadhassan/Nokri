"""
Parallel Resume Cleaning and PII Anonymization.
Uses ThreadPoolExecutor across available CPU cores to sanitize text,
extract skills, and partition train/val/test splits into data/processed/resumes_clean.parquet.
"""

import os
import sys
import re
import time
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.nlp.skill_extractor import extract_skills_from_text

def mask_pii(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "[EMAIL_REDACTED]", text)
    text = re.sub(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", "[PHONE_REDACTED]", text)
    text = re.sub(r"https?://\S+|www\.\S+", "[URL_REDACTED]", text)
    return re.sub(r"\s+", " ", text).strip()

def process_single_resume(row: dict) -> dict:
    raw_text = str(row.get("Resume_str", ""))
    sanitized = mask_pii(raw_text)
    words = sanitized.split()
    if len(words) < 20:
        return None
    skills = extract_skills_from_text(sanitized)
    return {
        "resume_id": str(row.get("ID")),
        "category": str(row.get("Category", "General")).strip(),
        "raw_text": sanitized,
        "skills": skills,
        "word_count": len(words)
    }

def clean_resumes_parallel(
    input_csv: str = "Resume/Resume.csv",
    output_parquet: str = "data/processed/resumes_clean.parquet",
    max_workers: int = None
):
    if max_workers is None:
        max_workers = os.cpu_count() or 8

    print(f"Parallel Resume Cleaning starting (Threads: {max_workers})...")
    t0 = time.time()

    df = pd.read_csv(input_csv)
    rows = df.to_dict(orient="records")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_single_resume, rows))

    cleaned = [r for r in results if r is not None]
    clean_df = pd.DataFrame(cleaned)

    # Deterministic split: 70% Train, 15% Val, 15% Test
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(clean_df))
    n = len(clean_df)
    train_end = int(0.70 * n)
    val_end = int(0.85 * n)

    splits = ["train"] * n
    for i in range(train_end, val_end):
        splits[shuffled_indices[i]] = "val"
    for i in range(val_end, n):
        splits[shuffled_indices[i]] = "test"

    clean_df["split"] = splits

    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
    clean_df.to_parquet(output_parquet, index=False)
    elapsed = time.time() - t0

    print(f"[SUCCESS] Cleaned {len(clean_df)} resumes in {elapsed:.2f} seconds!")
    print(f"Saved to: {output_parquet}")
    print(f"Split distribution:\n{clean_df['split'].value_counts()}")

if __name__ == "__main__":
    clean_resumes_parallel()
