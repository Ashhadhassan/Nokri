"""
High-Throughput Parallel Job Ingestion & Normalization.
Uses ThreadPoolExecutor across all available CPU cores to parse, clean, and standardize
50,000 job descriptions into data/processed/jobs_cleaned.parquet in seconds.
"""

import os
import sys
import csv
import re
import time
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.nlp.skill_extractor import extract_skills_from_text

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def process_single_job(row: dict) -> dict:
    """Processes a single raw job row. Thread-safe."""
    title = clean_text(row.get("Job Title", ""))
    company = clean_text(row.get("Company", ""))
    job_id = str(row.get("Job Id", "")).strip()

    if not title or len(title) < 3 or not company or not job_id:
        return None

    raw_desc = clean_text(row.get("Job Description", ""))
    responsibilities = clean_text(row.get("Responsibilities", ""))
    raw_skills = clean_text(row.get("skills", ""))
    exp = clean_text(row.get("Experience", ""))
    location = clean_text(row.get("location", ""))
    country = clean_text(row.get("Country", ""))
    role = clean_text(row.get("Role", ""))

    if responsibilities:
        full_desc = f"{raw_desc} Key Responsibilities: {responsibilities}"
    else:
        full_desc = raw_desc

    words = full_desc.split()
    if len(words) < 40:
        return None

    # Fast skill extraction
    skill_text = f"{raw_skills} {title} {role}"
    extracted_skills = extract_skills_from_text(skill_text)
    if raw_skills:
        for token in re.split(r"[,;•\n]+", raw_skills):
            t = token.strip()
            if t and len(t) >= 2 and t not in extracted_skills and len(t) <= 40:
                extracted_skills.append(t)

    loc_str = f"{location}, {country}".strip(", ")

    return {
        "job_id": job_id,
        "title": title,
        "role": role if role else title,
        "description": full_desc,
        "skills": extracted_skills,
        "experience": exp if exp else "Not specified",
        "location": loc_str if loc_str else "Remote",
        "company": company,
        "word_count": len(words)
    }

def process_batch(batch: list) -> list:
    """Processes a batch of raw rows."""
    results = []
    for r in batch:
        res = process_single_job(r)
        if res is not None:
            results.append(res)
    return results

def ingest_jobs_parallel(
    input_csv: str = "job_descriptions.csv",
    output_parquet: str = "data/processed/jobs_cleaned.parquet",
    target_count: int = 50000,
    batch_size: int = 2000,
    max_workers: int = None
):
    if max_workers is None:
        max_workers = os.cpu_count() or 8

    print(f"Parallel Job Ingestion starting...")
    print(f"Source: {input_csv} | Target: {target_count} jobs | CPU Cores / Threads: {max_workers}")

    t0 = time.time()
    seen_ids = set()
    cleaned_records = []
    total_scanned = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        current_batch = []
        pending_futures = []

        with open(input_csv, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_scanned += 1
                current_batch.append(row)

                if len(current_batch) >= batch_size:
                    # Submit batch to thread pool
                    future = executor.submit(process_batch, current_batch)
                    pending_futures.append(future)
                    current_batch = []

                    # Harvest completed batches if queue has several batches
                    if len(pending_futures) >= max_workers * 2:
                        for fut in pending_futures:
                            batch_results = fut.result()
                            for rec in batch_results:
                                jid = rec["job_id"]
                                if jid not in seen_ids:
                                    seen_ids.add(jid)
                                    cleaned_records.append(rec)
                                    if len(cleaned_records) >= target_count:
                                        break
                            if len(cleaned_records) >= target_count:
                                break
                        pending_futures = []
                        print(f"  -> Collected {len(cleaned_records)}/{target_count} jobs ({total_scanned} rows scanned)...")
                        if len(cleaned_records) >= target_count:
                            break

            # Flush any remaining batches
            if current_batch and len(cleaned_records) < target_count:
                pending_futures.append(executor.submit(process_batch, current_batch))

            for fut in pending_futures:
                if len(cleaned_records) >= target_count:
                    break
                batch_results = fut.result()
                for rec in batch_results:
                    jid = rec["job_id"]
                    if jid not in seen_ids:
                        seen_ids.add(jid)
                        cleaned_records.append(rec)
                        if len(cleaned_records) >= target_count:
                            break

    elapsed = time.time() - t0
    print(f"\nIngestion completed in {elapsed:.2f} seconds!")
    print(f"Total jobs saved: {len(cleaned_records)} (scanned {total_scanned} rows)")

    df = pd.DataFrame(cleaned_records)
    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
    df.to_parquet(output_parquet, index=False)
    
    file_size_mb = os.path.getsize(output_parquet) / (1024 * 1024)
    print(f"[SUCCESS] Saved to {output_parquet} ({file_size_mb:.2f} MB)")
    print(f"Sample row:\n{df[['job_id', 'title', 'company', 'skills']].head(2)}")

if __name__ == "__main__":
    ingest_jobs_parallel()
