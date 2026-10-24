# 🎯 Resume-to-Job Matcher: Semantic Search & Explainable Re-Ranking

An end-to-end Machine Learning information retrieval system that scores resume-to-job fit, ranks top job opportunities from a corpus of 50,000+ postings, and provides natural-language explainability for missing skills.

Built to run comfortably within **free-tier CPU constraints** (1 CPU, 1–2 GB RAM) with **sub-second retrieval latency**.

---

## ⚡ Performance Highlights
- **TF-IDF Baseline Latency:** ~7.0 ms
- **FAISS Dense Retrieval (MiniLM):** ~639 ms
- **Two-Stage LightGBM LambdaRank Latency:** ~422 ms (Target: < 2,000 ms)
- **Top-5 Relevance Precision (P@5):** `0.8000`
- **Unified Skills Taxonomy:** 108,000+ mapped aliases and 22,500+ canonical skills (ESCO v1.2.1 + O*NET)
- **Zero Cost Deployment:** Packaged in a multi-stage Docker container ready for Hugging Face Spaces & Render

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph Data_Pipeline["1. Offline Data Pipeline (Multi-Threaded)"]
        RawJobs["Raw Jobs (1.7 GB)"] -->|Parallel Chunking (12 Cores)| CleanJobs["jobs_cleaned.parquet (50k rows)"]
        RawResumes["LiveCareer Resumes"] -->|PII Sanitization & Split| CleanResumes["resumes_clean.parquet (2.4k rows)"]
        Taxonomy["ESCO + O*NET Taxonomy"] -->|O(1) N-Gram Lookup| CleanSkills["skills_taxonomy.json (108k terms)"]
        CleanJobs -->|Scikit-Learn TF-IDF| TFIDFModel["models/tfidf_ranker.pkl"]
        CleanJobs -->|all-MiniLM-L6-v2 Embeddings| FAISSIndex["models/faiss_index.bin (50k vectors)"]
        HFTrain["HuggingFace Fit Pairs"] -->|5 Feature Signals| LGBMRanker["models/lgbm_ranker.txt"]
    end

    subgraph Serving["2. Real-Time Inference & Explainability"]
        UserResume["Resume (PDF / DOCX / Text)"] --> Parser["Document Parser (pdfplumber)"]
        Parser --> Extractor["N-Gram Skill Extractor"]
        Extractor --> TwoStage["Two-Stage Ranker (FAISS Top 50 + LightGBM Re-Ranker)"]
        TwoStage --> Candidates["Top 10 Ranked Jobs"]
        Candidates & Extractor --> Explainer["Skill Gap & Justification Engine"]
        Explainer --> API["FastAPI REST Endpoints (/docs)"]
        Explainer --> UI["Streamlit Interactive Web App"]
    end
```

---

## 📂 Repository Structure
```
├── data/
│   ├── processed/
│   │   ├── jobs_cleaned.parquet      # 50,000 clean, deduplicated job postings
│   │   ├── resumes_clean.parquet     # 2,483 anonymized resumes (train/val/test)
│   │   └── skills_taxonomy.json      # 108k mapped skills & synonyms
│   ├── evaluation/
│   │   └── test_pairs_100.json       # Locked 100-pair benchmark dataset
│   └── sample_resumes/               # 30 curated test PDFs across 15 categories
├── backend/
│   ├── src/
│   │   ├── api/                      # FastAPI REST service & Pydantic schemas
│   │   ├── matching/                 # Skill gap analysis & natural language explainer
│   │   ├── nlp/                      # Sub-millisecond skill extractor & taxonomy
│   │   ├── parser/                   # PDF, DOCX, and text resume parsers
│   │   ├── ranking/                  # BaseJobRanker, TFIDF, FAISS, and LightGBM
│   │   └── schema.py                 # Shared Pydantic data schemas
│   ├── scripts/                      # Training, ingestion, and evaluation scripts
│   ├── tests/                        # Pytest suite (8 unit & integration tests)
│   ├── Dockerfile                    # Backend Dockerfile
│   └── requirements.txt              # Backend dependencies
├── frontend/
│   ├── app.py                        # Streamlit Interactive Web Application
│   ├── Dockerfile                    # Frontend Dockerfile
│   └── requirements.txt              # Frontend dependencies
├── data/
│   ├── processed/
│   │   ├── jobs_cleaned.parquet      # 50,000 clean, deduplicated job postings
│   │   ├── resumes_clean.parquet     # 2,483 anonymized resumes (train/val/test)
│   │   └── skills_taxonomy.json      # 108k mapped skills & synonyms
│   ├── evaluation/
│   │   └── test_pairs_100.json       # Locked 100-pair benchmark dataset
│   └── sample_resumes/               # 30 curated test PDFs across 15 categories
├── models/
│   ├── tfidf_ranker.pkl              # Fitted TF-IDF baseline model artifact
│   ├── faiss_index.bin               # 50,000-vector FAISS IndexFlatIP
│   ├── faiss_metadata.parquet        # Metadata for vector search
│   └── lgbm_ranker.txt               # Trained LightGBM Re-Ranker booster
├── docker-compose.yml                # Multi-container orchestration
└── requirements.txt                  # Consolidated dependencies
```

---

## 📊 Benchmark Evaluation (Locked 100 Test Pairs)
Evaluated across 20 query sessions (100 pairs) representing 10 diverse job categories:

| Model Architecture | NDCG@5 | P@5 | MRR (Grade 2) | Latency p95 |
| :--- | :--- | :--- | :--- | :--- |
| **TF-IDF Baseline Matcher** | `0.8867` | `0.8000` | `0.7792` | **7.0 ms** |
| **Dense Semantic (FAISS)** | `0.8224` | `0.8000` | `0.5083` | **639.9 ms** |
| **Two-Stage LightGBM Re-Ranker** | `0.8323` | `0.8000` | `0.5583` | **422.0 ms** |

---

## 🚀 Quickstart & Reproduction

### 1. Run Automated Test Suite
```bash
pytest -v backend/tests/test_pipeline.py
```
*(All 8 tests pass)*

### 2. Launch the Interactive Web App (Streamlit)
```bash
streamlit run frontend/app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser to:
- Test 1-to-1 matching with visual skill pill badges.
- Retrieve Top-10 job recommendations across the 50,000 jobs index.
- Switch between **Two-Stage LightGBM**, **Dense FAISS**, or **TF-IDF Baseline** on the fly.

### 3. Launch the FastAPI Backend
```bash
uvicorn backend.src.api.main:app --reload --port 8000
```
Explore the interactive Swagger API documentation at [http://localhost:8000/docs](http://localhost:8000/docs).

### 4. Run with Docker Compose
```bash
docker compose up --build
```

---

## 🛡️ Privacy, Fairness & Limitations
- **Zero Raw Data Retention:** Resumes are parsed in memory; temporary files are automatically purged immediately after text extraction.
- **PII Scrubbing:** Phone numbers, email addresses, and URLs are scrubbed from training datasets with zero residual leaks observed in sample audits.
- **Fairness & Demographic Invariance:** Evaluated under candidate name and gender perturbations (`John Smith` vs. `Jane Smith` vs. `Muhammad Ali` vs. anonymous). Yielded **`0.0000` score drift**, confirming demographic invariance.
- **Advisory Role:** This system is engineered as a candidate empowerment and skill-gap identification tool, never as an automated candidate filtering/rejection engine.
- **Independent Audit:** Full C1–C5 review report available in [`docs/INDEPENDENT_AUDIT_REPORT.md`](docs/INDEPENDENT_AUDIT_REPORT.md).

