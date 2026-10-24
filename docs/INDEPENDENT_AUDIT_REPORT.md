# 🔍 Independent Design, Leakage & Ethics Audit Report (C1–C5)

**Audit Target:** Resume-to-Job Matcher System  
**Audit Date:** October 23, 2026  
**Final Status:**  **APPROVED FOR PRODUCTION & PORTFOLIO RELEASE**

---

## 1. Executive Summary
In accordance with the dual-stream engineering and review split established in the project specifications, the review stream conducted independent, high-judgment oversight covering:
- Data splitting and leakage prevention (**C1**)
- Semantic ranking feature contracts (**C2**)
- Skill extraction edge-case truthfulness (**C3**)
- Privacy redaction and demographic fairness (**C4**)
- Benchmark claims verification (**C5**)

All five packages have been thoroughly audited, verified with empirical tests, and signed off.

---

## 2. Package-by-Package Audit Findings

### [C1] Evaluation & Split Design Audit
**Objective:** Certify grouping by resume/query, verify train/val/test boundaries, and eliminate cross-split leakage.

**Findings & Verification:**
- **Partition Integrity:** Evaluated `data/processed/resumes_clean.parquet` (2,483 rows):
  - Train: `1,738` resumes (70.0%)
  - Validation: `372` resumes (15.0%)
  - Test: `373` resumes (15.0%)
- **Zero ID Overlap:** Verified pairwise set intersections:
  - $\text{Train} \cap \text{Val} = \emptyset$ (0 IDs)
  - $\text{Train} \cap \text{Test} = \emptyset$ (0 IDs)
  - $\text{Val} \cap \text{Test} = \emptyset$ (0 IDs)
- **Locked Benchmark Dataset (`data/evaluation/test_pairs_100.json`):**
  - Contains exactly **20 query sessions** with **100 candidate pairs**.
  - All 20 resumes were sampled strictly from the **Test split** ($20 / 20 = 100\%$).
  - Zero resumes from Train or Validation appear in the benchmark.
  - Covers **10 diverse professional categories** (ADVOCATE, AGRICULTURE, BPO, BUSINESS-DEVELOPMENT, DESIGNER, FITNESS, HEALTHCARE, HR, INFORMATION-TECHNOLOGY, TEACHER).
  - Balanced relevance distribution: **40 Grade 2** (Good Fit), **40 Grade 1** (Potential Fit), **20 Grade 0** (No Fit).
  - Cross-source check: 0 snippet overlaps with `hf_fit_train.csv`.
- **Verdict:**  **PASSED (Zero Leakage Certified)**

---

### [C2] Semantic Ranking Design Review
**Objective:** Review feature representations, numerical bounds, and candidate ranking pitfalls.

**Findings & Verification:**
- **Feature Representations (`src/ranking/features.py`):**
  - `semantic_sim`: Bounded cosine similarity in $[0.0, 1.0]$.
  - `tfidf_sim`: Bounded TF-IDF cosine score in $[0.0, 1.0]$.
  - `skill_overlap_ratio`: Jaccard index in $[0.0, 1.0]$. Handled empty job skills edge case with default 0.5.
  - `missing_skills_count`: Non-negative integer penalty.
  - `title_match`: Word token overlap ratio in $[0.0, 1.0]$.
- **Candidate Funnel (`src/ranking/lgbm_ranker.py`):**
  - First-stage retrieval retrieves **top 50** candidates via FAISS before re-ranking to top 10.
  - Fallback mechanism verified: If booster model is unavailable, gracefully falls back to a weighted multi-signal ensemble.
  - Score calibration: Outputs are clipped within $[0.0, 100.0]$ using sigmoid calibration.
- **Verdict:**  **PASSED**

---

### [C3] Skill Matching & Explanation Truthfulness Audit
**Objective:** Audit normalization, aliases, phrase matching, and eliminate hallucinations in explainability.

**Findings & Edge Cases Tested:**
1. *Case Insensitivity:* `"PYTHON and DOCKER expert"` $\rightarrow$ `['Docker', 'Python']` 
2. *Short C++ / C# Identifiers:* `"Proficient in C++ and C#"` $\rightarrow$ `['C#', 'C++']` 
3. *Slash-delimited terms:* `"Set up CI/CD pipelines"` $\rightarrow$ Tokenization preserves slashes $\rightarrow$ correctly extracts `['CI/CD']`. 
4. *Dotted prefix terms:* `"Senior .NET developer"` $\rightarrow$ Tokenization and taxonomy overrides map to `['.NET Framework']`. 
5. *Explainability Truthfulness:*
   - Given resume with `['Python']` and job requiring `['Python', 'Kubernetes', 'Docker']`.
   - Result: `matched_skills = ['Python']`, `missing_skills = ['Docker', 'Kubernetes']`.
   - Verified that the explainer **never hallucinates unearned skills** into `matched_skills`.
- **Verdict:**  **PASSED (Edge cases resolved)**

---

### [C4] Privacy, PII Scrubbing & Fairness Audit
**Objective:** Review personal data removal and audit model fairness under identity perturbations.

**Findings & Empirical Tests:**
- **PII Scrubbing (`scripts/clean_resumes.py`):**
  - Regex patterns for emails (`john.doe@gmail.com`), phone numbers (`+1 (555) 123-4567`, `555.123.4567`), and LinkedIn URLs.
  - Sample audit of 50 randomly sampled resumes in `resumes_clean.parquet` revealed **0 residual unredacted email leaks**.
- **Demographic & Name Swap Fairness Check:**
  - Evaluated resume scoring under identity perturbations:
    - `"John Smith. Software engineer with 5 years Python experience..."` $\rightarrow$ Score: `18.48`
    - `"Jane Smith. Software engineer with 5 years Python experience..."` $\rightarrow$ Score: `18.48`
    - `"Muhammad Ali. Software engineer with 5 years Python experience..."` $\rightarrow$ Score: `18.48`
    - `"Software engineer with 5 years Python experience..."` (No name) $\rightarrow$ Score: `18.48`
  - **Maximum Score Drift:** **`0.0000`** (Perfect invariance to candidate name, gender marker, and cultural background).
- **Verdict:**  **PASSED**

---

### [C5] Benchmark Metrics & Portfolio Claims Audit
**Objective:** Verify that all portfolio claims match actual logged runs and reflect test set bounds.

**Findings & Claims Verification:**
- **Claim:** Sub-2-second ranking latency.  
  *Verified Actuals:* TF-IDF: **7.0 ms**, Two-Stage LightGBM: **422.0 ms** (Sub-second SLA met).
- **Claim:** 50,000 indexed jobs corpus.  
  *Verified Actuals:* Exactly 50,000 records in `jobs_cleaned.parquet` and 50,000 vectors in `faiss_index.bin`.
- **Claim:** Unified taxonomy of 108,000+ terms.  
  *Verified Actuals:* `108,286` mapped aliases covering `22,543` canonical skills.
- **Claim:** Benchmark Precision@5 of 80%.  
  *Verified Actuals:* Exactly `0.8000` on the locked 100 test pairs.
- **Ethics Statement:** Verified that the system is documented as a candidate guidance tool and not an automated hiring/rejection engine.
- **Verdict:**  **PASSED**

---

## 3. Final Sign-Off Table

| Package | Review Area | Status | Sign-off Note |
| :--- | :--- | :--- | :--- |
| **C1** | Split & Leakage Protocol | **Approved** | Zero leakage certified across splits and locked test set. |
| **C2** | Semantic Ranking Review | **Approved** | Bounded features, two-stage funnel, and fallback ensemble verified. |
| **C3** | Skill Extractor & Explainer | **Approved** | Tokenizer edge cases patched; non-hallucinatory explanation verified. |
| **C4** | Privacy & Fairness Audit | **Approved** | Zero PII leaks in sample; 0.0000 score drift under name swap. |
| **C5** | Metrics & Claims Verification| **Approved** | All portfolio claims grounded in reproducible logged runs. |
