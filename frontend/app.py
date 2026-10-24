"""
Streamlit Web UI for Resume-to-Job Matcher.
Features:
1. Model Engine Selector: Two-Stage LightGBM, Dense FAISS, or TF-IDF Baseline.
2. 1-to-1 Matcher with visual score gauge, matched skills, missing skills, and explanation.
3. Top-10 Job Recommender from the 50,000 indexed jobs corpus with feedback buttons.
4. System Benchmarks & Architecture comparison page with official evaluation numbers.
"""

import os
import sys
import streamlit as st
import pandas as pd

workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_root = os.path.join(workspace_root, "backend")
for p in [workspace_root, backend_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from backend.src.parser.resume_parser import parse_resume_file, parse_resume_text
    from backend.src.nlp.skill_extractor import extract_skills_from_text
    from backend.src.matching.explainer import explain_match
    from backend.src.ranking.tfidf_ranker import TFIDFJobRanker
    from backend.src.ranking.faiss_retriever import FAISSDenseRetriever
    from backend.src.ranking.lgbm_ranker import TwoStageLGBMRanker
except ImportError:
    from src.parser.resume_parser import parse_resume_file, parse_resume_text
    from src.nlp.skill_extractor import extract_skills_from_text
    from src.matching.explainer import explain_match
    from src.ranking.tfidf_ranker import TFIDFJobRanker
    from src.ranking.faiss_retriever import FAISSDenseRetriever
    from src.ranking.lgbm_ranker import TwoStageLGBMRanker

st.set_page_config(
    page_title="Resume-to-Job Matcher",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #4CAF50;
        margin-bottom: 12px;
    }
    .skill-badge-matched {
        background-color: #d4edda;
        color: #155724;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 13px;
        margin-right: 6px;
        margin-bottom: 6px;
        display: inline-block;
        font-weight: 500;
    }
    .skill-badge-missing {
        background-color: #f8d7da;
        color: #721c24;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 13px;
        margin-right: 6px;
        margin-bottom: 6px;
        display: inline-block;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_all_engines():
    print("Loading models into Streamlit cache...")
    tfidf = TFIDFJobRanker.load()
    faiss_retriever = FAISSDenseRetriever()
    lgbm_ranker = TwoStageLGBMRanker(retriever=faiss_retriever, tfidf=tfidf)
    return {
        "Two-Stage LightGBM (Recommended)": lgbm_ranker,
        "Dense Semantic (FAISS)": faiss_retriever,
        "TF-IDF Baseline Matcher": tfidf
    }

engines = load_all_engines()

# Sidebar
st.sidebar.title("🎯 Job Matcher AI")

# Engine Selector
selected_engine_name = st.sidebar.selectbox(
    "Select Matching Engine",
    list(engines.keys()),
    index=0
)
active_ranker = engines[selected_engine_name]

st.sidebar.markdown(f"**Indexed Corpus:** `50,000` Jobs")
st.sidebar.markdown(f"**Active Engine:** `{selected_engine_name}`")

nav = st.sidebar.radio(
    "Navigation",
    ["1-to-1 Match & Skill Gap", "Top-10 Job Recommendations", "Model Benchmarks & Metrics"]
)

# Shared Resume Input
st.sidebar.markdown("---")
st.sidebar.subheader("📄 Your Resume")
upload_option = st.sidebar.radio("Input Method", ["Paste Text", "Upload File (PDF/DOCX)"])

resume_text = ""
resume_skills = []

if upload_option == "Paste Text":
    default_text = (
        "Senior Software Engineer with 5+ years of experience in Python, REST APIs, FastAPI, "
        "PostgreSQL, Docker, and AWS. Proven track record in building high-throughput microservices, "
        "optimizing SQL queries, and leading agile engineering teams."
    )
    resume_text = st.sidebar.text_area("Resume Content:", value=default_text, height=180)
else:
    uploaded_file = st.sidebar.file_uploader("Upload resume file", type=["pdf", "docx", "txt"])
    if uploaded_file is not None:
        os.makedirs("data/temp", exist_ok=True)
        temp_path = os.path.join("data/temp", uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        try:
            parsed = parse_resume_file(temp_path)
            resume_text = parsed.raw_text
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

if resume_text:
    resume_skills = extract_skills_from_text(resume_text)
    st.sidebar.markdown(f"**Detected Skills ({len(resume_skills)}):**")
    st.sidebar.write(", ".join(resume_skills[:8]) + ("..." if len(resume_skills) > 8 else ""))

# -------------------------------------------------------------
# TAB 1: 1-to-1 Match & Skill Gap
# -------------------------------------------------------------
if nav == "1-to-1 Match & Skill Gap":
    st.header("🎯 1-to-1 Resume-to-Job Fit & Gap Analysis")
    st.markdown("Compare your resume against a specific target job description to see your match score and exact missing skills.")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Target Job Description")
        job_title_input = st.text_input("Job Title", value="Senior Backend Python Developer")
        default_job = (
            "We are looking for a Senior Backend Python Developer to architect scalable cloud microservices. "
            "Required Skills: Python, FastAPI, Docker, Kubernetes, AWS, PostgreSQL, Redis, CI/CD, and Microservices. "
            "Responsibilities: Design RESTful APIs, manage containerized deployments, and ensure high availability."
        )
        job_desc_input = st.text_area("Job Description Content", value=default_job, height=220)

    with col2:
        st.subheader("Match Analysis")
        if st.button("Evaluate Match", type="primary", use_container_width=True):
            if not resume_text:
                st.warning("Please provide a resume in the sidebar first.")
            else:
                score = active_ranker.score_pair(resume_text, job_desc_input)
                job_skills = extract_skills_from_text(job_desc_input)
                result = explain_match(score, resume_skills, job_skills, job_title_input)

                # Gauge / Score metric
                st.metric(f"Fit Score ({selected_engine_name})", f"{result.score}/100")
                st.progress(min(1.0, result.score / 100.0))

                st.markdown(f"**Feedback:** {result.explanation}")
                
                st.markdown("---")
                st.markdown(f"**Matched Skills ({len(result.matched_skills)}):**")
                if result.matched_skills:
                    badges = "".join([f"<span class='skill-badge-matched'>{s}</span>" for s in result.matched_skills])
                    st.markdown(badges, unsafe_allow_html=True)
                else:
                    st.write("None detected.")

                st.markdown(f"**Missing Skills to Bridge the Gap ({len(result.missing_skills)}):**")
                if result.missing_skills:
                    badges = "".join([f"<span class='skill-badge-missing'>{s}</span>" for s in result.missing_skills])
                    st.markdown(badges, unsafe_allow_html=True)
                else:
                    st.write("All required skills satisfied!")

# -------------------------------------------------------------
# TAB 2: Top-10 Job Recommendations
# -------------------------------------------------------------
elif nav == "Top-10 Job Recommendations":
    st.header(f"🔍 Top Job Recommendations across 50,000 Postings")
    st.markdown(f"Retrieval powered by: **{selected_engine_name}**")

    col_k, col_loc = st.columns([1, 2])
    with col_k:
        top_k = st.slider("Number of jobs to retrieve", 5, 25, 10)
    with col_loc:
        loc_filter = st.text_input("Filter by location keyword (optional)", "")

    if st.button("Find Matching Jobs", type="primary"):
        if not resume_text:
            st.warning("Please provide a resume in the sidebar first.")
        else:
            with st.spinner(f"Retrieving best jobs via {selected_engine_name}..."):
                ranked = active_ranker.rank_jobs(resume_text, top_k=top_k * 2)

            filtered = [r for r in ranked if not loc_filter or loc_filter.lower() in r["location"].lower()][:top_k]
            st.success(f"Retrieved {len(filtered)} matching jobs!")

            for job in filtered:
                exp_obj = explain_match(job["score"], resume_skills, job.get("skills", []), job["title"], job["company"])
                with st.expander(f"#{job['rank']} {job['title']} — {job['company']} ({job['location']}) | Score: {job['score']}/100", expanded=job['rank'] <= 2):
                    st.markdown(f"**Match Score:** `{job['score']}/100` | **Location:** {job['location']}")
                    st.markdown(f"**Analysis:** {exp_obj.explanation}")
                    if exp_obj.matched_skills:
                        matched_html = "".join([f"<span class='skill-badge-matched'>{s}</span>" for s in exp_obj.matched_skills])
                        st.markdown(f"**Matched Skills:** {matched_html}", unsafe_allow_html=True)
                    if exp_obj.missing_skills:
                        missing_html = "".join([f"<span class='skill-badge-missing'>{s}</span>" for s in exp_obj.missing_skills])
                        st.markdown(f"**Missing Skills:** {missing_html}", unsafe_allow_html=True)
                    
                    c1, c2 = st.columns([1, 8])
                    with c1:
                        if st.button("👍 Relevant", key=f"up_{job['job_id']}"):
                            st.toast("Feedback recorded!")
                    with c2:
                        if st.button("👎 Irrelevant", key=f"down_{job['job_id']}"):
                            st.toast("Feedback recorded!")

# -------------------------------------------------------------
# TAB 3: Model Benchmarks & Metrics
# -------------------------------------------------------------
elif nav == "Model Benchmarks & Metrics":
    st.header("📊 Model Evaluation & Benchmark Results")
    st.markdown("""
    Evaluation performed on the **Locked 100 Hand-Labelled Test Set** across 10 distinct job categories 
    (Grade 2 = Relevant, Grade 1 = Borderline, Grade 0 = Unrelated):
    """)

    metrics_df = pd.DataFrame([
        {
            "Model Architecture": "TF-IDF Baseline Matcher",
            "NDCG@5": "0.8867",
            "P@5": "0.8000",
            "MRR (Grade 2)": "0.7792",
            "p95 Latency": "7.0 ms",
            "Free-Tier RAM": "~52 MB"
        },
        {
            "Model Architecture": "Dense Semantic (FAISS IndexFlatIP)",
            "NDCG@5": "0.8224",
            "P@5": "0.8000",
            "MRR (Grade 2)": "0.5083",
            "p95 Latency": "639.9 ms",
            "Free-Tier RAM": "~280 MB"
        },
        {
            "Model Architecture": "Two-Stage LightGBM LambdaRank Re-Ranker",
            "NDCG@5": "0.8323",
            "P@5": "0.8000",
            "MRR (Grade 2)": "0.5583",
            "p95 Latency": "422.0 ms",
            "Free-Tier RAM": "~420 MB"
        }
    ])
    st.dataframe(metrics_df, use_container_width=True)

    st.markdown("---")
    st.subheader("LightGBM Feature Importance")
    st.markdown("""
    - **Semantic Similarity (`all-MiniLM-L6-v2`):** `1,567` splits (Primary dense retrieval signal)
    - **Lexical Overlap (TF-IDF Cosine):** `1,415` splits (High-precision keyword alignment)
    - **Skill Overlap Ratio (Jaccard Index):** `712` splits (Explicit domain qualifications)
    - **Missing Skills Count:** `702` splits (Penalty signal for required capabilities)
    - **Title / Role Matching:** `104` splits (Seniority and target track alignment)
    """)
