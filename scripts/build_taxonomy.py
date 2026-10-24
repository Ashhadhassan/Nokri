"""
Builds a unified, standardized skills taxonomy from ESCO and O*NET datasets.
Outputs data/processed/skills_taxonomy.json.
"""

import os
import json
import re
import pandas as pd

TECH_ALIASES = {
    "js": "JavaScript",
    "ts": "TypeScript",
    "py": "Python",
    "python3": "Python",
    "k8s": "Kubernetes",
    "kube": "Kubernetes",
    "aws": "Amazon Web Services",
    "amazon web services": "Amazon Web Services",
    "gcp": "Google Cloud Platform",
    "google cloud": "Google Cloud Platform",
    "azure": "Microsoft Azure",
    "ms azure": "Microsoft Azure",
    "ml": "Machine Learning",
    "machine learning": "Machine Learning",
    "dl": "Deep Learning",
    "deep learning": "Deep Learning",
    "nlp": "Natural Language Processing",
    "natural language processing": "Natural Language Processing",
    "cv": "Computer Vision",
    "computer vision": "Computer Vision",
    "llm": "Large Language Models",
    "llms": "Large Language Models",
    "sql": "SQL",
    "nosql": "NoSQL",
    "ci/cd": "CI/CD",
    "cicd": "CI/CD",
    "devops": "DevOps",
    "qa": "Quality Assurance",
    "react": "React",
    "reactjs": "React",
    "react.js": "React",
    "node": "Node.js",
    "nodejs": "Node.js",
    "node.js": "Node.js",
    "vue": "Vue.js",
    "vuejs": "Vue.js",
    "angular": "Angular",
    "angularjs": "Angular",
    "docker": "Docker",
    "git": "Git",
    "github": "GitHub",
    "gitlab": "GitLab",
    "linux": "Linux",
    "html": "HTML",
    "css": "CSS",
    "rest": "REST APIs",
    "restful": "REST APIs",
    "rest api": "REST APIs",
    "graphql": "GraphQL",
    "fastapi": "FastAPI",
    "flask": "Flask",
    "django": "Django",
    "spring": "Spring Boot",
    "spring boot": "Spring Boot",
    "tensorflow": "TensorFlow",
    "pytorch": "PyTorch",
    "pandas": "Pandas",
    "numpy": "NumPy",
    "scikit-learn": "Scikit-Learn",
    "sklearn": "Scikit-Learn",
    "spark": "Apache Spark",
    "pyspark": "PySpark",
    "kafka": "Apache Kafka",
    "postgres": "PostgreSQL",
    "postgresql": "PostgreSQL",
    "mysql": "MySQL",
    "mongodb": "MongoDB",
    "redis": "Redis",
    ".net": ".NET Framework",
    "dotnet": ".NET Framework",
    ".net core": ".NET Framework",
    "asp.net": "ASP.NET",
    "vb.net": "VB.NET"
}

def clean_term(term: str) -> str:
    """Normalizes term for dictionary key."""
    if not isinstance(term, str):
        return ""
    t = term.lower().strip()
    t = re.sub(r"[\(\)\[\]]", "", t)
    t = re.sub(r"\s+", " ", t)
    return t

def build_taxonomy(
    esco_csv: str = "data/raw/esco_skills_en.csv",
    onet_txt: str = "data/raw/onet_technology_skills.txt",
    output_json: str = "data/processed/skills_taxonomy.json"
):
    print("Building unified skills taxonomy...")
    taxonomy = {}  # normalized_alias -> canonical_name
    canonical_skills = set()

    # 1. Add common hand-curated tech aliases
    for alias, canonical in TECH_ALIASES.items():
        norm_a = clean_term(alias)
        if norm_a:
            taxonomy[norm_a] = canonical
            canonical_skills.add(canonical)

    # 2. Parse O*NET Technology Skills
    if os.path.exists(onet_txt):
        print(f"Reading O*NET technologies from {onet_txt}...")
        df_onet = pd.read_csv(onet_txt, sep="\t", usecols=["Example", "Commodity Title", "Hot Technology"])
        for _, row in df_onet.iterrows():
            example = str(row["Example"]).strip()
            if not example or len(example) < 2:
                continue
            canonical = example
            norm = clean_term(example)
            if norm and norm not in taxonomy:
                taxonomy[norm] = canonical
                canonical_skills.add(canonical)

    # 3. Parse ESCO Skills (focus on skill/competence and digital/knowledge)
    if os.path.exists(esco_csv):
        print(f"Reading ESCO skills from {esco_csv}...")
        df_esco = pd.read_csv(esco_csv, usecols=["preferredLabel", "altLabels", "skillType"])
        for _, row in df_esco.iterrows():
            pref = str(row["preferredLabel"]).strip()
            if not pref or len(pref) < 3:
                continue
            canonical = pref.title()
            norm_pref = clean_term(pref)
            if norm_pref and norm_pref not in taxonomy:
                taxonomy[norm_pref] = canonical
                canonical_skills.add(canonical)

            # Process alt labels (synonyms)
            alt = str(row["altLabels"])
            if alt and alt != "nan":
                for alt_term in alt.split("\n"):
                    norm_alt = clean_term(alt_term)
                    if norm_alt and len(norm_alt) >= 3 and norm_alt not in taxonomy:
                        taxonomy[norm_alt] = canonical

    # Save output
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    out_data = {
        "metadata": {
            "total_terms": len(taxonomy),
            "canonical_skills_count": len(canonical_skills),
            "version": "1.0.0"
        },
        "taxonomy": taxonomy,
        "canonical_skills": sorted(list(canonical_skills))
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)

    print(f"Taxonomy built successfully: {len(taxonomy)} mapped terms, {len(canonical_skills)} canonical skills.")
    print(f"Saved to: {output_json}")

if __name__ == "__main__":
    build_taxonomy()
