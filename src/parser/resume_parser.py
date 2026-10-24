"""
Unified Resume Document Parser.
Extracts clean plain text, structured sections, and skills from PDF, DOCX, and TXT files.
"""

import os
import re
from typing import Dict, Any, List
import pdfplumber
import docx

from src.nlp.skill_extractor import extract_skills_from_text
from src.schema import ParsedResume

def extract_text_from_pdf(file_path: str) -> str:
    """Extracts text from PDF using pdfplumber."""
    text_parts = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text_parts.append(extracted)
    return "\n".join(text_parts)

def extract_text_from_docx(file_path: str) -> str:
    """Extracts text from DOCX using python-docx."""
    doc = docx.Document(file_path)
    text_parts = [p.text for p in doc.paragraphs if p.text]
    return "\n".join(text_parts)

def clean_extracted_text(text: str) -> str:
    """Sanitizes text and normalizes spacing."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def parse_resume_file(file_path: str) -> ParsedResume:
    """
    Parses a resume file (PDF, DOCX, or TXT) into a structured ParsedResume object.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Resume file not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    raw_content = ""

    if ext == ".pdf":
        raw_content = extract_text_from_pdf(file_path)
    elif ext in [".docx", ".doc"]:
        raw_content = extract_text_from_docx(file_path)
    else:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_content = f.read()

    cleaned_text = clean_extracted_text(raw_content)
    skills = extract_skills_from_text(cleaned_text)
    
    file_id = os.path.splitext(os.path.basename(file_path))[0]

    return ParsedResume(
        resume_id=file_id,
        category="General",
        raw_text=cleaned_text,
        skills=skills
    )

def parse_resume_text(text: str, resume_id: str = "custom") -> ParsedResume:
    """Parses raw text directly."""
    cleaned = clean_extracted_text(text)
    skills = extract_skills_from_text(cleaned)
    return ParsedResume(
        resume_id=resume_id,
        category="Pasted",
        raw_text=cleaned,
        skills=skills
    )
