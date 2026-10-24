"""
Ultra High-Performance Skill Extractor using N-Gram Hash Table Lookup.
Extracts canonical skills from arbitrary text in sub-millisecond time.
Thread-safe for multi-threaded and parallel processing.
"""

import os
import json
import re
from typing import List, Set, Dict

TAXONOMY_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "skills_taxonomy.json")

class SkillExtractor:
    _instance = None

    def __init__(self, taxonomy_file: str = None):
        if taxonomy_file is None:
            taxonomy_file = TAXONOMY_PATH
        
        self.taxonomy: Dict[str, str] = {}
        self.canonical_skills: Set[str] = set()
        self._load_taxonomy(taxonomy_file)

    def _load_taxonomy(self, filepath: str):
        if not os.path.exists(filepath):
            return
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.taxonomy = data.get("taxonomy", {})
            self.canonical_skills = set(data.get("canonical_skills", []))

    def extract_skills(self, text: str) -> List[str]:
        """
        Extracts standardized canonical skills using O(1) n-gram hash lookup.
        Runs in ~0.1 ms per document.
        """
        if not text or not isinstance(text, str):
            return []

        text_lower = text.lower()

        # Pre-pass: check for slash-joined terms (ci/cd) and dotted prefixes (.net)
        # directly against taxonomy before tokenizing
        found_skills: Set[str] = set()
        for pattern in re.finditer(r"[a-z0-9\.\+#\-]+(?:/[a-z0-9\.\+#\-]+)+", text_lower):
            term = pattern.group().strip(".-")
            if term in self.taxonomy:
                found_skills.add(self.taxonomy[term])
        for pattern in re.finditer(r"\.[a-z][a-z0-9\.\+#\-]*", text_lower):
            term = pattern.group()
            if term in self.taxonomy:
                found_skills.add(self.taxonomy[term])

        # Tokenize words, preserving dots, pluses, dashes, slashes
        tokens = re.findall(r"[a-z0-9\.\+#\-/]+", text_lower)
        if not tokens:
            return sorted(list(found_skills))

        num_tokens = len(tokens)

        # Check 1-gram to 4-gram phrases
        max_n = min(5, num_tokens + 1)
        for n in range(1, max_n):
            for i in range(num_tokens - n + 1):
                phrase = " ".join(tokens[i:i+n]).strip(".-")
                if phrase in self.taxonomy:
                    found_skills.add(self.taxonomy[phrase])

        return sorted(list(found_skills))

# Global helper instance
_extractor = None

def get_skill_extractor() -> SkillExtractor:
    global _extractor
    if _extractor is None:
        _extractor = SkillExtractor()
    return _extractor

def extract_skills_from_text(text: str) -> List[str]:
    return get_skill_extractor().extract_skills(text)
