"""Privacy, Fairness, and Claims Audit."""
import re, sys, os
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

print("=== C4: PRIVACY & FAIRNESS ===")

# PII regex coverage
pii_patterns = {
    'email': r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+',
    'phone_us': r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
    'url': r'https?://\S+|www\.\S+'
}
test_cases = [
    ('john.doe@gmail.com', 'email'),
    ('+1 (555) 123-4567', 'phone_us'),
    ('https://linkedin.com/in/john', 'url'),
    ('555.123.4567', 'phone_us'),
]
for text, ptype in test_cases:
    matched = bool(re.search(pii_patterns[ptype], text))
    print(f'  PII [{ptype}] "{text[:30]}": {"CAUGHT" if matched else "MISSED"}')

# Check resume data for residual PII
res = pd.read_parquet('data/processed/resumes_clean.parquet')
sample = res['raw_text'].sample(50, random_state=42)
email_leaks = 0
for t in sample:
    emails = re.findall(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.\w{2,}', t)
    for e in emails:
        if 'REDACTED' not in e:
            email_leaks += 1
print(f'  Email leaks in 50-sample: {email_leaks}')

# Fairness: name/gender swap score invariance
from src.ranking.tfidf_ranker import TFIDFJobRanker
ranker = TFIDFJobRanker.load()
base = 'Software engineer with 5 years Python experience building REST APIs and microservices.'
job = 'Looking for senior Python engineer to build cloud microservices.'
s1 = ranker.score_pair('John Smith. ' + base, job)
s2 = ranker.score_pair('Jane Smith. ' + base, job)
s3 = ranker.score_pair('Muhammad Ali. ' + base, job)
s4 = ranker.score_pair(base, job)
print(f'  Score (John): {s1}, (Jane): {s2}, (Muhammad): {s3}, (no name): {s4}')
drift = max(s1,s2,s3,s4) - min(s1,s2,s3,s4)
print(f'  Max score drift from name swap: {drift:.4f} (threshold: <2.0)')

# C5: Claims check
print()
print("=== C5: CLAIMS AUDIT ===")
with open('README.md', encoding='utf-8') as f:
    readme = f.read()
print('  Claims sub-second latency:', '< 2,000 ms' in readme or 'sub-second' in readme.lower())
print('  Claims 50k jobs:', '50,000' in readme)
print('  Claims 108k taxonomy:', '108,000' in readme)
print('  Reports actual benchmark numbers:', '0.8867' in readme and '0.8000' in readme)
print('  Mentions limitations/advisory:', 'advisory' in readme.lower())
