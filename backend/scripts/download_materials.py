import os
import urllib.request
import urllib.parse
import sys

DOWNLOAD_TARGETS = [
    {
        "name": "ESCO Skills Taxonomy (v1.2.1)",
        "url": "https://raw.githubusercontent.com/Prerit-112/Industry-Skill-Taxonomy/main/" + urllib.parse.quote("data/ESCO dataset - v1.2.1 - classification - en - csv/skills_en.csv"),
        "dest": "data/raw/esco_skills_en.csv",
        "expected_min_mb": 5.0
    },
    {
        "name": "O*NET Technology Skills Database",
        "url": "https://www.onetcenter.org/dl_files/database/db_29_0_text/Technology%20Skills.txt",
        "dest": "data/raw/onet_technology_skills.txt",
        "expected_min_mb": 1.5
    },
    {
        "name": "HuggingFace Resume-Job Fit (Train Pairs)",
        "url": "https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit/resolve/main/train.csv",
        "dest": "data/raw/hf_fit_train.csv",
        "expected_min_mb": 30.0
    },
    {
        "name": "HuggingFace Resume-Job Fit (Test Pairs)",
        "url": "https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit/resolve/main/test.csv",
        "dest": "data/raw/hf_fit_test.csv",
        "expected_min_mb": 10.0
    }
]

def download_file(target):
    name = target["name"]
    url = target["url"]
    dest = target["dest"]
    min_mb = target["expected_min_mb"]
    
    if os.path.exists(dest) and os.path.getsize(dest) > min_mb * 1024 * 1024:
        print(f"[ALREADY EXISTS] {name} ({os.path.getsize(dest) / (1024*1024):.2f} MB) at {dest}")
        return True

    print(f"[DOWNLOADING] {name} from {url[:70]}...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as out_f:
            total_size = int(resp.headers.get("content-length", 0))
            downloaded = 0
            chunk_size = 1024 * 512
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out_f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    sys.stdout.write(f"\r  -> {pct:.1f}% ({downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB)")
                    sys.stdout.flush()
        print(f"\n[COMPLETED] {name} saved to {dest} ({os.path.getsize(dest) / (1024*1024):.2f} MB)")
        return True
    except Exception as e:
        print(f"\n[FAILED] {name}: {e}")
        return False

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    all_success = True
    for item in DOWNLOAD_TARGETS:
        ok = download_file(item)
        if not ok:
            all_success = False
    
    print("\nDownload summary completed. Success:", all_success)
