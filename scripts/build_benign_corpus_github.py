"""
Pull real README files from GitHub as benign 'document/web-page' tool-output
samples. Uses the unauthenticated GitHub REST API (60 req/hr limit - fine for
a demo batch; add a GITHUB_TOKEN env var and pass as Authorization header for
bulk collection to raise the limit to 5000/hr).
"""
import json, hashlib, time, urllib.request, urllib.error
from pathlib import Path

OUT = Path("benign_corpus/raw/github_readmes.jsonl")

# a small curated seed list across varied topics/tool-shapes; expand this list
# (or swap for the GitHub search API) to reach corpus-volume targets
REPOS = [
    "torvalds/linux", "pallets/flask", "psf/requests", "django/django",
    "pandas-dev/pandas", "numpy/numpy", "fastapi/fastapi", "scikit-learn/scikit-learn",
    "huggingface/transformers", "langchain-ai/langchain",
]

def make_id(*parts):
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]

def fetch_readme(repo):
    url = f"https://api.github.com/repos/{repo}/readme"
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github.raw+json",
        "User-Agent": "mcp-ipi-guard-corpus-builder",
    })
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.read().decode("utf-8", errors="ignore")

def main():
    records = []
    for repo in REPOS:
        try:
            text = fetch_readme(repo)
        except urllib.error.HTTPError as e:
            print(f"  skip {repo}: HTTP {e.code}")
            continue
        text = text[:4000]  # cap length; guard model will truncate anyway
        records.append({
            "id": make_id("github_readme", repo),
            "text": text,
            "label": 0,
            "attack_category": "none",
            "tool_type": "file_system_read",
            "source": f"github_readme:{repo}",
        })
        time.sleep(0.5)  # be polite to the unauthenticated rate limit

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(records)} benign README records to {OUT}")

if __name__ == "__main__":
    main()
