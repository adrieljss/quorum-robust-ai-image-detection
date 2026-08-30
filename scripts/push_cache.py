"""Push the shared cache to the team HF dataset repo. Counterpart to pull_cache.py.

    $env:QUORUM_CACHE_REPO = "your-org/quorum-cache"
    python scripts/push_cache.py

Vectors and manifests only -- never raw images. The raw sets are 60GB+ and every
teammate can re-derive them; the whole point of the cache is that they do not
have to.
"""
import os
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
REPO = os.environ.get("QUORUM_CACHE_REPO", "")
if not REPO:
    raise SystemExit("set $QUORUM_CACHE_REPO first (e.g. your-org/quorum-cache)")

# The integrity gate. main.csv only exists if build_manifest.py's assertions
# passed, so a cache without it has never been proven leak-free -- and a leak
# does not look like a bug, it looks like an unusually good number.
if not (ROOT / "data" / "manifests" / "main.csv").exists():
    raise SystemExit("no main.csv -- run scripts/build_manifest.py first; "
                     "pushing an unverified cache is how contamination spreads")

api = HfApi()
api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
api.upload_folder(
    folder_path=str(ROOT / "data"),
    repo_id=REPO,
    repo_type="dataset",
    allow_patterns=["cache/embeddings/**", "manifests/**", "models/**"],
    ignore_patterns=["**/rows_*smoke*", "**/*smoke*.npy"],
    commit_message="quorum cache update",
)
size = sum(f.stat().st_size for p in ("cache/embeddings", "manifests", "models")
           for f in (ROOT / "data" / p).rglob("*") if f.is_file())
print(f"pushed {size / 2**20:.0f} MB -> {REPO}")
print(f"teammates: $env:QUORUM_CACHE_REPO = \"{REPO}\"; python scripts/pull_cache.py")
