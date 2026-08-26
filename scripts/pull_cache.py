"""Pull the shared embedding cache. ~1.2GB, no images.

Set REPO once, after Step 8 creates it.
"""
import os

from huggingface_hub import snapshot_download

REPO = os.environ.get("QUORUM_CACHE_REPO", "YOUR_ORG/quorum-cache")
if REPO.startswith("YOUR_ORG"):
    raise SystemExit("set REPO in scripts/pull_cache.py (or $QUORUM_CACHE_REPO) first")

snapshot_download(REPO, repo_type="dataset", local_dir="data/cache/embeddings")
