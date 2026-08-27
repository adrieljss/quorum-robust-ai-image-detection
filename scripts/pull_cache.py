"""Pull the shared embedding cache + manifests. ~660MB, no images.

Set $QUORUM_CACHE_REPO once, after Step 8 creates it.
"""
import os

from huggingface_hub import snapshot_download

REPO = os.environ.get("QUORUM_CACHE_REPO", "YOUR_ORG/quorum-cache")
if REPO.startswith("YOUR_ORG"):
    raise SystemExit("set $QUORUM_CACHE_REPO first")

# embeddings/ and manifests/ both live at the repo root -- vectors without the
# rows CSV are unlabelled floats, so they always travel together.
snapshot_download(REPO, repo_type="dataset", local_dir="data")
print("pulled -> data/cache/embeddings/, data/manifests/")
