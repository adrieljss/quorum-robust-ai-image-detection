# Quorum — Robust Detection of AI-Generated Images

TikTok TechJam 2026, topic 5. Decides whether an image is camera-captured or
machine-generated, and holds up after it has been compressed, cropped, resized,
filtered and reposted.

No single model decides: several weak, independently calibrated signals are
fused by a learned meta-classifier, and the system reports its own reliability
alongside its verdict.

Design docs: [`docs/SPEC.md`](docs/SPEC.md) (architecture, constraints),
[`docs/DATA_LAYOUT.md`](docs/DATA_LAYOUT.md) (data),
[`docs/PIPELINE.md`](docs/PIPELINE.md) (model input contracts),
[`docs/RUNBOOK.md`](docs/RUNBOOK.md) (how the data pass is run).

## Required output

```bash
python predict.py --input-dir path/to/images --output preds.json
```

```json
[{"image_path": "path/to/images/001.jpg", "pred": 0.87}]
```

`pred` is P(AI-generated), calibrated, in [0,1]. Currently a stub returning
random scores — the contract is real, the model is not yet wired in.

## Getting started (team)

```bash
git clone <repo> && cd robust-ai-image-detection
python -m venv .venv && .venv\Scripts\activate      # source .venv/bin/activate on mac/linux
pip install torch --index-url https://download.pytorch.org/whl/cu130   # NVIDIA GPU: do this FIRST
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"             # must print True
hf auth login
python scripts/pull_cache.py          # ~1.2GB of embeddings, no images
```

```python
from quorum.embed import load_source
X, rows = load_source("sid_train")    # X (N,768) aligns 1:1 with rows
```

### Rules

- `train` trains. `calib` fits calibrators only. `test_*` is never trained on.
- Every `train_*.py` takes a required `--manifest` argument. No default.
- Do not re-run the embedding pass. If you think you need to, ask — it changes
  everyone's numbers.
- `data/raw/organizer_val/` is the competition validation set. Never train on it.
- Train rows carry 4 variants per image (clean + 3 sampled); eval rows carry all 15.

## Self-checks

```bash
python -m quorum.degrade        # transform grid: 14 settings, 15 variants, seeded
python -m quorum.embed          # cache format: ids, shards, shard-aware loader
```

## Status

| | |
|---|---|
| Done | `degrade.py`, `embed.py`, `stream_embed.py`, `embed_dir.py`, `build_manifest.py`, `predict.py` (stub) |
| Next | run the embedding pass (RUNBOOK steps 2–7), then `detectors/general.py` |
