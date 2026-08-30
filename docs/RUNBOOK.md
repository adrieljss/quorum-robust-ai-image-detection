# Quorum — Data & Embedding Runbook

**Audience:** one person (Data/Eval + Detector owner), empty repo, single machine.

**Target hardware:** Python 3.13.1, RTX 4060 Laptop 8GB. Tuned for this; notes
where a different box changes things.

**Outcome:** a repo the other four members clone and start modelling in, without
downloading a single image.

**Time:** ~4 hours, of which ~2 is an unattended GPU job.

Companion docs: `SPEC.md`, `DATA_LAYOUT.md`, `PIPELINE.md`.

---

## Corrections in this revision

Seven issues found in review. Listed here so nobody reintroduces them.

| # | Issue | Fix |
|---|---|---|
| 1 | Variant count wrong (14 settings, not 15) | `N_SETTINGS = 14`, `N_VARIANTS = 15` — asserted in code |
| 2 | Streamed images skipped JPEG-q95 normalisation | `normalise()` before hashing in `stream_embed.py` (§6) |
| 3 | Accumulating 840k vectors in a list → ~4GB peak, OOM | Shard flush every 20k (§6) |
| 4 | Noise/jitter unseeded → eval grid not reproducible | RNG seeded from `image_id` (§3) |
| 5 | MediaPipe has no Python 3.13 wheels | Use OpenCV YuNet instead (§10) |
| 6 | `--all-variants` on train wastes half the runtime | Sample 3 on train, full grid on eval (§6) |
| 7 | fp32 CLIP is tight in 8GB VRAM | fp16 autocast (§4) |

---

## Hardware notes — RTX 4060 Laptop 8GB

| Concern | Situation |
|---|---|
| VRAM | ViT-L/14 fp32 weights = 1.22GB; fp16 = 0.61GB. **Use fp16.** Batch 15 fits comfortably. |
| Real bottleneck | **CPU, not GPU.** PIL transforms + JPEG round-trips dominate. Hence fix #6. |
| Python 3.13 | torch ≥2.6, open_clip, datasets, opencv all fine. **mediapipe is not** — see §10. |
| Thermals | Laptop GPUs throttle. Run on mains, elevated, and expect ~20% slower than desktop equivalents. |

Embedding budget after fix #6:

```
train        40k images ×  4 variants = 160k
calib         6k images × 15 variants =  90k
test_ood     10k images × 15 variants = 150k
                                  total 400k ≈ 1.2GB fp32
```

vs 840k (2.6GB) if you ran the full grid everywhere. Same deliverable, half the
time.

---

## Step 0 — Prerequisites

```bash
python --version     # 3.13.1
nvidia-smi           # confirm RTX 4060, 8GB
df -h .              # need ~20GB free
```

HF account with a write token: `https://huggingface.co/settings/tokens`

---

## Step 1 — Scaffold

```bash
mkdir quorum && cd quorum && git init
mkdir -p quorum/detectors scripts data/{manifests,cache/{embeddings,face_crops,scores},raw,models} demo docs
touch quorum/__init__.py quorum/detectors/__init__.py
```

```
quorum/
├── README.md                    # required deliverable
├── requirements.txt
├── .gitignore
├── predict.py                   # required deliverable — stub now
│
├── quorum/
│   ├── degrade.py               # STEP 3 — official transform grid
│   ├── embed.py                 # STEP 4 — frozen CLIP, fp16
│   ├── provenance.py            # later, low priority
│   ├── fusion.py                # teammate
│   ├── calibrate.py             # teammate
│   ├── pipeline.py              # teammate
│   └── detectors/
│       ├── general.py           # teammate
│       ├── face.py              # teammate
│       ├── text.py              # teammate
│       └── spectral.py          # teammate
│
├── scripts/
│   ├── check_sizes.py           # STEP 2
│   ├── inspect_dataset.py       # STEP 6a — do not skip
│   ├── stream_embed.py          # STEP 6 — the big one
│   ├── download_small.py        # STEP 5
│   ├── build_manifest.py        # STEP 7
│   ├── load_embeddings.py       # shard-aware loader
│   └── pull_cache.py            # teammates run this
│
├── data/
│   ├── manifests/               # COMMITTED
│   ├── cache/                   # gitignored
│   ├── raw/                     # gitignored
│   └── models/                  # COMMITTED, ~2MB
│
├── demo/
└── docs/{SPEC,DATA_LAYOUT,PIPELINE}.md
```

**`.gitignore`:**
```
data/raw/
data/cache/
__pycache__/
*.pyc
.venv/
.env
```

**`requirements.txt`:**
```
torch>=2.6
open_clip_torch
datasets>=2.14
huggingface_hub
numpy
pandas
pyarrow
pillow
opencv-python>=4.8
scikit-learn
matplotlib
tqdm
```

`opencv-python>=4.8` matters — that is where `FaceDetectorYN` lives. **No
mediapipe** (§10).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
hf auth login
git add -A && git commit -m "scaffold"
```

---

## Step 2 — Verify dataset sizes first

`scripts/check_sizes.py`:

```python
from huggingface_hub import HfApi

DATASETS = ["saberzl/SID_Set", "saberzl/So-Fake-OOD",
            "pujanpaudel/deepfake_face_classification"]

api = HfApi()
for name in DATASETS:
    try:
        info = api.dataset_info(name, files_metadata=True)
        gb = sum(f.size for f in info.siblings if f.size) / 1e9
        print(f"{name:50s} {gb:8.1f} GB  ({len(info.siblings)} files)")
    except Exception as e:
        print(f"{name:50s} ERROR: {e}")
```

**Decision rule:** >5GB → stream. <5GB → download. Published sizes lie; SID_Set
is 140GB and an earlier draft of these docs guessed 30.

---

## Step 3 — `degrade.py`

The official grid from the brief. Blocks Step 6 — variants are generated inline
while each image is in memory, so this must exist before any data is fetched.

**Fix #1** (count asserted) and **fix #4** (seeded RNG) are both here.

`quorum/degrade.py`:

```python
"""Official robustness transform grid from the problem statement.

14 settings + clean = 15 variants. Seeded per image so the eval grid is
reproducible run-to-run — it is a required deliverable.
"""
import io
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

TRANSFORMS = {
    "jpeg":   [90, 70, 50, 30],    # 4  quality
    "blur":   [0.5, 1.0, 2.0],     # 3  gaussian sigma
    "resize": [0.5, 0.25],         # 2  downscale then back up
    "noise":  [0.02, 0.05, 0.10],  # 3  gaussian sigma on [0,1]
    "jitter": [0.20],              # 1  brightness/contrast/sat +-20%
    "crop":   [0.80],              # 1  center crop fraction
}

N_SETTINGS = sum(len(v) for v in TRANSFORMS.values())   # 14
N_VARIANTS = N_SETTINGS + 1                             # 15, incl. clean
assert N_SETTINGS == 14, f"expected 14 settings, got {N_SETTINGS}"


def variant_name(kind: str, param) -> str:
    return f"{kind}{param}".replace(".", "")            # jpeg70, blur05, resize025


def seed_from_id(image_id: str) -> int:
    return int(image_id[:8], 16)


def apply(img: Image.Image, kind: str, param, rng=None) -> Image.Image:
    rng = rng or np.random.default_rng(0)

    if kind == "jpeg":
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=int(param))
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    if kind == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=float(param)))

    if kind == "resize":
        w, h = img.size
        small = img.resize((max(1, int(w*param)), max(1, int(h*param))), Image.BICUBIC)
        return small.resize((w, h), Image.BICUBIC)

    if kind == "noise":                                  # SEEDED
        a = np.asarray(img, dtype=np.float32) / 255.0
        a = a + rng.normal(0, float(param), a.shape)
        return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))

    if kind == "jitter":                                 # SEEDED
        p = float(param)
        for Enh in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
            img = Enh(img).enhance(1.0 + rng.uniform(-p, p))
        return img

    if kind == "crop":
        w, h = img.size
        cw, ch = int(w*param), int(h*param)
        l, t = (w-cw)//2, (h-ch)//2
        return img.crop((l, t, l+cw, t+ch)).resize((w, h), Image.BICUBIC)

    raise ValueError(f"unknown transform: {kind}")


def all_variants(img, image_id):
    """Full grid: clean + all 14 settings = 15. For EVAL splits."""
    rng = np.random.default_rng(seed_from_id(image_id))
    out = [("clean", img)]
    for kind, params in TRANSFORMS.items():
        for p in params:
            out.append((variant_name(kind, p), apply(img, kind, p, rng)))
    assert len(out) == N_VARIANTS
    return out


def sample_variants(img, image_id, k=3):
    """Clean + k randomly sampled settings. For the TRAIN split (fix #6).

    Deterministic given image_id, so a rerun reproduces the same augmentation.
    """
    rng = np.random.default_rng(seed_from_id(image_id))
    flat = [(k_, p) for k_, ps in TRANSFORMS.items() for p in ps]
    idx = rng.choice(len(flat), size=min(k, len(flat)), replace=False)
    out = [("clean", img)]
    for i in idx:
        kind, p = flat[i]
        out.append((variant_name(kind, p), apply(img, kind, p, rng)))
    return out
```

**Sanity check:**

```bash
python -c "
from PIL import Image
from quorum.degrade import all_variants, sample_variants, N_VARIANTS
img = Image.new('RGB',(512,512),'gray')
v = all_variants(img, 'a1b2c3d4e5f6a7b8')
print(len(v), N_VARIANTS)
print([n for n,_ in v])
print('sampled:', [n for n,_ in sample_variants(img, 'a1b2c3d4e5f6a7b8')])
"
# expect: 15 15
```

---

## Step 4 — `embed.py` (fp16, fix #7)

`quorum/embed.py`:

```python
"""Frozen CLIP embedding, fp16, versioned on-disk cache."""
from pathlib import Path
import numpy as np
import torch
import open_clip

BACKBONE = "ViT-L-14-quickgelu"   # NOT "ViT-L-14" -- see below
PRETRAINED = "openai"
CACHE_VERSION = "vitl14_v1"      # bump if preprocessing changes
CACHE = Path("data/cache/embeddings") / CACHE_VERSION
DIM = 768


class Embedder:
    def __init__(self, device=None, fp16=True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.fp16 = fp16 and self.device == "cuda"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            BACKBONE, pretrained=PRETRAINED)
        self.model = self.model.to(self.device).eval()
        if self.fp16:
            self.model = self.model.half()               # 1.22GB -> 0.61GB
        for p in self.model.parameters():
            p.requires_grad = False                      # never unfreeze

    @torch.no_grad()
    def embed_batch(self, pil_images) -> np.ndarray:
        x = torch.stack([self.preprocess(i) for i in pil_images]).to(self.device)
        if self.fp16:
            x = x.half()
        v = self.model.encode_image(x)
        v = v / v.norm(dim=-1, keepdim=True)             # L2 normalise — do NOT skip
        return v.float().cpu().numpy().astype(np.float32)
```

Note `.float()` before `.numpy()` — the cache stays fp32 so downstream sklearn
does not choke on half precision.

**Why `-quickgelu`:** the `openai` weights were trained with QuickGELU
activations. Plain `ViT-L-14` loads them into a standard-GELU model — no error,
just a `UserWarning` and subtly wrong embeddings. Measured cosine between the two
backbones on identical images: **0.88–0.91**. Caught after a 34GB streaming pass,
this costs a re-stream. `python -m quorum.embed --clip` must print no warning.

**Why L2-normalise:** without it the probe partly fits vector magnitude, which
tracks image statistics rather than provenance.

**Why never unfreeze:** LoRA fine-tuning a comparable frozen encoder dropped
in-the-wild accuracy from 95.9% to 63.5%. The 2B parameter cap makes it moot
anyway.

**Sanity check:**

```bash
python -c "
from PIL import Image
from quorum.embed import Embedder
e = Embedder(); v = e.embed_batch([Image.new('RGB',(512,512),'red')])
print(v.shape, v.dtype, abs(float((v**2).sum())-1) < 1e-3)
"
# expect: (1, 768) float32 True
```

---

## Step 5 — Small downloads

```bash
# Organizer validation — QUARANTINED
mkdir -p data/raw/organizer_val/coco_val2017 && cd data/raw/organizer_val/coco_val2017
curl -O http://images.cocodataset.org/zips/val2017.zip
unzip -q val2017.zip && rm val2017.zip && cd -
echo "COCO val2017 + WildFake DALL-E Advanced are the organizer's VALIDATION set.
Never train on anything in this folder." > data/raw/organizer_val/README_DO_NOT_TRAIN.txt

# Faces — face branch only
hf download pujanpaudel/deepfake_face_classification \
  --repo-type dataset --local-dir data/raw/faces

# Real diversity — Open Images, NOT COCO val2017
pip install fiftyone
python -c "
import fiftyone.zoo as foz
foz.load_zoo_dataset('open-images-v7', split='validation', max_samples=5000,
                     dataset_dir='data/raw/real_extra/openimages')
"
```

WildFake DALL·E Advanced comes from ModelScope (`hy2628982280/WildFake`) — use
the translate button, take **only** that subset, into
`data/raw/organizer_val/wildfake_dalle_adv/`.

---

## Step 6 — Streaming embedding pass

The main event. Contains fixes #2, #3 and #6.

`scripts/stream_embed.py`:

```python
"""Stream a HF dataset, embed degradation variants, never touch disk.

Fix #2: JPEG-q95 normalise BEFORE hashing, so streamed and downloaded images
        get comparable ids and file format cannot leak the label.
Fix #3: flush to shards instead of accumulating everything in RAM.
Fix #6: sample variants on train, full grid on eval.
"""
import argparse, hashlib, io
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from quorum.embed import Embedder, CACHE, DIM
from quorum.degrade import all_variants, sample_variants

FLUSH_EVERY = 20_000          # embeddings per shard (~60MB fp32)


def normalise(img: Image.Image) -> Image.Image:
    """JPEG q95 round-trip. MUST run before hashing (fix #2)."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=95)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def image_id(img: Image.Image) -> str:
    return hashlib.sha256(img.tobytes()).hexdigest()[:16]


class ShardWriter:
    """Flush every FLUSH_EVERY embeddings so peak RAM stays bounded (fix #3)."""

    def __init__(self, source: str):
        self.source, self.shard = source, 0
        self.vecs, self.rows = [], []
        CACHE.mkdir(parents=True, exist_ok=True)
        Path("data/manifests").mkdir(parents=True, exist_ok=True)

    def add(self, vec, row):
        self.vecs.append(vec)
        self.rows.append(row)
        if len(self.vecs) >= FLUSH_EVERY:
            self.flush()

    def flush(self):
        if not self.vecs:
            return
        tag = f"{self.source}_{self.shard:03d}"
        np.save(CACHE / f"{tag}.npy", np.stack(self.vecs))
        pd.DataFrame(self.rows).to_csv(f"data/manifests/rows_{tag}.csv", index=False)
        print(f"  flushed {tag}: {len(self.vecs)} embeddings")
        self.vecs, self.rows = [], []
        self.shard += 1


def main(a):
    emb = Embedder()
    writer = ShardWriter(a.source)

    ds = load_dataset(a.dataset, split=a.split, streaming=True)
    if a.shuffle:
        ds = ds.shuffle(seed=42, buffer_size=10_000)

    counts = {0: 0, 1: 0}
    pbar = tqdm(total=a.n_per_class * 2, desc=a.source)

    for ex in ds:
        sub = str(ex.get(a.label_field, "")).lower()
        if "tamper" in sub:
            continue                       # different task — DATA_LAYOUT 4.1
        y = 0 if "real" in sub else 1

        if counts[y] >= a.n_per_class:
            if all(c >= a.n_per_class for c in counts.values()):
                break
            continue
        counts[y] += 1

        img = normalise(ex[a.image_field])          # fix #2
        iid = image_id(img)

        # this image is in memory once and never again
        variants = (all_variants(img, iid) if a.full_grid
                    else sample_variants(img, iid, k=a.n_sampled))

        batch = emb.embed_batch([v for _, v in variants])
        for (name, _), vec in zip(variants, batch):
            writer.add(vec, {
                "image_id": iid, "variant": name, "label": y,
                "source": a.source, "subclass": sub,
                "generator": str(ex.get("generator", "unknown")),
                "split": a.assign_split,
            })
        pbar.update(1)

    pbar.close()
    writer.flush()
    print(f"done: {sum(counts.values())} images {counts}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--source", required=True)
    p.add_argument("--assign-split", required=True)
    p.add_argument("--n-per-class", type=int, default=20_000)
    p.add_argument("--image-field", default="image")
    p.add_argument("--label-field", default="label")
    p.add_argument("--full-grid", action="store_true",
                   help="all 15 variants — use for EVAL splits")
    p.add_argument("--n-sampled", type=int, default=3,
                   help="variants sampled per image when not --full-grid")
    p.add_argument("--shuffle", action="store_true")
    main(p.parse_args())
```

### 6a. Inspect one record — do not skip

Field names and label formats vary. Guessing costs two hours.

`scripts/inspect_dataset.py`:

```python
import sys
from datasets import load_dataset

name, split = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "train"
ex = next(iter(load_dataset(name, split=split, streaming=True)))
for k, v in ex.items():
    print(f"{k:20s} {type(v).__name__:12s} {str(v)[:60] if k != 'image' else v.size}")
```

```bash
python scripts/inspect_dataset.py saberzl/SID_Set train
python scripts/inspect_dataset.py saberzl/So-Fake-OOD test_image
```

### 6a-bis. Confirmed dataset facts (checked 2026-08-26)

Verified against the HF datasets-server, so nobody has to guess again.

| | SID_Set train | SID_Set validation | So-Fake-OOD |
|---|---|---|---|
| split name | `train` | `validation` | **`test_image`** (not `test`) |
| rows | 210,000 | 30,000 | 91,370 |
| size | 123 GB | 17 GB | **135 GB** |
| per image | 573 KB | 547 KB | **1.44 MB** |
| label field | `label`, **int** | `label`, int | `label`, **string** |
| label values | 0 / 1 / 2 | same | `REAL` / `FULL_SYNTHETIC` / `TAMPERED` |
| image field | `image` | `image` | `image` |
| generator field | absent | absent | `generator`, populated for fakes |

`INT_MAP = {0: 0, 1: 1, 2: None}` in `stream_embed.py` is **confirmed**: SID_Set
label 0 rows carry a bare-hash `img_id`, label 1 rows are `full_synthetic_*`, and
label 2 rows are `tampered_*`. So-Fake-OOD spells the same three classes out as
strings, which `to_label()` handles on its string path. Both orderings agree, so
one mapping covers both. The classes run at roughly one third each.

So-Fake-OOD generators seen in sampling: GPT-image-1.5, GPT-image-2, GPT4o,
Imagen3, Imagen4, Flux.1_pro, Hidream, Recraftv3, Ideogram2/3, Seedream3.0,
seedream4.5, nano_banana, nano_banana_2. That is the unseen-generator table, free.

### 6a-ter. Bandwidth — the real schedule risk

Streaming fetches whole rows, so the ~1/3 of rows that are `tampered` are paid
for and discarded. Cost of the documented run:

| Pass | Kept | Rows streamed | Download |
|---|---|---|---|
| SID_Set train, 20k/class | 40k | ~60k | **~34 GB** |
| SID_Set validation, 3k/class | 6k | ~9k | ~5 GB |
| So-Fake-OOD, 5k/class | 10k | ~15k | **~22 GB** |
| | | | **~61 GB** |

Not the 19 GB an earlier draft assumed. At 100 Mbps that is ~1.5 h of pure
transfer on top of GPU time; at 20 Mbps it is most of a day.

**Halve it if the link is slow.** A frozen-CLIP linear probe converges long
before 20k/class — `--n-per-class 8000` on train, `3000` on So-Fake-OOD brings
the total to ~30 GB with no measurable accuracy cost. Scale back up later if the
probe is still improving; the shard naming makes a second pass additive.

### 6b. Smoke test

```bash
python scripts/stream_embed.py \
  --dataset saberzl/SID_Set --split train \
  --source smoke --assign-split train \
  --n-per-class 20 --full-grid
```

Under a minute. Expect `600 embeddings` (40 images × 15).

Verify:
```bash
python -c "
import numpy as np, pandas as pd
v = np.load('data/cache/embeddings/vitl14_v1/smoke_000.npy')
r = pd.read_csv('data/manifests/rows_smoke_000.csv')
print(v.shape, len(r), r.variant.nunique())
assert v.shape[0] == len(r), 'ROW MISALIGNMENT'
"
```

Then `rm data/cache/embeddings/vitl14_v1/smoke_* data/manifests/rows_smoke_*`.

### 6c. Real runs

```bash
# TRAIN — sampled variants (fix #6). ~160k embeddings.
python scripts/stream_embed.py \
  --dataset saberzl/SID_Set --split train \
  --source sid_train --assign-split train \
  --n-per-class 20000 --n-sampled 3 --shuffle

# CALIB — full grid: calibrators must be valid across the degradation range
python scripts/stream_embed.py \
  --dataset saberzl/SID_Set --split validation \
  --source sid_calib --assign-split calib \
  --n-per-class 3000 --full-grid

# OOD EVAL — full grid, this produces the required robustness table
python scripts/stream_embed.py \
  --dataset saberzl/So-Fake-OOD --split test_image \
  --source so_fake_ood --assign-split test_ood \
  --n-per-class 5000 --full-grid
```

Run under `tmux`. An SSH drop or a closed laptop lid kills a two-hour job.

**Crash recovery:** shards are already written. Rerun with a new `--source`
(e.g. `sid_train_b`) and a smaller `--n-per-class`; the manifest globs all
`rows_*.csv`, so shards merge cleanly.

**If VRAM still spikes:** `--full-grid` sends a batch of 15. Drop to
`--n-sampled 2` on train, or chunk inside `embed_batch`.

---

## Step 7 — Manifest and assertions

`scripts/load_embeddings.py` — shard-aware loader teammates use:

```python
import glob, re
import numpy as np, pandas as pd
from quorum.embed import CACHE

def load_source(source: str):
    """Returns (X, rows) with X[i] corresponding to rows.iloc[i]."""
    shards = sorted(glob.glob(str(CACHE / f"{source}_*.npy")))
    Xs, Rs = [], []
    for s in shards:
        tag = re.sub(r"\.npy$", "", s.split("/")[-1])
        Xs.append(np.load(s))
        Rs.append(pd.read_csv(f"data/manifests/rows_{tag}.csv"))
    X = np.concatenate(Xs)
    R = pd.concat(Rs, ignore_index=True)
    assert len(X) == len(R), f"{source}: {len(X)} vecs vs {len(R)} rows"
    return X, R
```

`scripts/build_manifest.py`:

```python
import glob
import pandas as pd

df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("data/manifests/rows_*.csv"))],
               ignore_index=True)
df = df[~df.source.str.startswith("smoke")]
# ... append rows for organizer_val / faces / openimages / social_real ...

# ---------------- ASSERTIONS: must fail loudly ----------------

# A. organizer validation quarantined
val = df[df.source == "organizer_val"]
assert len(val) > 0, "organizer val missing — cannot report reference number"
assert (val.split == "test_organizer").all(), "ORGANIZER VAL LEAKED INTO TRAINING"

# B. OOD set is eval-only apart from the deliberate calib_ood carve, and the two
#    sides must share no generator family and no image. See HANDOVER.md 5d.
ood = df.source == "so_fake_ood"
assert df[ood].split.isin(["test_ood", "calib_ood"]).all(), "OOD LEAKED"
ai = df[ood & (df.label == 1)]
gen_cal = set(ai[ai.split == "calib_ood"].generator)
gen_ev = set(ai[ai.split == "test_ood"].generator)
assert gen_cal and gen_ev, "carve produced an empty side"
assert not (gen_cal & gen_ev), f"generator on both sides: {gen_cal & gen_ev}"

# C. calibration split trains nothing
assert (df[df.source == "sid_calib"].split == "calib").all()

# D. no image_id spans two splits
spans = df.groupby("image_id").split.nunique()
assert (spans == 1).all(), f"{(spans > 1).sum()} image_ids span splits"

# E. class balance sane in train
tr = df[(df.split == "train") & (df.variant == "clean")]
r = tr.label.mean()
assert 0.35 < r < 0.65, f"train is {r:.0%} AI — rebalance"

# F. variant counts match the grid
from quorum.degrade import N_VARIANTS
ev = df[df.split.str.startswith("test") | (df.split == "calib")]
per = ev.groupby("image_id").variant.nunique()
assert (per == N_VARIANTS).all(), f"eval images missing variants (expect {N_VARIANTS})"

df.to_csv("data/manifests/main.csv", index=False)

with open("data/manifests/stats.md", "w") as f:
    clean = df[df.variant == "clean"]
    f.write("# Manifest stats\n\n## Rows per split (clean only)\n\n")
    f.write(pd.crosstab(clean.split, clean.label).to_markdown())
    f.write("\n\n## Sources\n\n")
    f.write(pd.crosstab(clean.source, clean.split).to_markdown())
    f.write(f"\n\n## Variants\n\nGrid = {N_VARIANTS} (14 settings + clean)\n")
    f.write("\n## Assertions\n\nA-F all passed.\n")
```

Assertion F is new — it catches a partially-crashed eval shard, which would
silently produce an incomplete robustness table.

```bash
python scripts/build_manifest.py && cat data/manifests/stats.md
```

**Nobody trains until this is green.** A contaminated split does not look like a
bug — it looks like an unusually good number.

Commit `main.csv` and `stats.md`.

---

## Step 8 — Push the cache

```bash
hf repo create quorum-cache --repo-type dataset --private
hf upload YOUR_ORG/quorum-cache data/cache/embeddings --repo-type dataset
```

`scripts/pull_cache.py`:

```python
from huggingface_hub import snapshot_download
snapshot_download("YOUR_ORG/quorum-cache", repo_type="dataset",
                  local_dir="data/cache/embeddings")
```

Add teammates as collaborators on the HF repo.

---

## Step 9 — Handoff

```bash
git add -A && git commit -m "data pipeline: streaming embed, manifest, assertions" && git push
```

Into `README.md`:

````markdown
## Getting started (team)

```bash
git clone <repo> && cd quorum
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
hf auth login
python scripts/pull_cache.py          # ~1.2GB, no images
```

```python
from scripts.load_embeddings import load_source
X, rows = load_source("sid_train")    # X (N,768) aligns 1:1 with rows
```

### Rules

- `train` trains. `calib` and `calib_ood` fit calibrators and fusion only.
  `test_ood` and `test_organizer` are never fitted on.
- **`calib_ood` is carved out of So-Fake-OOD by generator family**, so selecting
  rows by `source == "so_fake_ood"` without filtering `split` puts your eval set
  in your training set. Filter by split, always. `load()` takes `split` from the
  manifest, never from the shard.
- Every `train_*.py` takes a required `--manifest` argument. No default.
- Do not re-run the embedding pass. If you think you need to, ask — it changes
  everyone's numbers.
- `data/raw/organizer_val/` is the competition validation set. Never train on it.
- Train rows have 4 variants per image (clean + 3 sampled); eval rows have all 15.
- **Never junction `data/` into a git worktree.** `git worktree remove --force`
  follows Windows directory junctions and deletes the target — it wiped the
  embedding cache once already.
````

| Owner | Task | Reads |
|---|---|---|
| Detector | `detectors/general.py` — probe on embeddings | `PIPELINE.md` §2.1 |
| Fusion | `detectors/face.py` + crops (§10 below), then `fusion.py` | `PIPELINE.md` §2.2, §5 |
| Fusion | `detectors/text.py`, `detectors/spectral.py` | `PIPELINE.md` §6, §7 |
| Data/Eval | `eval_grid.py` — the robustness table | `SPEC.md` Phase 3 |
| Data/Eval | `make_figures.py` — all six figures, recomputed from cache | `HANDOVER.md` §5h |
| Data/Eval | `error_cases.py` — the ten stable errors; needs organizer_val PIXELS | `ERROR_ANALYSIS.md` §5 |
| Frontend | demo against `predict.py` output | `SPEC.md` §8 |
| Frontend | error analysis note | `SPEC.md` Phase 6 |

---

## Step 10 — Face detector on Python 3.13 (fix #5)

**mediapipe has no 3.13 wheels.** Rather than a second venv, use **OpenCV YuNet**
— it ships inside `opencv-python`, has no version constraint, and returns exactly
the 5 landmarks alignment needs.

```python
import cv2, numpy as np

# one-time: download the ~85KB model
# https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
MODEL = "data/models/face_detection_yunet_2023mar.onnx"

def make_detector(w, h):
    return cv2.FaceDetectorYN.create(MODEL, "", (w, h),
                                     score_threshold=0.7, nms_threshold=0.3)

def detect_faces(bgr):
    """Returns [(box, landmarks5)] — landmarks are
    right-eye, left-eye, nose, right-mouth, left-mouth."""
    h, w = bgr.shape[:2]
    _, faces = make_detector(w, h).detect(bgr)
    if faces is None:
        return []
    out = []
    for f in faces:
        box = f[:4]                          # x, y, w, h
        lm = f[4:14].reshape(5, 2)           # 5 landmark points
        out.append((box, lm))
    return out
```

**Landmark order differs from ArcFace** — YuNet gives right-eye first, ArcFace's
template is left-eye first. Swap rows 0/1 and 3/4 before computing the similarity
transform, or your faces come out mirrored.

Fallback options if YuNet underperforms: `insightface` (check 3.13 wheels), or a
3.11 venv for the face branch only. Do not block on this — the general detector
does not need it.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `KeyError: 'image'` | Field name differs | Run §6a, set `--image-field` |
| CUDA OOM | Batch 15 too large in 8GB | Confirm fp16 active; drop `--n-sampled` |
| Streaming stalls | HF rate limit / network | Resume with a new `--source` shard |
| `label` is an int | Class indices, not strings | Edit `sub`/`y` in `stream_embed.py` |
| Assertion D fires | Same image in two splits | Datasets overlap — drop dupes, keep eval copy |
| Assertion F fires | Eval shard crashed midway | Rerun that source with `--full-grid` |
| Embeddings not unit norm | L2 line removed | Restore; probes misbehave subtly |
| `mediapipe` install fails | No 3.13 wheels | Expected — use YuNet (§10) |
| Faces come out mirrored | YuNet vs ArcFace landmark order | Swap rows 0/1 and 3/4 |

---

## Time budget

| Step | Time |
|---|---|
| 1–2 scaffold + size check | 30 min |
| 3–4 `degrade.py` + `embed.py` + checks | 45 min |
| 5 small downloads | 20 min (background) |
| 6a–6b inspect + smoke | 20 min |
| 6c real runs | **90–150 min unattended** |
| 7 manifest + assertions | 30 min |
| 8–9 push + handoff | 20 min |

Start 6c before lunch. While it runs, write `predict.py` as a stub that walks a
directory and emits `{image_path, pred}` with random scores — it is a required
deliverable and it unblocks your frontend people immediately.
