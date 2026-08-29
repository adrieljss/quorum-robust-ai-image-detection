# Quorum — Data Layout & Download Guide

Exact folder tree, what goes where, how to get it, and how to share it across
five people without everyone downloading everything.

**Local disk budget: under 15GB.** **Git budget: under 20MB.**

> **Check every dataset's real size before downloading.** Sizes on paper and
> sizes on disk diverge badly — SID_Set is 140GB, not the 30GB an earlier draft
> of this document guessed.
>
> ```python
> from huggingface_hub import HfApi
> info = HfApi().dataset_info("saberzl/SID_Set", files_metadata=True)
> print(sum(f.size for f in info.siblings if f.size) / 1e9, "GB")
> ```
>
> Run this for SID_Set, So-Fake-OOD, and the faces set before anyone starts a
> download. Two minutes now saves a stalled day.

---

## 1. The single most important rule

**COCO val2017 (5,000 images) and the WildFake DALL·E Advanced subset (3,719
images) are the organizer's validation set. They must never enter training.**

Training on them makes every number meaningless and would look like cheating in
a public repo. An assertion in §6 enforces it. This supersedes any earlier
guidance that put COCO val2017 in the real class.

---

## 2. The tree

```
quorum/
└── data/
    ├── raw/                          # downloads, NEVER modified, NEVER committed
    │   │
    │   │   # SID_Set (140GB) and So-Fake-OOD are STREAMED, not stored.
    │   │   # See §5.1. Nothing lands here for them.
    │   │
    │   ├── organizer_val/            # ~2GB   ← EVAL ONLY, QUARANTINED
    │   │   ├── coco_val2017/
    │   │   └── wildfake_dalle_adv/
    │   │
    │   ├── real_extra/               # ~1GB   ← REAL-CLASS DIVERSITY
    │   │   └── openimages_sample/
    │   │
    │   ├── faces/                    # ~2GB   ← FACE BRANCH ONLY
    │   │
    │   └── social_real/              # ~200MB ← MANUALLY COLLECTED
    │
    ├── manifests/                    # COMMITTED
    │   ├── main.csv
    │   └── stats.md
    │
    ├── cache/                        # generated, gitignored, shared via HF Hub
    │   ├── normalised/
    │   ├── embeddings/
    │   ├── face_crops/
    │   └── scores/
    │
    └── models/                       # COMMITTED, ~2MB total
        └── *.pkl
```

`.gitignore`:
```
data/raw/
data/cache/
```

---

## 3. What goes where

| Dataset | Full size | On your disk | Access | Role |
|---|---|---|---|---|
| SID_Set | **140GB** | **0** | **stream** | Primary training |
| So-Fake-OOD | **135GB** | **0** | **stream** | Headline eval |
| COCO val2017 + WildFake DALL·E | ~2GB | 2GB | download | Reference benchmark, **quarantined** |
| Open Images sample | ~1GB | 1GB | download | Real diversity |
| DF40 faces | ~2GB | 2GB | download | Face branch |
| Social screenshots | 200MB | 200MB | you collect | Platform reals |

Streamed datasets never touch disk. What you keep is **embeddings** — 40k images
× 4-15 variants ≈ 1.2GB of `.npy`, versus 140GB of pixels you would never look at
again.

---

## 4. Why these datasets

### 4.1 SID_Set — primary training

Organizer-listed, and a genuine fit: 300K AI-generated/tampered and authentic
images built specifically for social media detection, spanning fully synthetic
and tampered categories, with realism high enough that visual inspection alone
often fails. CVPR 2025 (SIDA paper).

Three classes: `real`, `full_synthetic`, `tampered`.

**Use `real` vs `full_synthetic` to train the general probe.** `tampered` is a
different problem — localised manipulation of a real photo, which is globally
authentic — so it gets its own probe rather than being folded into the general
one. `train_tampered()` fits tampered-vs-SID-reals and never sees a synthetic
image; `predict.py` ships `max()` of the two.

> **Superseded, 30 Aug.** This paragraph used to read "out of scope for AIGC
> detection as the brief frames it", which contradicted both `HANDOVER.md` §5g
> and what actually ships. The problem statement's §5.1 lists "or lightly
> edited" among the post-processing operations, and the project's own definition
> of robust is "survives editing". **Tampered images are in scope and the branch
> ships.** The full measurement of what that costs and buys is `HANDOVER.md`
> §5h; the counterfactual is `docs/figures-no-tampered/`.

**Caveat to check on download:** SID_Set may not carry per-generator labels. That
is fine here — the unseen-generator guarantee comes from So-Fake-OOD being a
purpose-built OOD set, not from splitting SID_Set by generator.

### 4.2 So-Fake-OOD — the headline eval

~100k images sourced from Reddit, explicitly evaluation-only, using generators
not seen during training including GPT-4o, Imagen3 and HiDream. Carries a
`generator` field per image. CC BY 4.0.

This is better than manufacturing a held-out split yourself:
- Modern generators, including ones no public detector trained on
- Real social-media provenance, not clean benchmark photos
- Same research group as SID_Set, so the train/test relationship is intentional
- Per-generator breakdown for free — a good results table

**Never train on it.** The dataset card says so.

### 4.3 Organizer validation — quarantined

COCO val2017 (5,000 real) + WildFake DALL·E Advanced (3,719 AI — see the note
below; it is NOT 8,843). The brief provides this to demonstrate performance and
track iteration. It does **not** count toward
the final score, and must not be trained on.

Keep it in its own folder with its own manifest split. Report it as a reference
number in the README.

### 4.4 Real diversity — Open Images, not COCO val

SID_Set's reals and So-Fake-OOD's reals both come from social media, which is
good. Add a second real provenance so the model cannot key on one source.

**Use Open Images or COCO train2017.** Not val2017 — that is the quarantine set.

### 4.5 Datasets we are not using, and why

State these in the README; deliberate exclusions read as judgement, not
oversight.

- **CIFAKE** (organizer-listed) — 32×32 resolution. Too small for CLIP's 224px
  input and for any meaningful robustness testing; the transform grid is
  meaningless at that resolution.
- **WildFake beyond the validation subset** — usable, but the ModelScope download
  is awkward and SID_Set covers the same ground. Reconsider if SID_Set access
  fails.
- **GenImage** — ~500GB full, ~97GB per subset. See §9.
- **IEEE DataPort v2** — good dataset, but SID_Set + So-Fake-OOD covers the same
  need with better social-media alignment and easier access.

---

## 5. Downloads

### 5.1 SID_Set — streamed, never stored

**140GB on disk if you download it. Do not download it.**

You need ~40k images; SID_Set has 300k. Stream, subsample, embed, discard.

```python
# scripts/stream_embed.py
from datasets import load_dataset
from quorum.embed import Embedder
from quorum.degrade import TRANSFORMS, apply
import numpy as np, hashlib, io

emb = Embedder()
N_PER_CLASS = 20_000
rows, vecs = [], []
counts = {0: 0, 1: 0}

ds = load_dataset("saberzl/SID_Set", split="train", streaming=True)
ds = ds.shuffle(seed=42, buffer_size=10_000)

for ex in ds:
    sub = ex["label"]                       # real / full_synthetic / tampered
    if sub == "tampered":
        continue                            # different task — see §4.1
    y = 0 if sub == "real" else 1
    if counts[y] >= N_PER_CLASS:
        if all(c >= N_PER_CLASS for c in counts.values()):
            break
        continue
    counts[y] += 1

    img = ex["image"].convert("RGB")
    iid = hashlib.sha256(img.tobytes()).hexdigest()[:16]

    # ALL variants in one pass — the image is in memory now and never will be again
    variants = [("clean", img)] + [
        (f"{k}{p}".replace(".", ""), apply(img, k, p))
        for k, ps in TRANSFORMS.items() for p in ps
    ]
    batch = emb.embed_batch([v for _, v in variants])
    for (name, _), vec in zip(variants, batch):
        vecs.append(vec)
        rows.append({"image_id": iid, "variant": name, "label": y,
                     "source": "sid_set", "subclass": sub})

np.save("data/cache/embeddings/vitl14_v1/sid_set.npy", np.stack(vecs))
pd.DataFrame(rows).to_csv("data/manifests/sid_set_rows.csv", index=False)
```

**Why this is better than downloading**, not just smaller:

- One pass over each image produces every variant you will ever need. Download-first
  means a second full pass when Phase 3 starts.
- Bandwidth ≈ 140GB × (40k/300k) ≈ 19GB, and **zero** disk.
- Output is ~1.2GB of embeddings that the whole team can share (§7).

Runtime: roughly 1–2 GPU-hours for 40k × 15 = 640k embeddings. Start it early and
let it run.

**Fallback if streaming is unreliable:** the Google Drive mirror at
`https://github.com/hzlsaber/SIDA` ships `train.zip` and `validation.zip`
separately, so you can take validation only, or one part of the split
`train_full_synthetic`. Still large — prefer streaming.

### 5.2 So-Fake-OOD — streamed, never stored

Same pattern, `split="test_image"` (not `test`). Subsample to ~10k for iteration speed; run the full
set once at the end for the final number.

```python
ds = load_dataset("saberzl/So-Fake-OOD", split="test_image", streaming=True)
# keep ex["generator"] — it gives you a free per-generator results table
```

### 5.3 Organizer validation — `raw/organizer_val/`

```bash
mkdir -p data/raw/organizer_val/coco_val2017
cd data/raw/organizer_val/coco_val2017
curl -O http://images.cocodataset.org/zips/val2017.zip
unzip -q val2017.zip && rm val2017.zip
ls val2017 | wc -l    # expect 4998–5000
```

WildFake DALL·E Advanced comes from ModelScope (`hy2628982280/WildFake`). Do NOT
download it by hand: the images ship in one 25.6 GB `DALLE.zip` and the subset we
want is ~1.5 GB of it. `scripts/fetch_wildfake.py` reads the zip's central
directory over HTTP range requests and inflates only `DALLE/Advanced/DALLE3`.

```bash
python scripts/fetch_wildfake.py --list   # asserts the subset is still 3,719
python scripts/fetch_wildfake.py          # ~1.5 GB, resumable
```

**3,719 distinct images. WildFake files them as 8,843 entries under prompt-run
folders, but 1,808 basenames repeat with an identical CRC32 and size -- the
names are content hashes. Unique (CRC, size) pairs total 3,719 exactly,
verified from the zip's central directory. The 8,843 figure the brief quotes is
a FILE count; anyone reporting it as an image count is off by 5,124 duplicates
that `image_id` silently drops.**

Put a `README_DO_NOT_TRAIN.txt` in this folder. It costs nothing and it stops the
mistake at 3am.

### 5.4 Real diversity — `raw/real_extra/`

Open Images sample via FiftyOne, or COCO train2017:

```bash
pip install fiftyone
python -c "
import fiftyone.zoo as foz
foz.load_zoo_dataset('open-images-v7', split='validation', max_samples=5000,
                     dataset_dir='data/raw/real_extra/openimages_sample')
"
```

### 5.5 Faces — `raw/faces/`

```bash
hf download pujanpaudel/deepfake_face_classification \
  --repo-type dataset --local-dir data/raw/faces
```

16k real / 16k fake, pre-split, CC BY-NC 4.0. Face branch only — do not merge
into the general manifest.

### 5.6 Social reals — `raw/social_real/`

Manual. 200–500 screenshots of real photos from social feeds.

Tedious, and the only data that tells you whether the detector survives contact
with the deployment surface. Assign it to whoever is blocked waiting on the
SID_Set download.

---

## 6. Manifest

```bash
python scripts/build_manifest.py \
  --raw-dir data/raw \
  --out data/manifests/main.csv \
  --stats data/manifests/stats.md
```

### 6.1 Schema

| column | notes |
|---|---|
| `image_id` | content hash, 16 hex chars — **the join key** |
| `path` | points at `cache/normalised/`, not `raw/` |
| `label` | 1 = AI, 0 = real |
| `source` | `sid_set`, `so_fake_ood`, `organizer_val`, `openimages`, ... |
| `generator` | from So-Fake-OOD's field; `unknown` for SID_Set; `real` for reals |
| `subclass` | `real` / `full_synthetic` / `tampered` |
| `split` | `train` / `calib` / **`calib_ood`** / `test_ood` / `test_organizer` / `test_wild` |
| `orig_format` | `jpg` / `png` / `webp` |

Use a **content hash** for `image_id`, not the filename. Filenames collide across
datasets.

### 6.2 Normalise before hashing

Re-encode everything to **JPEG quality 95** before computing the hash, so the
hash matches what you actually embed, and so file format cannot leak the label.

```python
def normalise(src, dst):
    Image.open(src).convert("RGB").save(dst, "JPEG", quality=95)
```

Write to `data/cache/normalised/`. Leave `raw/` untouched.

### 6.3 Required assertions

All four must **fail loudly**, not warn.

**A. Organizer validation is quarantined**

```python
val = df[df.source == "organizer_val"]
assert (val.split == "test_organizer").all(), "organizer val leaked into training"
assert len(val) > 0, "organizer val missing — cannot report reference number"
```

**B. So-Fake-OOD is eval-only**

```python
ood = df[df.source == "so_fake_ood"]
assert ood.split.isin(["test_ood", "calib_ood"]).all(), "OOD set leaked into training"
# calib_ood is the generator-family-disjoint calibration carve (HANDOVER.md 5d).
# build_manifest.py additionally asserts the two sides share no generator and no image.
```

**C. Format does not predict label**

Some public sets store reals as JPG and fakes as PNG; published work has
accidentally trained JPEG-artifact detectors this way.

```python
ct = pd.crosstab(df.orig_format, df.label, normalize="index")
assert ct.max().max() < 0.75, f"format predicts label:\n{ct}"
```

**D. Source does not trivially predict label**

```python
for src, grp in df[df.split == "train"].groupby("source"):
    if grp.label.nunique() == 1 and len(grp) > 0.3 * len(df[df.split=="train"]):
        raise AssertionError(f"{src} is single-label and dominant — real class will key on it")
```

### 6.4 Expected stats

```
split             real     ai      total
train            18000   18000    36000
calib             3000    3000     6000
test_ood         (from So-Fake-OOD, ~100k — subsample to 10k for speed)
test_organizer    5000    3719     8719
test_wild          400       0      400
```

Commit `stats.md`. It is both your record that the split was clean and a slide.

---

## 7. Sharing data across the team

**Nothing large goes in git.** GitHub caps single files at 100MB, and Git LFS's
free tier (1GB storage, 1GB/month bandwidth) breaks the moment five people pull.

**The key insight: after the embedding pass, nobody needs the images.** An
embedding is 768 floats where the image was ~150k numbers. 36k images ≈ 40GB;
36k embeddings ≈ 110MB.

Workflow:

1. **One person** (whoever has the GPU) downloads, normalises, builds the
   manifest, runs the embedding pass
2. They push `manifest.csv` + `embeddings/` to a shared HF Hub dataset repo
3. **Everyone else pulls ~1GB and never touches an image**

```bash
hf repo create quorum-cache --repo-type dataset --private
hf upload your-org/quorum-cache data/cache/embeddings \
  --repo-type dataset
```

That turns 5 × 40GB into 1 × 40GB + 4 × 1GB, and unblocks four people from a
long download.

**Version the cache.** If someone re-runs embedding with different preprocessing,
stale `.npy` files silently corrupt everyone's results. Put backbone name and a
preprocessing version in the path:

```
embeddings/vitl14_v1/{image_id}__{variant}.npy
```

Alternative: GitHub Releases (2GB/file, no repo bloat). Avoid Git LFS.

---

## 8. Cache sizing and the augmentation trap

| Cache | Count | Size |
|---|---|---|
| `embeddings/` train, 4 variants/image | ~160k | ~0.5GB |
| `embeddings/` calib + eval, 15 variants/image | ~240k | ~0.7GB |
| `normalised/` (downloaded sets only) | ~15k | ~4GB |
| `face_crops/` | ~15k | 300MB |

No `normalised/` entries for streamed data — those images never exist on disk.

**Augmented images are different images and need their own embeddings.** You
cannot degrade a vector after the fact. The streaming script in §5.1 handles this
correctly by generating all 15 variants while the image is in memory. **Do not
"optimise" it into a clean-only pass** — you would be re-streaming 19GB later.

The official grid is 14 settings (4 JPEG + 3 blur + 2 resize + 3 noise + 1 jitter
+ 1 crop) plus clean = 15 variants.

**On a laptop GPU, sample variants on train and keep the full grid only on eval.**
CPU transforms, not the GPU, are the bottleneck; this halves runtime with no
effect on the required robustness table. See `RUNBOOK.md` fix #6.

---

## 9. GenImage — optional, probably skip

~500GB full, ~97GB for one subset. The most common way a hackathon project stalls
on day one.

You are not data-limited — a frozen-CLIP linear probe converges on ~2–3k images
per generator. SID_Set already gives you far more than that.

If you want it anyway, best option is a **Kaggle notebook with the dataset
attached** (per-subset mirrors indexed at
`https://github.com/vtphatt2/GenImage-mirror`): run embedding extraction there,
export only `.npy`. You never store 97GB.

Note its format trap — reals as JPG, fakes as PNG. §6.2 normalisation handles it,
assertion C catches it if you forget.

---

## 10. Ownership and timing

| Task | Owner | When |
|---|---|---|
| Run the size-check script on all datasets | Data/Eval | hour 0, first thing |
| `degrade.py` + variant naming | Data/Eval | hour 0 — **blocks streaming** |
| `embed.py` + versioned cache | Detector | hour 0 — **blocks streaming** |
| Download organizer val + faces + Open Images | Fusion | hour 0, parallel |
| Collect social screenshots | Frontend | hour 0, while blocked |
| `stream_embed.py` on SID_Set | Detector | hour 3, runs 1–2h unattended |
| `stream_embed.py` on So-Fake-OOD | Detector | after SID_Set |
| Build manifest from streamed rows + downloads | Data/Eval | after streaming |
| Verify assertions, commit `stats.md` | Data/Eval | before anyone trains |
| Push embeddings to HF cache | Detector | after streaming |

**Note the dependency inversion versus a download-first plan.** `degrade.py` and
`embed.py` now block the data pass rather than following it, because variants are
generated inline. Get those two written first — they are small, and everything
waits on them.

**Nobody trains until `stats.md` is green.** A contaminated split does not look
like a bug — it looks like an unusually good number, and you will not catch it
later.
