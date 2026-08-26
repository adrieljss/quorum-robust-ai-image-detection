# Quorum — Pipeline Implementation Guide

File bytes → model input. Companion to `SPEC.md` and `DATA_LAYOUT.md`.

Covers `provenance.py`, `embed.py`, `degrade.py`, branch preprocessing, and the
frozen model input contracts. Out of scope: training loops, fusion, demo.

*(Replaces the earlier `FRONTEND_PIPELINE.md`, which was misnamed and predated
the official brief.)*

---

## 1. Stage map

```
file bytes
   │
   ├─ manifest lookup ──────────► image_id, label, split
   │
   ├─ T0  provenance.py ────────► C2PA verdict  [may short-circuit; LOW PRIORITY]
   ├─ T0b metadata features ────► EXIF / PNG chunk flags  [never gates]
   │
   ├─ decode + normalise ───────► RGB uint8
   │
   ├─ T1  embed.py ─────────────► 768 floats → general probe
   ├─ T2a face crop ────────────► 768 floats → face probe
   ├─ T2b OCR regions ──────────► 6 scalars  → text scorer
   └─ T2c high-pass FFT ────────► 8 scalars  → spectral scorer
```

Left of the arrows is preprocessing. Right is a model. The models are small; the
preprocessing is where the work is.

**No VLM branch.** Cut on parameter-cap grounds (§2 of `SPEC.md`).

---

## 2. Model input contracts

**Freeze these before anyone writes a training script.** They are what let three
people work in parallel without integration pain.

### 2.1 General probe

```
input:  float32[768]   L2-normalised CLIP embedding of the full image
output: float32        raw score [0,1], UNCALIBRATED
model:  LogisticRegression, or Linear(768→256)→ReLU→Dropout(0.2)→Linear(256→1)
train:  X = np.stack(embeddings)   # (N, 768)
        y = manifest.label.values  # (N,)
```

Start with `sklearn.linear_model.LogisticRegression(max_iter=2000)`. If the MLP
does not clearly beat it on So-Fake-OOD, keep the linear model — fewer knobs,
less overfit surface, and it trains in two seconds.

### 2.2 Face probe

```
input:  float32[768]   L2-normalised CLIP embedding of an ALIGNED 224px face crop
output: float32        raw score [0,1], UNCALIBRATED
        bool           face_present
```

Same model class, different preprocessing (§5). **Same CLIP instance** — this is
what keeps the parameter budget at ~317M.

### 2.3 Text scorer

```
input:  float32[6]     hand-built OCR features (§6)
output: float32        raw score [0,1]
        bool           text_present
model:  LogisticRegression  (six features — a threshold would also work)
```

### 2.4 Spectral scorer

```
input:  float32[8]     radial FFT band energies + peak stats (§7)
output: float32        raw score [0,1]
model:  LogisticRegression
```

### 2.5 Fusion (reference)

```
input:  float32[~14]  [ cal_general, cal_face, face_present,
                        cal_text, text_present, cal_spectral,
                        content_onehot(5), degradation_estimate,
                        provenance_prior ]
output: float32       calibrated P(AI)
```

**Missing-branch rule:** presence flag + neutral `0.5` fill. "No face here" must
never read as "the face model says real."

---

## 3. T0 — Provenance (low priority)

**Deprioritised against the brief.** The organizer validation set is COCO +
DALL·E; C2PA will fire on essentially none of it. Keep it as a small innovation
talking point and a real-deployment argument, but it is not day-one work and it
is first on the cut list after spectral.

### 3.1 C2PA vs EXIF — not comparable

C2PA is cryptographically signed and may gate a decision. EXIF is plaintext
anyone can forge in one line of Python and may only ever be a feature. Do not
put them in the same tier.

### 3.2 The four outcomes

| # | Condition | Action | Confidence |
|---|---|---|---|
| 1 | Valid signature + AI/generative assertion | **SHORT-CIRCUIT** → AI | 0.98 |
| 2 | Valid signature + hardware capture + trusted cert | Strong prior, continue | cap 0.15 |
| 3 | Signature present but invalid | Low-reliability flag, continue | no change |
| 4 | No manifest | Nothing | no change |

**Case 2 does not short-circuit.** C2PA proves a file was signed by a device — it
cannot verify the camera was pointed at what it claims. Photograph a monitor
showing an SDXL image with a Pixel 10 and you get a valid hardware-backed
credential on a picture of a fake. First-mile trust gap.

**Case 3 is not suspicious.** Ordinary re-encoding breaks signatures.

**Case 4 is not evidence.** Absence proves only absence of provenance.

### 3.3 Implementation sketch

```python
# quorum/provenance.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class ProvenanceResult:
    present: bool = False
    signature_valid: Optional[bool] = None
    assurance_level: Optional[int] = None    # 2 = hardware-backed
    trusted_signer: bool = False
    has_ai_assertion: bool = False
    has_capture_assertion: bool = False
    chain_depth: int = 0
    short_circuit: bool = False
    confidence: Optional[float] = None
    prior: Optional[float] = None            # fed to fusion when no gate

def check_provenance(path: str) -> ProvenanceResult:
    r = ProvenanceResult()
    try:
        import c2pa
        manifest = c2pa.Reader.from_file(path).json()
    except Exception:
        return r                              # case 4 — the common case

    r.present = True
    r.signature_valid = _validate(manifest)
    if not r.signature_valid:
        return r                              # case 3

    # walk the whole chain — a capture later edited by a generative tool
    # is a different story from either alone
    if _any_ai_assertion_in_chain(manifest):  # case 1
        r.short_circuit, r.confidence = True, 0.98
        return r

    if _hardware_capture(manifest) and _on_trust_list(manifest):
        r.prior = 0.15                        # case 2
    return r
```

`pip install c2pa-python pillow piexif`

### 3.4 T0b — weak metadata, features only

```python
{
  "exif_present": bool,
  "exif_software_suspicious": bool,   # 'stable diffusion','midjourney','dall'
  "exif_camera_make_present": bool,
  "png_text_has_prompt": bool,
  "png_text_has_model": bool,
  "metadata_stripped": bool,          # true for most social images
}
```

Six booleans into fusion. Never a gate.

---

## 4. T1 — Embedding

### 4.1 What an embedding is

CLIP maps a 224×224×3 image (~150k numbers) to **768 floats**. That vector is
what the probe trains on; it never sees pixels.

- 36k images ≈ 40GB. 36k embeddings ≈ **110MB**. Fits in RAM.
- CLIP runs **once per image, ever**. Training is then seconds, not hours.
- Frozen backbone ⇒ an embedding is valid forever.

This caching is not an optimisation, it is the schedule. It converts your team
from 3 experiments to 50.

### 4.2 Backbone

`ViT-L/14`, 768-dim, ~304M params. Documented default, comfortably under the 2B
cap, and shared across both probes.

Newer frozen encoders (Perception Encoder, MetaCLIP 2, DINOv3) now outperform it
as linear-probe backbones. Keep `embed.py` backbone-agnostic so swapping is one
line — "we compared three frozen backbones" is a strong Technical Execution
result. Check parameter counts before swapping.

**Never unfreeze.** LoRA fine-tuning a comparable encoder dropped in-the-wild
accuracy from 95.9% to 63.5% — manifold distortion and catastrophic forgetting.

### 4.3 Implementation

```python
# quorum/embed.py
import torch, open_clip, numpy as np
from pathlib import Path

CACHE = Path("data/cache/embeddings/vitl14_v1")   # versioned — see DATA_LAYOUT §7

class Embedder:
    def __init__(self, name="ViT-L-14", pretrained="openai", device="cuda"):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            name, pretrained=pretrained)
        self.model = self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.device = device

    @torch.no_grad()
    def embed_batch(self, pil_images):
        x = torch.stack([self.preprocess(i) for i in pil_images]).to(self.device)
        v = self.model.encode_image(x)
        v = v / v.norm(dim=-1, keepdim=True)      # L2 normalise — do not skip
        return v.cpu().numpy().astype(np.float32)

    def get(self, image_id, variant, loader):
        p = CACHE / f"{image_id}__{variant}.npy"
        if p.exists():
            return np.load(p)
        v = self.embed_batch([loader()])[0]
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, v)
        return v
```

**L2-normalise.** Without it, logistic regression partly fits vector magnitude,
which tracks image statistics rather than provenance.

Batch 64–128 on a 16GB GPU for ViT-L/14.

### 4.4 The augmentation trap

**Augmented images are different images and need their own embeddings.** You
cannot degrade a vector after the fact.

Plan variant coverage before the pass. See `DATA_LAYOUT.md` §8 for the
recommended split (all 15 variants on eval, 3 sampled per training image).

### 4.5 Optional: multi-crop

Cheapest meaningful upgrade. One downscaled 224px view discards high-frequency
detail — exactly where generator artifacts live. Embed 3–5 crops at **native
resolution** and mean-pool:

```python
crops = [center_crop_224(img)] + [random_crop_224(img) for _ in range(4)]
v = embedder.embed_batch(crops).mean(axis=0); v /= np.linalg.norm(v)
```

One hour, measurable gain, good ablation.

---

## 5. Face branch

### 5.1 Why alignment is the whole point

YuNet, MediaPipe and RetinaFace all return a bounding box **and 5 landmarks** — eye centres,
nose tip, mouth corners. The landmarks are the payload.

They let you warp every face into the same coordinate frame: left eye always at
the same pixel, right eye always at the same pixel, upright, same scale. That
turns "is the catchlight in one eye consistent with the other" from a hard vision
problem into a fixed-position comparison.

**This is why faces get a specialist and dogs do not.** You can canonically align
a face. There is no equivalent transform for "a golden retriever, possibly
sideways, possibly occluded."

### 5.2 Pipeline

```python
CANONICAL_5 = np.array([          # ArcFace 5-point template at 112, ×2 for 224
    [38.29, 51.69], [73.53, 51.50], [56.02, 71.74],
    [41.55, 92.37], [70.73, 92.20]], dtype=np.float32) * 2.0

def face_crop(img):
    dets = detector(img)
    if not dets:
        return None, False
    d = max(dets, key=lambda d: d.box_area)
    if min(d.box_w, d.box_h) < 64:            # size floor
        return None, False
    M = similarity_transform(d.landmarks_5, CANONICAL_5)
    return cv2.warpAffine(img, M, (224, 224)), True
```

- **Use OpenCV YuNet** (`cv2.FaceDetectorYN`): ships inside `opencv-python`, ~85KB
  model, CPU-fast, returns the 5 landmarks. MediaPipe has no Python 3.13 wheels —
  see `RUNBOOK.md` §10. Note YuNet's landmark order is right-eye-first while the
  ArcFace template is left-eye-first; swap rows 0/1 and 3/4 or faces come out
  mirrored.
- **Size floor 64px.** A 40px face upscaled to 224 is mostly interpolation, and
  the model will learn "blurry ⇒ fake."
- **Multiple faces:** take the largest, or pool with **max**. Never average — one
  fake face in a group photo is the signal.
- **Cache the crops.** They need their own embeddings; re-detection is slow.

---

## 6. Text branch

No training data needed.

```python
def text_features(img) -> tuple[np.ndarray, bool]:
    regions = ocr.detect(img)
    if not regions:
        return np.zeros(6, dtype=np.float32), False
    confs = np.array([r.confidence for r in regions])
    words = [r.text for r in regions]
    return np.array([
        confs.mean(),
        confs.std(),                       # variance is the signal
        dictionary_hit_rate(words),        # real signage uses real words
        mean_glyph_consistency(regions),   # same char, same shape?
        frac_nonascii(words),
        float(len(regions)),
    ], dtype=np.float32), True
```

PaddleOCR or EasyOCR (~10M params). Garbled signage is among the
highest-precision signals available and costs no training data.

---

## 7. Spectral branch

Generator-artifact detection, not scene regularity.

Transposed convolutions and upsampling leave periodic traces in the frequency
spectrum; generated images systematically fail to reproduce the high-frequency
Fourier modes of real photographs. Spectral analysis of a residual image is
well-established forensics and survives mild JPEG.

```python
def spectral_features(gray_f32) -> np.ndarray:
    residual = gray_f32 - cv2.medianBlur(gray_f32, 3)      # high-pass
    F = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(residual))))
    return np.concatenate([
        radial_band_energies(F, n_bands=5),
        [peak_to_median_ratio(F)],
        [grid_peak_strength(F)],            # upsampling grid signature
        [high_freq_rolloff_slope(F)],
    ]).astype(np.float32)
```

~30 lines with numpy and scipy. Zero parameters.

### 7.1 The catch — tell your team

**Compression eats this signal.** JPEG destroys high-frequency content, which is
exactly where the artifact lives. It survives mild compression; it will not
survive q=30 or a 0.25× downscale.

High-precision, low-recall. **Always feed `degradation_estimate` into fusion
alongside the spectral score**, so the meta-classifier can learn the difference
between "no artifacts found" and "couldn't measure."

### 7.2 Why this branch exists

CLIP compresses 150k numbers to 768 and discards high-frequency detail. Frozen
CLIP clusters real images well in an abstract space but does not inherently
distinguish real from generated; when artifacts are subtle it falls back on
semantic priors like object category, which overshadow the faint manipulation
signal.

**The spectral branch is the patch for that specific hole.** It operates on raw
pixels and sees exactly what the embedding threw away. That is the real argument
for the ensemble, and a better pitch line than "we ensembled for robustness":

> We chose a backbone that generalises across generators at the cost of low-level
> sensitivity, then recovered that sensitivity with cheap orthogonal features.

---

## 8. `degrade.py` — the official grid

Implement exactly the brief's transforms. Both training augmentation and the
evaluation grid come from this one module.

```python
TRANSFORMS = {
    "jpeg":   [90, 70, 50, 30],           # quality
    "blur":   [0.5, 1.0, 2.0],            # gaussian sigma
    "resize": [0.5, 0.25],                # scale down then back up
    "noise":  [0.02, 0.05, 0.10],         # gaussian sigma, on [0,1] pixels
    "jitter": [0.20],                     # brightness/contrast/sat ±20%
    "crop":   [0.80],                     # center crop fraction
}

def apply(img, kind: str, param) -> Image:
    ...

def variant_name(kind, param) -> str:
    return f"{kind}{param}".replace(".", "")   # 'jpeg70', 'blur05', 'resize025'
```

14 settings plus clean = 15 variants. Variant names are the cache keys in §4.3.

**Training:** sample randomly per image per epoch.
**Evaluation:** exhaustive grid — one row per setting. That table *is* the
required Robustness Evaluation Summary.

---

## 9. Assembling the fusion table

Each branch writes parquet keyed by `(image_id, variant)`:

```
data/cache/scores/general.parquet    image_id, variant, score
data/cache/scores/face.parquet       image_id, variant, score, present
data/cache/scores/text.parquet       image_id, variant, score, present
data/cache/scores/spectral.parquet   image_id, variant, score
data/cache/scores/meta.parquet       image_id, variant, provenance_*, exif_*
```

Fusion left-joins onto the manifest. Missing rows → fill `0.5`, presence `0`.

**This is why branches cannot own separate data.** The join is on `image_id` from
one shared manifest. Independent splits make the table unbuildable — and worse,
silently miscalibrated when one branch trained on another's calibration rows.

Enforcement: give every `train_*.py` a **required** `--manifest` argument with no
default. Hard to use the wrong data when there is no fallback.

---

## 10. Build order

| # | Task | Owner | Blocks |
|---|---|---|---|
| 1 | `degrade.py` + variant naming | Data/Eval | the streaming pass |
| 2 | `embed.py` + versioned cache | Detector | the streaming pass |
| 3 | `stream_embed.py` — SID_Set, then So-Fake-OOD | Detector | both probes |
| 4 | `predict.py` scaffold, stubbed | Fusion | **day 1, unblocked** |
| 5 | `build_manifest.py` + assertions | Data/Eval | training |
| 6 | Face crop pipeline | Fusion | face probe |
| 7 | Text + spectral features | Fusion | those scorers |
| 8 | `provenance.py` | Fusion | nothing — last |

**Steps 1 and 2 come before the data pass, not after.** Because the primary
datasets are streamed (`DATA_LAYOUT.md` §5.1), variants are generated inline
while each image is in memory. `degrade.py` and `embed.py` must exist before a
single image is fetched, or you will stream 19GB twice.

**Day-one parallelism.** Step 4 is the genuinely unblocked work, so the fusion
owner takes it: a `predict.py` that walks a directory and emits
`{image_path, pred}` with random scores. It proves the required deliverable early
and gives the frontend something real to render.
