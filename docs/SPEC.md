# Quorum — Robust AI-Generated Image Detection

Implementation spec, aligned to the official problem statement
*"Robust Detection of AI-Generated Images Under Real-World Transformations."*

Read fully before writing code. The ordering constraints matter more than any
individual component.

---

## 0. What this is

A detection system that decides whether an image is camera-captured or
machine-generated, and holds up after the image has been compressed, cropped,
resized, filtered, and reposted.

Core design claim: **no single model decides.** Several weak, independently
calibrated signals are fused by a learned meta-classifier, and the system reports
its own reliability alongside its verdict.

---

## 1. Competition constraints — read first

These come from the brief and override everything else.

| Constraint | Consequence |
|---|---|
| **All models < 2B parameters** | No large VLMs. Frozen CLIP + probes fits easily (§7) |
| **Image-level only; video/audio out of scope** | No frame sampling, no temporal aggregation |
| **COCO val2017 is the validation set** | **Never train on it.** Use train2017 or Open Images |
| **WildFake DALL·E Advanced subset is validation** | Exclude if training on WildFake |
| **Hackathon-scale compute assumed** | Frozen backbone is the right call, and it scores under Feasibility |
| **Required output: dir → JSON of `image_path` + `pred`** | Build `predict.py` to that exact shape (§8) |

**Judging weights:** Technical Execution 35%, Innovation & Problem Insight 20%,
Impact & Relevance 20%, Feasibility & Practicality 15%, Presentation 10%.

Note what is absent: **raw accuracy is not a judging criterion.** Well-structured
code, a reliable demo, sharp problem framing, and evident deliberate
decision-making outscore a better AUROC. Budget effort accordingly.

### 1.1 Open questions for the 28 Aug webinar

1. Is the 2B cap **per model** or **total across an ensemble**?
2. Are external APIs permitted at all, or must everything run locally?
3. Training on WildFake — how do we identify the excluded validation rows?

Answer 1 changes the architecture. Ask it first.

---

## 2. Non-negotiable invariants

Settled. Do not redesign mid-build; if you think one is wrong, say so rather than
working around it.

1. **The organizer validation set never enters training.** COCO val2017 and the
   WildFake DALL·E Advanced subset are eval-only. Assert this in the manifest.
2. **Unseen-generator evaluation is the headline.** So-Fake-OOD provides it.
3. **A held-out calibration split exists** that trains nothing. Fusion needs it.
4. **Every branch is calibrated before fusion.** Raw sigmoids are not
   probabilities and must not be averaged.
5. **Robustness augmentation is on by default.** Clean-only training is a debug
   mode, not a deliverable.
6. **Three branches maximum** — general, face, text — plus the spectral feature.
   No content-type specialists (§6.3).
7. **VLM is deferred.** Out on parameter grounds and non-essential. Revisit only
   if the webinar clears it and everything else is done.

---

## 3. Architecture

```
image
  │
  ├─► [T0] provenance (C2PA) ──────────► short-circuit if conclusive  [low priority]
  │
  ├─► [T1] CLIP ViT-L/14 embed (frozen, cached, SHARED)
  │         ├─► general probe
  │         └─► content-type zero-shot label
  │
  ├─► [T2] face branch      (aligned crop → same CLIP → probe)
  ├─► [T2] text branch      (OCR features → scorer)
  ├─► [T2] spectral branch  (high-pass FFT → scorer)
  │
  ├─► [T3] Platt calibration, per branch
  ├─► [T4] fusion meta-classifier (logistic regression)
  │
  └─► JSON verdict + reliability
```

**One CLIP instance serves both the general and face branches.** This matters for
the parameter budget (§7) and for inference cost.

---

## 4. Repo layout

```
quorum/
  data/                     # see DATA_LAYOUT.md
  quorum/
    provenance.py
    embed.py
    detectors/{general,face,text,spectral}.py
    degrade.py              # the official transform grid
    calibrate.py
    fusion.py
    pipeline.py
  scripts/
    build_manifest.py
    embed_dataset.py
    train_general.py
    train_fusion.py
    eval_grid.py
  predict.py                # REQUIRED DELIVERABLE — dir → JSON
  demo/
  README.md                 # REQUIRED DELIVERABLE — see §9
```

**Manifest-first.** Every dataset is normalised to a CSV manifest before any
model touches it. No training script walks a directory tree.

---

## 5. Build phases

### Phase 0 — Data
See `DATA_LAYOUT.md`. Nobody trains until the manifest assertions pass.

### Phase 1 — Thin slice
`bytes → JSON` end to end with stubbed branches. Proves integration while it is
cheap. Build `predict.py` here, not at the end.

### Phase 2 — General detector
Frozen CLIP ViT-L/14 + linear probe on cached embeddings. Embed once, cache to
`.npy`, then each training run takes seconds. Expect dozens of runs.

Record AUROC on So-Fake-OOD. That is the reference number for everything after.

**Never unfreeze.** LoRA fine-tuning a comparable frozen encoder dropped
in-the-wild accuracy from 95.9% to 63.5% — manifold distortion and catastrophic
forgetting. The 2B cap makes this moot anyway.

### Phase 3 — Robustness (highest value)

The brief hands you the exact grid. Implement precisely these, no more, no less:

| Transform | Parameters | Real-world analog |
|---|---|---|
| JPEG compression | q = 90, 70, 50, 30 | Social re-encode, messaging |
| Gaussian blur | σ = 0.5, 1.0, 2.0 | Out of focus |
| Resize | 0.5× / 0.25× then upscale | Thumbnail generation |
| Gaussian noise | σ = 0.02, 0.05, 0.10 | Low-light sensor noise |
| Color jitter | brightness/contrast/sat ±20% | Filter apps, auto-enhance |
| Center crop | 80% | Profile-picture cropping |

Used two ways:
- **Training augmentation** — randomly sampled per image per epoch
- **Evaluation grid** — one row per transform × parameter, clean vs transformed

The eval grid **is** the required Robustness Evaluation Summary. Build it as a
deliverable from the start, not a chart at the end.

Retrain Phase 2 with augmentation on. Expect clean to dip slightly and degraded
to rise a lot. That trade is correct — do not tune back toward clean.

### Phase 4 — Branches

**Face.** OpenCV YuNet (`cv2.FaceDetectorYN`) → 5-point align → 224px crop → same CLIP →
own probe. Fires only above a 64px face size.

*Why faces and not animals or objects:* alignment. You can warp every face to a
canonical frame so the probe learns positional features. No equivalent transform
exists for "a dog, possibly sideways, possibly occluded."

**Text.** PaddleOCR/EasyOCR → character confidence variance, dictionary hit
rate, glyph consistency. **No training data needed** — hand-built features.
Garbled signage is among the highest-precision signals available.

**Spectral.** High-pass residual → 2D FFT → radial band energies, peak-to-median,
grid-peak strength, roll-off slope. ~30 lines of numpy.

### 6.3 Deliberately NOT built

Recorded so it does not get reintroduced:

- **Content-type specialists** (animal/object/scene). Generator fingerprints look
  the same regardless of subject; six content models learn one signal six times.
- **Animal specialist.** Fur and feathers are dense stochastic texture — exactly
  where the general detector is already strongest.
- **Object specialist.** Not a coherent visual category. The real signal in it is
  periodic structure, captured by the spectral feature.

Content type still enters — as a **feature into fusion**, via CLIP zero-shot.

### Phase 5 — Calibration and fusion

1. **Platt scaling per branch** on the calibration split. Verify with reliability
   diagrams.
2. **CLIP zero-shot content label** (face / animal / object / scene / text-heavy)
   — free, reuses the embedding.
3. **Logistic regression** over:

```
[ cal_general, cal_face, face_present, cal_text, text_present,
  cal_spectral, content_onehot(5), degradation_estimate, provenance_prior ]
```

**Missing-branch rule:** presence flag + neutral 0.5 fill. "No face here" must
never read as "the face model says real."

**Gate:** the ensemble must beat the best single branch on So-Fake-OOD. If it
does not, calibration is wrong. Fix calibration; do not add models.

### Phase 6 — Error analysis (required deliverable)

Not an afterthought — it feeds Innovation & Problem Insight at 20%.

Assign an owner now. Produce:
- 3–5 representative **false positives** with a hypothesis for each
- 3–5 representative **false negatives**, same
- The stated trade-offs: robustness vs clean accuracy, generalisation vs
  in-distribution ceiling, false-positive cost at platform scale
- Per-content-bucket AUROC — if it varies wildly, you are partly reading
  semantics rather than artifacts, and saying so is a strength

---

## 7. Parameter budget

Under the 2B cap with room to spare. This is a slide.

| Component | Params |
|---|---|
| CLIP ViT-L/14 visual encoder (shared) | ~304M |
| OpenCV YuNet face detection | ~0.1M |
| PaddleOCR detection + recognition | ~10M |
| General probe | ~0.8M |
| Face probe | ~0.8M |
| Text scorer | <0.01M |
| Spectral scorer | <0.01M |
| Fusion meta-classifier | <0.01M |
| **Total** | **~316M** |

One CLIP instance serves both probes. If the webinar says the cap is per-model
rather than total, you have even more headroom — but do not spend it.

Fallback if you need smaller: CLIP ViT-B/16 is ~86M with a modest accuracy cost.

---

## 8. Required output contract

The brief specifies it. Build exactly this and do not let the richer demo schema
displace it.

```bash
python predict.py --input-dir path/to/images --output preds.json
```

```json
[
  {"image_path": "path/to/images/001.jpg", "pred": 0.87},
  {"image_path": "path/to/images/002.jpg", "pred": 0.03}
]
```

`pred` is P(AI-generated), calibrated, in [0,1].

The demo consumes a richer internal schema — signals, content type, reliability,
degradation estimate — but `predict.py` emits only the required two fields.
**This is the artifact judges will actually run.** Test it on a fresh clone.

---

## 9. Required deliverables checklist

- [ ] **Devpost description** — approach, tools, models, libraries, datasets
- [ ] **Public GitHub repo** — structured, commented
- [ ] **`predict.py`** — dir → JSON, exact shape above
- [ ] **README** — overview, setup, reproduction steps, limitations, per-member contributions
- [ ] **Demo video** — YouTube, public, end-to-end, **no third-party trademarks**
- [ ] **Robustness summary** — clean vs transformed table (= Phase 3 eval grid)
- [ ] **Error analysis note** — representative FPs, FNs, trade-offs

The README's "limitations and what we'd improve" section is easy marks — you
already have the material in §6.3 and the error analysis.

**Trademark caution:** the demo can be a feed-style slideshow UI, since slideshows
are image-level and in scope. Do not reproduce TikTok's logo or branding in it.

---

## 10. Metrics

- **TPR@1%FPR** on So-Fake-OOD — headline
- AUROC on So-Fake-OOD
- The robustness grid — clean vs each transform × parameter
- Per-content-bucket breakdown
- Single-branch vs ensemble comparison
- Organizer validation set (COCO val2017 + WildFake DALL·E) as a reference point

Frame false positives in moderation terms: at platform scale, 1% FPR means
millions of real creators wrongly flagged. The asymmetry between "missed a fake"
and "libelled a real creator" is the product-maturity argument, and it lands
under Impact & Relevance.

---

## 11. Known failure modes

| Failure | Symptom | Fix |
|---|---|---|
| **Validation-set contamination** | Suspiciously strong organizer-set numbers | Assert COCO val2017 / WildFake-DALL·E excluded |
| Disk exhaustion day 1 | 500GB download stalls the team | `DATA_LAYOUT.md` budget; GenImage optional |
| Format predicts label | ~perfect accuracy, collapses in the wild | Re-encode all to JPEG q95 before hashing |
| Uncalibrated fusion | Ensemble worse than best branch | Platt scaling, reliability diagrams |
| Phantom branch votes | Landscapes systematically misjudged | Presence flags + 0.5 fill |
| Clean-data overfit | Great clean, dies on screenshots | Augmentation on by default |
| Semantic shortcut | AUROC varies wildly by content type | Report it; lean on spectral + text branches |

---

## 12. Cut order under time pressure

Cut: **spectral → text branch → face branch → provenance.**

**Never cut:** the validation-set exclusion assertions, the robustness grid, or
calibration. And never cut `predict.py`, the README, or the error analysis —
those are graded deliverables, not nice-to-haves.

A single well-calibrated, robustness-trained CLIP probe with a real evaluation
story and clean code beats an uncalibrated six-model ensemble. The judging
weights say so explicitly.

---

## 13. Reference

- SIDA / SID_Set — `https://arxiv.org/pdf/2412.04292` · `https://github.com/hzlsaber/SIDA`
- So-Fake — `https://arxiv.org/abs/2505.18660`
- Community Forensics (cross-generator generalisation) — `https://arxiv.org/abs/2411.04125`
- NTIRE 2026 robust detection in the wild — `https://arxiv.org/pdf/2604.11487`
- RAID adversarial robustness — `https://huggingface.co/datasets/aimagelab/RAID`

Worth carrying into the pitch: the Community Forensics authors explicitly state
they do not intend their dataset to train detectors deployed in the wild, because
errors mean falsely accusing someone of faking an image, or certifying
misinformation as real. That caution is the right register for the whole project
and it lands under Impact & Relevance.
