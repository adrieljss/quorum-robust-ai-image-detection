# Quorum — Robust Detection of AI-Generated Images

**TikTok TechJam 2026 · Topic 5** · MIT licensed ([weights are non-commercial](#licences))

Decides whether an image was captured by a camera or produced by a machine, and
keeps deciding correctly after it has been compressed, blurred, resized, noised,
cropped and colour-shifted the way a real upload is.

One frozen CLIP backbone, three linear probes totalling **2,310 trained
parameters (11 KB on disk)**, combined by `max` rather than a learned
meta-classifier — because we built the meta-classifier, measured it, and it lost.

```bash
python predict.py --input-dir path/to/images --output preds.json
# [{"image_path": "path/to/images/001.jpg", "pred": 0.87}]
```

---

## 1. The problem

The brief asks for detection that survives real-world transformations. Building
it surfaced a second axis the brief does not name, and the two together shape the
architecture.

**"Robust" means surviving the platform.** A detector reading high-frequency
generator artifacts dies at JPEG q30. Measured on released weights: FatFormer
(CVPR'24) scores **0.646 clean, 0.252 under blur** on our organizer benchmark —
*inverted*, calling blurred DALL·E images real more often than COCO photographs.

**"AI-generated" is two questions.** A fully synthetic image and a real
photograph with an AI-inpainted patch are opposite problems. Our general probe
scores **AUROC 0.37 on locally-edited photographs** — worse than chance, and
correctly so: an edited photo *is* globally authentic. One model asked both
questions is forced into a single additive trade-off between them.

## 2. How it addresses that

Every image is embedded once by a **frozen CLIP ViT-L/14-quickgelu** (OpenAI
weights, 304M params, fp16, 768-d L2-normalised). Nothing is fine-tuned. Three
linear probes read it:

| branch | input | params | question |
|---|---|---|---|
| `general` | 768-d image embedding | 769 | fully synthetic? |
| `tampered` | 768-d image embedding | 769 | contains an AI-edited region? |
| `face` | 769-d = CLIP(face crop) + standardised `log₂(face_px)` | 772 | is any face synthetic? |

```
pred = max( σ(max(z_general, 1.25·z_tampered) − SHIFT),  max_i p_face,i )
```

- **The combiner is a disjunction, and that was measured.** `quorum/fusion.py` is
  the learned alternative — logistic regression over 14 calibrated inputs. It
  scores **0.8511 against `max`'s 0.8597**, and reaches parity on the headline set
  only by collapsing into the general probe, at which point it can match the
  headline *or* detect tampering, never both. The task is disjunctive; a linear
  model in log-odds cannot express "or". Six branches have lost their place in
  `max()` on measurement — all six shared the 768-d features. The one that won,
  `face`, brought different ones.
- **0.5 is an operating point, not a sigmoid default.** The score is shifted so
  the cut lands on `OPERATING_POINT = 0.8092`, picked on a
  generator-family-disjoint carve. Monotone, so AUROC is bit-identical; it buys
  precision 0.766 → 0.902 and cuts false accusations ~3×, at a stated cost in
  recall.
- **Faces are scored plurally.** A YuNet ONNX detector finds every face ≥ 64 px
  (capped at 8) and each is scored — one synthetic face is enough.
- **Robustness is a grid, not a claim.** 15 settings: clean + JPEG (90/70/50/30),
  blur (σ 0.5/1/2), resize (0.5×/0.25×), noise (0.02/0.05/0.1), jitter, 80% crop.
  Training sources see clean + 3 *randomly sampled* settings, seeded on the
  image's content hash. All 196 composed pairs were measured separately: ~0.013
  AUROC, and degradation **does not compound**.
- **Provenance is read but never scored.** `quorum/provenance.py` parses C2PA
  (JUMBF/`caBX`/RIFF), EXIF, XMP and PNG text from the *original bytes*. All 7
  GPT-image-2 test files carry a signed `trainedAlgorithmicMedia` manifest and
  the pixel model misses 4 of them — yet it stays out of `pred`: unmeasurable on
  our benchmarks (null for 100% of eval rows), ~0 recall after platform
  processing, and trivially forged. The demo shows it as an unvalidated claim.

## 3. Results

| eval set | n | AUROC | ACC | PREC | RECALL | F1 |
|---|---|---|---|---|---|---|
| **So-Fake-OOD clean** *(headline)* | 4,198 | **0.9265** | 0.8380 | 0.9021 | 0.7588 | 0.8243 |
| So-Fake-OOD, all 15 variants | 62,970 | 0.9206 | 0.8221 | 0.9076 | 0.7177 | 0.8016 |
| Organizer set, clean † | 8,719 | 0.9722 | 0.9185 | 0.8872 | 0.9266 | 0.9065 |
| Organizer set, all 15 † | 130,785 | 0.9594 | 0.8989 | 0.8675 | 0.9005 | 0.8837 |
| SID_Set tampered, clean | 3,595 | 0.9120 | 0.8473 | 0.8665 | 0.7492 | 0.8036 |
| **Foreign tampered, clean** ‡ | 5,096 | **0.7260** | 0.5608 | 0.8464 | 0.3103 | 0.4541 |

† One generator (DALL·E 3) against one photo corpus, and the brief excludes it
from scoring. **Quote 0.9265, not 0.9722.** ‡ The tampered branch's honest
cross-corpus number: 0.91 same-dataset, 0.7260 on a foreign corpus's edits.

**Per branch, clean → worst of 15** ([full grid](docs/robustness.md)):
`general` 0.9245 → 0.9013 · `face` 0.9421 → 0.9168 · `tampered` 0.9528 → 0.8962 ·
`spectral` 0.6736 → 0.5471 *(demo display only)*.

**Against published detectors**, downloaded at released weights and run here on
identical pixels — 600 COCO val2017 reals vs 600 WildFake DALL·E fakes, in
neither training set ([details](docs/BASELINES.md)):

| detector | clean | jpeg30 | blur20 | resize025 | noise005 | mean | drop |
|---|---|---|---|---|---|---|---|
| **Quorum** | **0.977** | 0.983 | 0.953 | 0.974 | 0.947 | **0.967** | **0.030** |
| FatFormer (CVPR'24) | 0.646 | 0.883 | 0.252 | 0.302 | 0.515 | 0.520 | 0.394 |
| CNNDetection (CVPR'20, crop) | 0.586 | 0.527 | 0.505 | 0.506 | 0.521 | 0.529 | 0.081 |
| CNNDetection (resize) | 0.390 | 0.378 | 0.324 | 0.314 | 0.451 | 0.371 | 0.076 |

**Cost.** 769 trained parameters on a backbone you already run: where CLIP
embeddings exist a verdict costs **0.32 µs** and a 3.5 KB file, a new branch
costs zero extra inference, and a new generator costs seconds rather than
GPU-hours. The honest half — we do run a 304M ViT on every image, and the frozen
backbone is our largest limitation.

## 4. What didn't work

Sixteen negative results are in [`docs/ERROR_ANALYSIS.md`](docs/ERROR_ANALYSIS.md) §8,
because they are the evidence the design is a decision rather than a default.

| attempt | result |
|---|---|
| OCR features for garbled text | **0.4627 — below chance**; 5 of 6 features flip sign across datasets, falsifying our own design doc |
| CLIP on warped text crops | Works (0.8083 transfer) but worth **+0.0022**, and misses the gap it was built for |
| More real-photo diversity for `tampered` | False positives got **worse**, 13.6% → 53.5%, while its AUROC *rose*. It fits a construction artifact; data is not the lever |
| Learned fusion meta-classifier | 0.8511 vs `max`'s 0.8597 |
| 2026-generator data as a 6th branch | −0.0211, and it would have *spent* an unseen generator we evaluate on |

## 5. Development tools

**Python 3.13.1** + `venv` (no conda, no notebooks) · **VS Code** ·
**Claude Code** (co-author on 29 of 70 commits) · **Git + GitHub**, PR-based ·
**Ruff** for lint, enforced by the self-check · **Docker** for the demo image ·
**Hugging Face Hub** for dataset streaming and a private 2.01 GB embedding cache ·
**Hugging Face Spaces** for demo hosting · **diagrams.net**, with `.drawio` files
generated by scripts so they cannot drift from the code · one **RTX 4060 Laptop
8 GB** for the embedding passes; the demo runs CPU-only.

## 6. Models and APIs

**No external inference APIs.** Everything runs locally; the only network calls
are dataset streaming at build time and a one-time CLIP weight download.

| model | role | frozen? |
|---|---|---|
| **CLIP ViT-L/14-quickgelu** (`openai`, via `open_clip_torch`) | the only feature extractor, 304M params | frozen |
| **YuNet** (`yunet.onnx`, 227 KB, OpenCV Zoo) | face detection + alignment | frozen |
| **RapidOCR PP-OCRv4** | text location, demo signal only | frozen |
| `general` / `tampered` / `face` `.npz` | the shipped scorer — 769 / 769 / 772 params | **ours** |
| `spectral` (9) · `text_crop` (772) `.npz` | demo display signals, never in `pred` | **ours** |
| `content_prompts.npz` | CLIP zero-shot content buckets for per-class error analysis | prompts |

`general` is a RidgeClassifier with Platt scaling folded into its saved weights.
Baselines run for comparison but not shipped: **CNNDetection** (Wang et al.,
CVPR 2020) and **FatFormer** (Liu et al., CVPR 2024).

## 7. Libraries and frameworks

**Model / data** — `torch` ≥2.6 · `open_clip_torch` · `numpy` · `pandas` ·
`pyarrow` · `scikit-learn` · `Pillow` · `opencv-python` · `datasets` ·
`huggingface_hub` · `tqdm` · `matplotlib`

**Demo** — `Flask` 3 · `gunicorn` · `opencv-python-headless` ·
`rapidocr_onnxruntime` (optional; without it the demo reports
`signals.text: null` rather than failing) · vanilla HTML/CSS/JS, no build step

**Tooling** — `ruff` (lint + format, one binary)

Deliberately absent: no fine-tuning framework, no experiment tracker, no training
loop. Every trained artifact here is a linear fit that takes seconds on CPU.

## 8. Datasets

**487,636 manifest rows · 51,905 unique images · 10 sources · 15 variants each.**
~57 GB of source imagery is read **once** and reduced to **2.01 GB of
embeddings**, so the whole project fits on a laptop.

| dataset | role | n (clean) |
|---|---|---|
| **SID_Set** | primary training — real vs fully-synthetic; also face crops and spectral features | 16,000 |
| **SID_Set tampered** | trains `tampered` — AI-inpainted real photographs | 3,949 |
| **WildFake — Midjourney** | generator diversity for `general` | 1,500 |
| **So-Fake-OOD** `calib_ood` carve | calibration + a 4-of-5-family training rotation | 2,044 |
| COCO train2017 reals | tried as extra negatives, **reverted** — it worsened false positives | 5,000 |
| 🏆 **So-Fake-OOD** `test_ood` | **the headline** — 10 generator families absent from training | 4,198 |
| 🔒 **Organizer validation** (COCO val2017 + WildFake DALL·E) | the brief's benchmark, **quarantined by the problem statement** | 8,719 |
| **SID_Set tampered** eval | edited photos, same corpus as training | 1,499 |
| **So-Fake tampered** eval | edited photos, **foreign corpus** — the honest number | 3,000 |
| **LAION-5B real holdout** | false positives on an unfamiliar *corpus*, not just unfamiliar images | 2,000 |

`scripts/build_manifest.py` **asserts** that no image and no generator family
crosses the train/eval line. The calibration carve splits by generator *family*,
not generator — Ideogram2 and Ideogram3 on opposite sides would let us call a
sibling model "unseen". Layout: [`docs/DATA_LAYOUT.md`](docs/DATA_LAYOUT.md).

**Assets:** `data/models/*.npz` (7 files, ~34 KB, tracked) · `yunet.onnx`
(227 KB) · `test-images/` for the self-check. No corpus imagery is in this repo.

## Licences

Code is **MIT** ([`LICENSE`](LICENSE)). That grant cannot relicense the data, so:

| asset | licence | consequence |
|---|---|---|
| SID_Set | CC BY 4.0 | attribution |
| **So-Fake-OOD** | **CC BY-NC 4.0** | **`general.npz` is trained partly on its `calib_ood` carve, so the shipped weights are NON-COMMERCIAL** |
| WildFake | none stated on the ModelScope mirror used | treated as research use only |
| COCO 2017 | images under original Flickr terms; annotations CC BY 4.0 | attribution |
| LAION-5B | CC BY 4.0 metadata; images linked, not owned | — |
| CLIP ViT-L/14, open_clip | MIT | — |
| YuNet (OpenCV Zoo) | MIT | — |
| RapidOCR / PP-OCRv4 | Apache 2.0 | — |

`tampered.npz`, `face.npz`, `spectral.npz` and `text_crop.npz` train on SID_Set
only, so they carry attribution rather than a non-commercial restriction. The
embedding cache is a **private** HF repo and is never redistributed;
`.dockerignore` is an allowlist enforcing that as a build assertion. No TikTok
branding appears anywhere.

## 9. The demo

Flask, single origin, deployed as a Docker Space on HF's free CPU tier. Upload an
image and it returns the verdict plus the reasoning: per-branch scores on the
shipped scale, the face box, a CLIP zero-shot content label, a degradation
estimate, the spectral and text display signals, and any C2PA/EXIF provenance
from the original bytes. Two deliberate behaviours:

- **A branch that cannot measure an image abstains** rather than guessing "real".
  No face in frame means `face: null`, never `face: 0`.
- **An earned uncertainty band.** Scores in 0.40–0.60 are reported as uncertain;
  measured accuracy inside that band is **0.5229** against **0.8726** outside.

Deployment, and three Dockerfile bugs that all passed a green build:
[`docs/DEPLOY.md`](docs/DEPLOY.md).

## 10. Limitations

1. **Failure tracks generator *recency*, not family.** Every second-generation
   model beats its predecessor: GPT-image-2 **70.2% FNR** vs GPT-4o 15.9%;
   nano_banana_2 59.4% vs nano_banana 14.0%. A frozen 2023 backbone has never
   seen a 2026 generator. It is a distribution problem, not a backbone one — an
   in-distribution probe on those families reaches 0.84–0.97.
2. **False accusations on genuinely unfamiliar photography are 19.50%**, not the
   8.25% the operating point is anchored to. Both are reported; only the anchor
   is tuned against.
3. **`tampered` fits a dataset, not a concept** — 0.91 same-corpus, 0.7260
   cross-corpus. Kept anyway, because dropping it takes edited-photo AUROC to
   0.5286, a coin flip.
4. **No explainability beyond per-branch scores.** Patch-level scoring would give
   a heat map; designed, unbuilt.
5. **We know how to close part of the gap and chose not to.** Training on the
   held-out families lifts the four worst generators **+0.0559** and *lowers*
   false positives — but retires the unseen-generator evaluation that gives every
   number here its force.

## 11. Team

**Adriel Jansen Siahaya** — data pipeline, embedding, manifest, evaluation, error
analysis · **Albert Ariel Putra** — general probe, spectral · **Kacey Isaiah
Yonathan** — face probe, text, calibration and fusion · **Michael Cenreng** and
**Valentino Nathan** — demo backend and frontend.

## 12. Documentation

| | |
|---|---|
| [`ERROR_ANALYSIS.md`](docs/ERROR_ANALYSIS.md) | required deliverable — scorecard, failure by transform / generator / content, case studies, four measured trade-offs, sixteen negative results |
| [`robustness.md`](docs/robustness.md) | required deliverable — AUROC per branch under all 15 settings |
| [`BASELINES.md`](docs/BASELINES.md) | Quorum vs CNNDetection and FatFormer, run here |
| [`SPEC.md`](docs/SPEC.md) · [`PIPELINE.md`](docs/PIPELINE.md) · [`DATA_LAYOUT.md`](docs/DATA_LAYOUT.md) | architecture, model input contracts, data |
| [`HANDOVER.md`](docs/HANDOVER.md) · [`HANDOVER-MODELS.md`](docs/HANDOVER-MODELS.md) | the working record, including what failed |
| [`DEPLOY.md`](docs/DEPLOY.md) · [`RUNBOOK.md`](docs/RUNBOOK.md) | deploying the demo; running the data pass |
| `docs/figures/*.drawio` | architecture, system and dataset diagrams, generated from code |

---

# Development

```bash
git clone https://github.com/adrieljss/robust-ai-image-detection && cd robust-ai-image-detection
python -m venv .venv && .venv\Scripts\activate      # source .venv/bin/activate on mac/linux
pip install torch --index-url https://download.pytorch.org/whl/cu130   # NVIDIA GPU: FIRST
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"             # must print True
hf auth login
export QUORUM_CACHE_REPO=techjam2026blueberryjam/quorum-cache   # $env:... on PowerShell
python scripts/pull_cache.py          # ~2GB of embeddings, no images
```

```python
from quorum.detectors.general import load
X, rows = load("sid_train")     # X (N,768) aligns 1:1 with rows
```

Use `load()`, never `np.load` or `load_source` — it drops re-embedded duplicates
and cross-split leaks, and takes `split` from the manifest rather than the shard
([`HANDOVER.md`](docs/HANDOVER.md) §1).

**Rules.** `train` trains; `calib` and `calib_ood` fit calibrators and fusion
only; `test_ood` and `test_organizer` are never fitted on, ever. `calib_ood` is
carved out of So-Fake-OOD by generator family, so selecting rows by
`source == "so_fake_ood"` without filtering `split` trains on your own eval set —
filter by split, always. Every `train_*.py` requires `--manifest`, no default.
Never train on `data/raw/organizer_val/`. Do not re-run the embedding pass — it
changes everyone's numbers.

## Tests

```bash
python scripts/selfcheck.py         # offline, ~30s -- runs on a fresh clone
python scripts/selfcheck.py --all   # + the checks that read data/cache, ~90s
```

One command, one exit code. The offline set needs no cache and no GPU: the probes
it scores with are tracked, the 2 GB cache is not.

| check | what fails if it breaks |
|---|---|
| `predict.py --self-check` | the operating point stops landing on 0.5; the shift stops being monotone; `max` quietly becomes something else; the output contract (two fields, posix paths, five extensions, nested dirs) drifts; a private copy of the scoring path in `scripts/` diverges from `score_embeddings` |
| `quorum.degrade` | the 14-setting grid, or its per-image seeding |
| `quorum.embed` | image ids stop being format-stable; shard order is lost |
| `quorum.calibrate` | Platt stops improving ECE; calib_a/calib_b stops being a partition |
| `quorum.features` | spectral features go non-finite; a face is "found" in pure noise |
| `quorum.provenance` | a container parser breaks, or `normalise()` stops destroying the signal |
| `chain_eval` | the metric helpers, on populations where AUROC is undefined |
| `app/app.py --self-check` | `/api/analyze` stops rejecting bad uploads, or its response shape drifts |
| `build_manifest.py` *(--all)* | assertions A–F: organizer val leaks, the OOD carve overlaps, an image spans two splits |
| `general --check` *(--all)* | an image sits on both sides of the train/eval line |
| `spectral` *(--all)* | the calibration carve leaks back into the reported number |

`selfcheck.py` deliberately skips `quorum.fusion`, `quorum.detectors.face` and
bare `quorum.detectors.general`: those retrain and overwrite the shipped weights.
They are training entry points that happen to assert, not tests. `general
--check` is the assertion half, split out. `quorum.detectors.text` is skipped
unless `rapidocr_onnxruntime` is installed.

## Lint

Ruff, config in `pyproject.toml`. `ruff check` is enforced by `selfcheck.py`;
`ruff format` is available but **not** enforced.

```bash
python -m ruff check .            # runs in selfcheck.py
python -m ruff format --diff .    # what the formatter would change
```

The rule set is four families wide (`F`, `E9`, `E741`, `W`), deliberately: a
default set reports 181 findings here of which 4 were real, the rest being
hand-aligned comment columns and `sys.path.insert` before imports, both on
purpose. The formatter is opt-in for the same reason — a Black-style pass
rewrites 2,161 lines across 26 files, mostly collapsing alignment like
`quorum/degrade.py`'s `TRANSFORMS` table into ragged single spaces.
`pyproject.toml` records what was left out and why.

## Inspecting single images

```bash
python scripts/try_face.py photo.jpg other.png --save-crops out/   # face + general probes
python scripts/try_grid.py photo.jpg [--chain]                     # 15 variants, or 196 pairs
python predict.py --input-dir test-images --output preds.json --provenance
```

`try_face.py` loads models in ~13 s then ~50 ms an image — pass them all in one
invocation.
