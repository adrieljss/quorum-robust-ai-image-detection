# Quorum — Data Handover

**Status:** all three cached branches complete on every source, `organizer_val`
included. Face probe, calibration, and fusion all built (Kacey). Text branch cut.
`predict.py` ships `max(general, tampered)` on measurement — see §5e.
**Data owner:** Adriel — ask about anything in the cache or the manifest.
**Manifest:** `data/manifests/main.csv` — 330,851 rows, assertions A–F pass.
**Reply doc:** `docs/HANDOVER-MODELS.md` is Kacey's handover back; §8 there is the
single sharpest result anyone has produced and is worth reading before you model.

---

## 0. Read this first

Four rules. Breaking any of them invalidates every number the team produces.

1. **Never train on `test_organizer`.** COCO val2017 + WildFake DALL·E Advanced
   are the organizer's validation set. Assertion A enforces the split; nothing
   enforces a wrong label, so check it by hand.
2. **Never train on `test_ood`.** So-Fake-OOD is the headline eval. It is the
   only number that measures generalisation to unseen generators.
3. **`calib_ood` is the one deliberate exception, and it is not a licence to
   touch `test_ood`.** Part of So-Fake-OOD is now carved off as a calibration
   slice (§5d). Calibrators and fusion fit on `calib_ood`; nothing ever fits on
   `test_ood`. The two are disjoint by generator family *and* by image, and
   `scripts/build_manifest.py` asserts both. If you select rows by
   `source == "so_fake_ood"` without also filtering `split`, you are training on
   your own eval set — filter by split, always.
4. **Every `train_*.py` takes a required `--manifest` with no default.** One
   shared manifest, or the fusion join silently miscalibrates.

Contamination does not look like a bug. It looks like an unusually good number.

---

## 1. Getting the data

```powershell
$env:QUORUM_CACHE_REPO = "techjam2026blueberryjam/quorum-cache"
python scripts/pull_cache.py          # ~1.3GB, vectors + manifests, no images
```

You never need the raw images. 60GB of pixels became 1.3GB of vectors; that is
the entire point of the cache.

```python
from quorum.detectors.general import load
X, R = load("sid_train")    # X[i] is the 768-d vector for row R.iloc[i]
```

`load()` does three things `np.load` does not, and all three are load-bearing:

1. drops re-embedded duplicates within a source (a restarted run re-draws the
   same shuffle order, so one image can land in two shards);
2. drops **cross-split leaks** — an image the manifest assigned to a different
   source, e.g. the 43 images SID_Set ships in both its tampered train and
   validation splits;
3. **takes `split` from the manifest, not from the shard.** The shard records
   what the embedding pass was told; the manifest records what the split
   resolved to after dedupe and the `calib_ood` carve. Never read `R.split`
   expecting the shard's value.

Use `load()`, never `np.load` directly.

---

## 2. What is in the cache

| source | split | images | general | spectral | face |
|---|---|---|---|---|---|
| `sid_train` | train | 16,000 | yes | 15,753 | 4,059 |
| `sid_tampered` | train | 3,949 | yes | 3,992 | 472 |
| `sid_calib` | calib | 3,996 | yes | 3,996 | 964 |
| `so_fake_ood` | **test_ood + calib_ood** | 6,242 | yes | 5,987 | 1,910 |
| `sid_tampered_eval` | test_ood | 1,499 | yes | 1,499 | 244 |
| `organizer_val` | test_organizer | 5,000 | yes | 5,000 | 386 |

Face coverage is ~25% by nature: most images contain no detectable face. That is
not missing data, it is the `face_present` flag doing its job.

**`organizer_val` is the exception: 386 of 5,000 (7.7%).** COCO val2017 is
object-centric and faces in scene photos usually fall under the 64px minimum.
On the organizer benchmark the face branch abstains on 92% of images, so fusion
must lean on general+spectral there. Do not read that as the face probe failing.

**Variants.** 15 per image (14 degradation settings + clean). Train images carry
clean + 3 sampled; eval images carry the full grid. Names are the join key:
`clean`, `jpeg90`, `blur05`, `resize025`, `noise01`, `jitter02`, `crop08`, and so on.

---

## 3. Model input contracts — frozen

| branch | input | output | model |
|---|---|---|---|
| general | `float32[768]` L2-normed CLIP | raw score | LogisticRegression |
| face | `float32[769]` — aligned 224px crop + standardised `log2(face_px)` | score + `face_present` | LogisticRegression |
| spectral | `float32[8]` | score | LogisticRegression |
| text | `float32[6]` | score + `text_present` | LogisticRegression |

Branch scores are **uncalibrated**. Calibration happens once, in fusion: the
Platt `(a, b)` pairs live in `data/models/fusion.npz` as `cal_general`,
`cal_face`, `cal_spectral`, `cal_tampered`, fitted on `calib_ood` (§5d).
The face branch is **769**-d, not 768 — the extra column is Kacey's `face_px`
conditioning and it is worth +0.043 clean. `quorum.detectors.face.design()`
builds it; do not hand-roll the concatenation.

**Missing-branch rule:** presence flag + neutral `0.5` fill. "No face here" must
never read as "the face model says real."

---

## 4. Albert — General probe + Regularity (spectral)

### 4a. General probe — the headline model

Baseline is trained and committed: `data/models/general.npz`.

```
So-Fake-OOD (test_ood, held out)   clean 0.9170   worst 0.8848 (noise002)   drop 0.0321
```

Numbers moved slightly from the first handover: the eval is now the held-out
`test_ood` only, with `calib_ood` excluded, and `load()` drops the 43-image
SID_Set leak. Regenerate any table with `python scripts/eval_grid.py`.

Your job is to beat it. What is already known:

- **Degradation augmentation works.** Clean-only training gives 0.9065 / 0.8488.
  Adding 3 sampled variants gives 0.9124 / 0.8798 — worst-case collapse cut 43%
  *and* clean accuracy improved. Do not train clean-only.
- **`sid_calib` is saturated at 0.9996 and useless for decisions.** Same
  generators as train. Make every call on So-Fake-OOD.
- **Do not fold tampered into this probe.** Measured: it drops OOD to 0.7920
  while only reaching 0.7831 on tampered, where a dedicated probe reaches 0.9521.
  One linear boundary cannot serve both tasks. `tampered.npz` stays separate.
- **Do not bother with an MLP head — that question is now answered.** Kacey ran
  it on the face branch (`HANDOVER-MODELS.md` §9): linear 0.9382, MLP(64,)
  0.9282, MLP(256,) 0.9230, MLP(256,64) 0.9315. Every model is ~0.999
  in-distribution and they differ *only* in how far they fall on unseen
  generators. Extra capacity buys a better fit to the shortcut, nothing else.
  Retest on general if you like, but expect the same shape.
- Multi-crop embedding (PIPELINE §4.5) is the cheapest untried upgrade, and with
  the MLP question closed it is now the *only* untried one on this branch.
- A different frozen backbone is the real headroom. ViT-L/14 is ~304M against a
  2B cap — you are using 15% of the budget.

### 4b. Regularity / spectral scorer

**This is an hour of work, not a day.** The features already exist; the model is
9 parameters and trains in 0.73 seconds.

```python
from quorum.detectors.general import load, fit, auc_by_variant
X, R = load("spec_sid_train")      # (N, 8)
clf = fit(X, R.label.values)
```

**Measured on held-out `test_ood`, which is the number that counts:**

```
clean 0.6736   worst 0.5471 (noise01)   drop 0.1265
```

The older `sid_calib` figures (clean 0.8347) were same-generator and flattered
it by ~0.16. Use So-Fake-OOD for every decision.

```
clean 0.8347   jpeg90 0.8313   blur05 0.7953   jpeg30 0.7576   <- sid_calib, optimistic
blur20 0.5713  resize05 0.5177  resize025 0.5124               <- chance
```

**Known data defect, your call where the guard goes:** `spec_so_fake_ood` holds
25 all-zero feature vectors (3 distinct images, ~8 variants each, all label 0).
A zero vector is not a missing branch — it flows through as a real score instead
of tripping the presence-flag path, so those images get neither a signal nor an
abstention. The other four spectral sources are clean.

**It collapses to coin-flip under downscaling.** Expected and documented
(PIPELINE §7.1) — JPEG and resize destroy the high frequencies the branch reads.
Do not try to fix it. It is high-precision, low-recall by design.

Its value is that it fails on *different* inputs than CLIP does. Report the
per-variant table; that complementarity is the real argument for the ensemble.

**Deliverable:** `quorum/detectors/spectral.py` plus the per-variant AUC table.

---

## 5. Kacey — Face probe, Text, and Fusion

### 5a. Face probe

```python
X, R = load("face_sid_train")      # (N, 768); rows carry face_present, face_px
```

Preprocessing is done — YuNet detection, 5-landmark Umeyama alignment to the
ArcFace template, 224px crops, all cached. You train the probe only.

Two measured facts that will shape your model:

- **`face_px` matters — this shipped and it is the branch's one clear win.**
  Box sizes run **64–612px** (an earlier 64–181 here was wrong), so the same
  nominal degradation lands far harder on a small face upscaled to 224 than on a
  large one downscaled to it. `face.py` conditions on standardised
  `log2(face_px)`: 0.8952 → **0.9382** clean, 0.8620 → **0.9151** worst.
- **More face data will not help.** Learning curve on `face_sid_train`, eval
  on `face_so_fake_ood`: 500 imgs -> 0.8892 clean / 0.8407 worst; all 4,128 ->
  0.8952 / 0.8620. Flat. A 769-parameter probe cannot absorb more. Spend the
  time on conditioning and fusion, not collection.
- **The detector dies under heavy degradation** — `noise010` loses 77% of faces,
  `resize025` and `noise005` lose 15%. Those images have no face row at all.
  That is honest system behaviour, not a bug: at inference you do not have a
  clean copy to detect on. Report coverage per variant alongside AUC.

**Deliverable:** `quorum/detectors/face.py` plus per-variant AUC *and* coverage.

### 5b. Text scorer — build or cut

Not built. No OCR library installed. Extraction is ~9 hours over the full grid
for a 7-parameter model, and it is the weakest of the four signals.

**Recommendation: cut it** and spend the time on fusion. If the team wants it,
run OCR on `clean` only (~45 min) and let fusion fill the variants.

### 5c. Fusion — the real work

This is where the branches become a system, and it is the critical path —
Albert's probes, the eval grid, and the demo all wait on it. If your schedule
slips, cut text (5b) to protect this.

```
input:  [ cal_general, cal_tampered, cal_face, face_present,
          cal_spectral, cal_text, text_present,
          content_onehot(5), degradation_estimate, provenance_prior ]
output: calibrated P(AI)
```

The input list has **five** branches, not four — `tampered` is its own probe.
See §4a.

- Fit calibration on `calib` **only**. Never on train, never on test.
- Left-join branch scores onto `main.csv` by `(image_id, variant)`.
- Missing rows become `0.5`, presence flag `0`.
- `predict.py` currently fakes this with `max(P_general, P_tampered)` and is
  marked `ponytail:` in the source. Replacing it is your deliverable.

**Deliverable:** `quorum/fusion.py`, `quorum/calibrate.py`, and `predict.py`
wired to them.

---

## 5d. `calib_ood` — the generator-disjoint calibration slice

Built in response to `HANDOVER-MODELS.md` §6. **Use this, not `sid_calib`, for
anything that produces a probability.**

`sid_calib` shares generators with `train`, so every branch scores ~0.999 on it.
Platt fitted against a branch that cannot be wrong learns an extreme slope, and
that slope manufactures over-confidence the moment a new generator appears.
Measured ECE on unseen generators:

```
branch    AUC on cal set   ECE (sid_calib)   ECE (calib_ood)   factor
general           0.9996            0.1026            0.0217     4.7x
face              0.9976            0.1665            0.0333     5.0x
spectral          0.6789            0.0774            0.0519     1.5x   <- control
```

The two saturated branches improve ~5x. The one that never aced its calibration
set barely moves — which is what confirms the mechanism rather than a coincidence.

**The carve is by generator FAMILY, not by generator.** So-Fake-OOD ships 15
generators but only 8 families: splitting `Ideogram2` from `Ideogram3`, or
`imagen3` from `Imagen4`, would call a model "unseen" when its sibling was in the
calibration set. `scripts/build_manifest.py` carves whole families —
Flux + Ideogram + Recraft to calibration, GPT + Imagen + Seedream + nano_banana +
Hidream held back for eval — and asserts the two sides share no generator and no
image. 2,044 calibration images / 4,198 eval images.

```python
X, R = load("so_fake_ood")           # split is now calib_ood or test_ood
cal  = R.split == "calib_ood"        # fit calibrators and fusion here
ev   = R.split == "test_ood"         # report here, never fit
```

`load()` takes `split` from the manifest, not from the shard, so the carve is
visible everywhere without re-embedding anything.

**What it did and did not fix.** Fusion's deficit against the general probe went
from −0.0112 clean to −0.0018 — the mechanism was real and the carve closed 85%
of it. But fusion reaches *parity*, not a win. Kacey measured +0.0042 with a
random generator split; under the stricter family-disjoint carve that gain does
not survive. **`predict.py` stays on `max`** — see §5e.

### 5e. Why `max` and not a learned combiner

On the full task — So-Fake-OOD (fully synthetic) pooled with `sid_tampered_eval`
(locally edited) against the same reals:

```
combiner          FULL avg   FULL worst   ood clean   ood worst
general alone       0.7331       0.7036      0.9124      0.8798
max(gen,tamp)       0.8728       0.8414      0.9028      0.8589
fusion LR           0.8440       0.8175      0.8583      0.8337
```

The task is **disjunctive**: "AI touched this" = fully synthetic **OR** locally
edited, and the general probe is *inverted* on tampering (0.37). A linear model in
log-odds is forced into one additive trade-off across two complementary
detectors; `max` lets whichever one fires win. Prevalence re-weighting and a
noisy-OR hybrid were both tried and neither beats it.

Reporting fusion on So-Fake-OOD alone understates it — that set contains no
tampered images at all — which is why the pooled column exists.

---

## 5f. Model polish since Kacey's handover — Adriel

Nothing here changed an architecture. Kacey's calls all survived review; what
follows is integrity, calibration, and honest measurement. Every number in the
repo moved slightly as a result, so regenerate rather than quoting old tables.

### 1. Closed a cross-split leak that touched every branch

SID_Set ships **43 images in both its tampered train and validation splits**.
`build_manifest.py` resolved it (eval wins) — but `load()` read the cache
directly and never consulted the manifest, so every branch script silently
bypassed the dedupe: Kacey's fusion, `eval_grid.py`, and `train_tampered()`
alike. 80 rows of eval data were in the tampered probe's training set.

Fixed in `load()` rather than in six callers, so all of them inherit it. Effect
was small but in the dishonest direction — tampered **0.9464 → 0.9440**. Guarded
by an assert in `general.py`'s `__main__` that fails if any two sources ever
share an `image_id` again.

### 2. Built `calib_ood` and it fixed calibration ~5x

Full detail in §5d. The short version: calibrating on `sid_calib` was fitting
Platt against branches that score 0.999 there, and the resulting slope
manufactured over-confidence on every unseen generator.

```
branch    AUC on cal set   ECE (sid_calib)   ECE (calib_ood)   factor
general           0.9996            0.1026            0.0217     4.7x
face              0.9976            0.1665            0.0333     5.0x
spectral          0.6789            0.0774            0.0519     1.5x  <- control
```

The control row is what makes this a mechanism rather than a coincidence: the
branch that never aced its calibration set barely moves. **This matters most for
the demo** — every confidence number a judge sees is now ~5x better calibrated.

Carved by generator **family**, not generator, which is stricter than the
in-memory experiment it replaces. Under that stricter test fusion reaches parity
with the general probe rather than beating it, so `predict.py` keeps `max()`.

### 3. Established *why* `max` beats a learned combiner

Kacey measured that fusion loses. The ablation localises it and the pooled
evaluation explains it. Adding `cal_tampered` to the fusion input is the entire
regression (0.9124 → 0.8658 on its own); the rest of the vector is noise. And
So-Fake-OOD understates every combiner that handles tampering, because it holds
no tampered images at all. Pooling both eval sets against the same reals:

```
combiner          FULL avg   FULL worst
general alone       0.6851       0.6542
max(gen,tamp)       0.8597       0.8210
fusion LR           0.8511       0.8150
```

The task is **disjunctive** — "AI touched this" = fully synthetic OR locally
edited — and the general probe is *inverted* on tampering (0.37). A linear model
in log-odds is forced into one additive trade-off across two complementary
detectors; `max` lets whichever fires win. Prevalence re-weighting and a noisy-OR
hybrid were both tried; neither beats it. `predict.py`'s `max()` now rests on two
independent measurements instead of one.

### 4. Made every reported number held-out

`eval_grid.py` and `face.py` both excluded nothing before the carve existed and
would have scored the calibration slice as if it were held out. Both now filter
`calib_ood`, and they agree exactly (face 0.9421 / 0.9168).

### 5. Tooling

- **`scripts/eval_grid.py`** — regenerates `docs/robustness.md`, the required
  Robustness Evaluation Summary. Refits from cache every run so it cannot drift
  from a stale `.npz`. Gains a `fused` column in one line if fusion ever ships.
- **`scripts/try_face.py`** — score individual images through the face and
  general probes; `--save-crops` writes the aligned 224px crop so alignment can
  be eyeballed. Alignment broke silently once in this project and a mirrored
  face still scores confidently, so this is the check for it.

### 6. Doc corrections

The face coverage figures in §5a were wrong (`noise01` loses 15–18%, not 77%),
the `face_px` range was wrong (64–612, not 64–181), and `robustness.md` was
selling the blur AUC rise as robustness when `HANDOVER-MODELS.md` §8 shows it is
a shortcut. All three are corrected in place. Kacey caught the first two.

---

## 6. Gotchas that already cost us time

- **Backbone is `ViT-L-14-quickgelu`, not `ViT-L-14`.** OpenAI weights need
  QuickGELU. The wrong one silently shifts every embedding — measured cosine
  0.876–0.908 between the two on identical images.
- **`load_source` prefix collision.** `sid_tampered_*` used to also match
  `sid_tampered_eval_*`, training a model on its own eval set. Fixed, but if you
  add a source whose name prefixes another, re-check it.
- **SID_Set's own train/validation splits share 43 images.** Content hashing
  caught them; the manifest resolved them to eval. Trust `image_id`, not the
  dataset's split labels.
- **Never unfreeze the backbone.** LoRA fine-tuning a comparable encoder dropped
  in-the-wild accuracy 95.9% to 63.5%. Corollary from `HANDOVER-MODELS.md` §9:
  with a frozen encoder, a shortcut lives in the *embedding*, so it has to be
  removed on the input side. No amount of retraining the probe can unsee it.
- **Never junction `data/` into a git worktree.** `git worktree remove --force`
  does not treat a Windows directory junction as a link — it follows it and
  deletes the target. This wiped `data/cache/embeddings/` mid-session; 107 of
  130 shards came back from the HF push and the rest had to be re-embedded.
  Copy the files, or run branch code in the main tree.
- **`pd.read_csv(usecols=[...])` returns columns in FILE order, not the order you
  asked for.** `dict(zip(*df.to_dict("list").values()))` therefore builds the map
  backwards and the filter silently passes everything. It cost an hour here.
  Zip by explicit column name.
- **CSV line endings.** `ShardWriter` now writes LF and `.gitattributes` pins
  `*.csv eol=lf`. Before that, a fresh `pull_cache.py` rewrote every tracked
  manifest CSV to CRLF — 109 files, ~973k insertions, zero content change — and
  the next `git add -A` committed it.

---

## 7. Open items

| item | owner | blocking |
|---|---|---|
| **WildFake DALL·E Advanced** (ModelScope, manual) | Michael or Valentino | the organizer benchmark — see below |
| **Push the cache again** — remote predates the carve | Adriel | anyone who pulls calibrates on `sid_calib` |
| `quorum/detectors/spectral.py` — features exist, model is 9 params | Albert | fusion's weakest input |
| 25 all-zero vectors in `spec_so_fake_ood` (§4b) | Albert | nothing, but it is silent |
| `real_extra/openimages` — real-class diversity | optional | — |
| `social_real/` — 200–500 screenshots, manual | Michael / Valentino | deployment realism |
| `provenance.py` (C2PA) | unassigned — cut candidate | nothing |
| ~~generator-disjoint calibration slice~~ — **done**, §5d | — | — |
| ~~`organizer_val` face + spectral~~ — **done** | — | — |
| ~~`faces/` dataset~~ — **cut**, probe saturates at ~500 imgs | — | — |
| ~~text branch~~ — **cut** by Kacey, slots kept at neutral fill | — | — |

**`organizer_val` cannot be scored at all until WildFake lands.** COCO val2017 is
100% real, so `test_organizer` has 5,000 negatives and zero positives — there is
no positive class and therefore no AUROC, not even a bad one. It is the only
externally-comparable number the submission gets.

### WildFake commands, for whoever takes it

```powershell
# modelscope.cn -> hy2628982280/WildFake
# ONLY the "DALL-E Advanced" subset (8,843 images). Not the whole dataset.
# Extract to: data/raw/organizer_val/wildfake_dalle_adv/

python scripts/embed_dir.py --dir data/raw/organizer_val/wildfake_dalle_adv `
  --source organizer_val --assign-split test_organizer --label 1 `
  --generator dalle --full-grid

python scripts/embed_dir.py --dir data/raw/organizer_val/wildfake_dalle_adv `
  --source organizer_val --assign-split test_organizer --label 1 `
  --generator dalle --full-grid --features

python scripts/build_manifest.py
```

`--label 1` and `--assign-split test_organizer` are **not optional**. Assertion A
catches a wrong split; nothing catches a wrong label.
