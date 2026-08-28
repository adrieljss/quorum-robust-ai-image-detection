# Quorum — Kacey's branch plan

Working plan for the `face-improve` branch. Forked from `TODO.md` at `15c9c6c` so
the shared roadmap keeps one owner; **`TODO.md` stays authoritative for everything
outside Stage F and Stage T**.

**Priority, agreed with Adriel 2026-08-28: face first. Text is parked.**

| stage | status |
|---|---|
| **Stage F — face branch improvement** | **active, this is the work** |
| Stage T — text deformation | parked, optional, do not start |
| Candidate features | for discussion, nothing assigned |

Face is mine by the ownership table. General, tampered and spectral are Albert's —
measure them if useful, never edit them.

Baseline on `15c9c6c`, the number to beat:

```
face     clean 0.9421   worst 0.9168 (noise01)
general  clean 0.9125   worst 0.8798 (noise002)   <- Albert's, for context
```

**Decision discipline for everything below.** Pick on `calib_ood`, confirm on
`test_ood`. Never pick on `test_ood` — that is tuning on the headline eval. And
`calib_ood` holds only ~550 face images, so it cannot resolve differences below
about 0.005: only act on effects that are large *and* consistent across both
splits. The C sweep below is the worked example of what happens otherwise.

---

From here to a working frontend demo with real models behind it.
Ordered by dependency: each stage unblocks the next.

| who | owns |
|---|---|
| **Adriel** | data embedding, data handling, manifest, evaluation |
| **Albert** | general probe + regularity (spectral) |
| **Kacey** | face probe + text + **fusion** |
| **Michael & Valentino** | demo backend and frontend, and everything software around it |

Judging weights raw accuracy at zero. Well-structured code, a reliable demo,
sharp framing, and evident deliberate decisions outscore a better AUROC.
Budget effort accordingly.

---

## Stage 0 — Blocking questions (28 Aug webinar) — **everyone**

- [ ] **Is the 2B parameter cap per model or total across the ensemble?**
      Answer changes the architecture. Ask first.
- [ ] Are external APIs permitted, or must everything run locally?
- [ ] Training on WildFake — how do we identify the excluded validation rows?

If the cap is *total*, the shared-CLIP design (~317M across all branches)
already holds. If *per model*, there is headroom for a second backbone.

---

## Stage 1 — Finish the data — **Adriel**

- [x] SID_Set train / calib / tampered embeddings
- [x] So-Fake-OOD embeddings (headline eval)
- [x] COCO val2017 → `organizer_val`, quarantined
- [x] `main.csv` + `stats.md`, assertions A–F green
- [x] `sid_tampered_eval` face + spectral (1,499 spec / 244 face)

### Handover — DONE, Albert and Kacey are unblocked
- [x] HuggingFace org `techjam2026blueberryjam` created
- [x] `push_cache.py` -> `techjam2026blueberryjam/quorum-cache`, 1,210 MB, private
- [x] `docs/HANDOVER.md` §1 points at the org path
- [ ] Invite Albert, Kacey, Michael, Valentino with **write** access
- [ ] Delete the stale personal `adrieljss/quorum-cache`
- [ ] **Revoke the write token that was pasted in chat**, issue a fresh one

### Second push — **now blocking, was not before**

The remote cache predates the `calib_ood` carve. Anyone who pulls today gets
pre-carve manifests, calibrates on `sid_calib`, and ships probabilities ~5x
worse on unseen generators without any error telling them so. Push before
Albert or Kacey pulls again.

- [x] `organizer_val` face + spectral pass — 75,000 spectral / 3,941 face rows
- [x] ~~`faces/` dataset~~ — **CUT.** Face probe saturates at ~500 images
      (500 -> 0.8892 OOD clean, all 4,128 -> 0.8952). 5.4 GB and a `.rar` to
      feed a model that plateaued eight times ago. Coverage/demographic
      diversity is a *limitations* note, not a download.
- [x] Re-run `build_manifest.py` — 330,851 rows, assertions A–F green
- [x] Local cache deleted mid-session by a `git worktree remove --force` that
      followed a Windows directory junction into the real `data/cache`. Restored
      107/130 shards from the HF push; the 23 `organizer_val` face/spec shards
      postdated it and were re-embedded. **Do not junction `data/` into a
      worktree** — copy, or point the code at it.
- [ ] `push_cache.py` again. Delta is the 23 `organizer_val` face/spec shards,
      the re-carved `main.csv`, and refreshed `face/fusion/tampered.npz`
- [ ] Fresh-pull check afterwards: `.gitattributes` now pins `*.csv eol=lf` and
      `ShardWriter` writes LF, so a pull should no longer show ~973k phantom
      CSV insertions. Verify once, then trust it

### Handed to Michael or Valentino
- [ ] **WildFake DALL·E Advanced** — ModelScope, manual. Commands in
      `docs/HANDOVER.md` §7. `--label 1 --assign-split test_organizer` are
      not optional.

---

## Stage 2 — Branch models (Albert and Kacey in parallel)

Both read the same `main.csv`. Neither blocks the other.
Full briefs in `docs/HANDOVER.md` §4 and §5.

### Albert — general probe
- [x] Baseline 0.9124 clean / 0.8798 worst on So-Fake-OOD
- [ ] Try MLP head; keep linear unless it clearly wins on OOD
- [ ] Try multi-crop embedding (PIPELINE §4.5) — cheapest untried upgrade
- [ ] Freeze the winner into `data/models/general.npz`

### Albert — regularity / spectral
- [ ] `quorum/detectors/spectral.py` — ~1 hour, 9 parameters
- [ ] Per-variant AUC table (expect collapse under resize; that is correct)

### Kacey — face probe
- [x] `quorum/detectors/face.py`, conditioning on `face_px` — 0.9382 clean /
      0.9151 worst, up from 0.8952 / 0.8620
- [x] Per-variant AUC **and coverage**
- [x] Explain why face AUC *rises* under blur (0.9382 → 0.9500 at `blur20`).
      **Shortcut learning, not survivorship** — blur hurts in-distribution
      (-0.0047) and helps OOD (+0.0314). The probe partly reads resampling
      texture in upsampled crops; gain concentrates in small faces (+0.047 vs
      +0.009) and is anti-correlated with clean AUC across 14 generators
      (r = -0.685, p = 0.007). HANDOVER-MODELS.md §8
- [x] Linear vs MLP on the face branch — **linear wins**, 256x fewer parameters
      and a smaller transfer gap. Was a documented default, now a result. §9

### Kacey — text
- [x] Decide build or cut. Cut on 2026-08-27 to protect fusion time
- [x] **Reopened 2026-08-28** at Adriel's request. Fusion is done and at parity,
      so the cut's justification no longer holds. Full plan in Stage T below

---

## Stage 3 — Calibration and fusion — **Kacey**

Built, but **not shipping as the scorer**. Full reasoning and numbers in
`docs/HANDOVER-MODELS.md`.

- [x] Platt scaling per branch, `quorum/calibrate.py`. Calibrators fit on
      `calib_a`, fusion on `calib_b`, split by `image_id`
- [x] Reliability diagrams — `docs/figures/reliability.png`
- [x] CLIP zero-shot content label — `data/models/content_prompts.npz`
- [x] `degradation_estimate` feature — P(degraded) from the 8 spectral features
- [x] `quorum/fusion.py` — logistic regression over the 14-column input vector
- [x] Verify output contract exactly: `[{"image_path": ..., "pred": 0.87}]` —
      passes, including nested dirs, grayscale/RGBA, and all five extensions
- [x] Wire `predict.py` to fusion — **resolved: it stays on `max()`.** No longer
      blocked on the calibration slice; that landed and fusion still does not win. Measured on so_fake_ood clean/worst: raw `max()`
      0.9042 / 0.8634, fusion 0.8587 / 0.8340, so swapping it in *today* costs
      ~6 points. `predict.py` keeps `max()` and its docstring records why

### The calibration slice — **DONE (Adriel)**

- [x] **`calib_ood` carved from So-Fake-OOD**, by generator **family** rather
      than generator: `Ideogram2`/`Ideogram3` or `imagen3`/`Imagen4` on opposite
      sides would call a sibling model "unseen". Flux + Ideogram + Recraft
      calibrate (2,044 imgs); GPT + Imagen + Seedream + nano_banana + Hidream
      stay held back for eval (4,198 imgs). `scripts/build_manifest.py` asserts
      no generator and no image straddles the boundary
- [x] `load()` now takes `split` from the manifest, not the shard, so the carve
      is visible to every branch with no re-embedding. `HANDOVER.md` §5d
- [x] `calibrate.py` / `fusion.py` default to it

**It fixed calibration, and did not make fusion win.** ECE on unseen generators:

```
branch    AUC on cal set   ECE (sid_calib)   ECE (calib_ood)   factor
general           0.9996            0.1026            0.0217     4.7x
face              0.9976            0.1665            0.0333     5.0x
spectral          0.6789            0.0774            0.0519     1.5x  <- control
```

Kacey's mechanism is confirmed — the saturated branches improve ~5x, the branch
that never aced its calibration set barely moves. Fusion's deficit against
general narrowed from -0.0112 clean to **-0.0018**, i.e. parity. The +0.0042 from
the random *generator* split does not survive a family-disjoint one; splitting
model siblings was flattering the result.

So `predict.py` stays on `max()` on its own merits: the task is disjunctive and a
linear combiner loses on the pooled full task (0.8440 vs 0.8728). `HANDOVER.md`
§5e. The carve still earns its place — every probability the demo displays is now
~5x better calibrated on generators it has never seen.

Original blocker note, for the record:

```
branch     AUC on cal set   ECE in  ECE out  factor
general            0.9995   0.0022   0.1085   49.5x
face               0.9985   0.0143   0.1697   11.9x
tampered           0.9636   0.0062   0.0112    1.8x
spectral           0.6807   0.0520   0.0741    1.4x
```

The fusion retrain afterwards is seconds of compute. The split is the blocker,
not the training.

---

## Stage F — Face branch improvement — **Kacey, ACTIVE**

Mine by ownership, and the only branch I can improve without crossing a line.
Everything here runs against cached embeddings in seconds — no re-embedding, no
shared file touched.

### F0 — Settled, do not redo

- [x] **`face_px` conditioning** — 0.9382 clean / 0.9151 worst against a 0.8952 /
      0.8620 baseline. Standardised `log2`, because the effect is a ratio
- [x] **Linear beats MLP** — 770 params vs 49k/197k/214k, and the transfer gap
      *widens* with capacity while all models sit at ~0.999 in-distribution. Extra
      capacity only buys a better fit to the shortcut. `HANDOVER-MODELS.md` §9
- [x] **Blur anomaly is shortcut learning**, not survivorship. §8
- [x] **More data will not help** — learning curve flat from 500 to 4,128 images
- [x] **Do not tune `C`.** Sweep picks C=0.1 on `calib_ood` (0.9415), which then
      *loses* on `test_ood` (0.9385 vs 0.9421 at the default). Negative transfer,
      and the worked example of why `calib_ood` cannot resolve small effects

### F1 — Found, measured, not yet shipped

- [x] **Test-time augmentation works.** Blur the input, average the score:

```
inference does            calib_ood (pick)   test_ood (confirm)
clean only (today)            0.9337              0.9421
clean + blur20                0.9454              0.9539
blur20 only                   0.9494              0.9512
```

~+0.01, consistent in direction on both disjoint splits. Follows from the
shortcut finding: blur strips the resampling texture the probe leans on, and with
a frozen CLIP an input-side fix is the only kind that works — training on blurred
crops gave only +0.004/+0.001.

- [ ] **Do not implement yet.** Two reasons. It is validated for *clean* inputs
      only: the cache holds single transforms, so `blur20 ∘ jpeg30` does not exist
      and TTA on degraded inputs is unmeasured. And `predict.py` does not use the
      face branch at all — face reaches inference only through fusion, which is
      not wired in. Ship it when face is actually on the inference path
- [x] **TTA does not transfer to `general`** — 0.9033 → 0.8996 (calib), 0.9170 →
      0.9137 (test), monotonically worse. So the shortcut is specific to the face
      crop path, where small faces are upsampled ~3.5x into 224px. Whole-image
      embeddings already low-pass by resizing to 224. Hand this to Albert; it
      redirects him to multi-crop (`PIPELINE.md` §4.5) rather than TTA

### F2 — Next, testable today against cache

The size stratification says where the loss is. Small faces are far worse:

```
band     px range     clean AUC
small      64-153        0.8568
mid       153-305        0.9231
large    305-2192        0.9387
```

- [x] **Size-conditional modelling — tested, does not work.**

```
model                     calib_ood (pick)   test_ood (confirm)
baseline  [X, z]              0.9337              0.9421
+ z^2                         0.9357              0.9436
+ interaction X*z             0.9416              0.9348   <- calib up, test down
per-tercile probes            0.9444              0.9411   <- calib up, test down
```

`calib_ood` picks per-tercile probes; they lose on `test_ood`. Only `z^2` improves
both, by +0.0015, which is under the resolution limit. **Third independent
confirmation that capacity hurts this branch** after the MLP and now the
interaction model. Stop trying to make the probe bigger.

- [x] **Face + general ensemble on face-present rows — this is the win.**

```
combiner (face-present rows)   calib_ood (pick)   test_ood (confirm)
face alone                         0.9337              0.9421
general alone                      0.9477              0.9618
mean(face, general)                0.9586              0.9739
0.3*face + 0.7*general             0.9589              0.9736
```

`calib_ood` picks 0.3/0.7, confirming at **0.9736** — +0.0118 over general alone,
+0.0315 over face alone, consistent on both splits. Two things follow:

1. **General beats face even on face-present rows** (0.9618 vs 0.9421). The face
   branch's value is not standalone accuracy, it is complementarity
2. **This is fusion's job and fusion is not capturing it.** Fusion sits at parity
   overall while a fixed 0.3/0.7 blend gains +0.012 on the 27% of rows that carry
   a face. Likely cause: fusion fits one global weighting over all rows, and on
   the 73% with no face `cal_face` is a constant 0.5, so the optimal
   general-vs-face balance differs between the two populations and a purely
   additive model cannot express the switch

- [x] **Interaction terms in `fusion.py` — tested, no effect, hypothesis wrong.**
      `cal_general x face_present`, `cal_face x face_present` and both together
      all give **0.9149 clean / 0.8834 worst**, identical to shipped fusion to
      four decimals.

      It is algebraically obvious in hindsight: with the neutral fill,
      `cal_face * face_present == cal_face - 0.5 + 0.5*face_present`, a linear
      combination of columns already present. Perfectly collinear, so it cannot
      add information. The missing-branch rule was already doing its job and my
      "a linear model cannot express the switch" reasoning was simply wrong.

### F4 — The fusion trade is now explicit — `--fit` flag

Running `python -m quorum.fusion` produced **-0.0642** against general while
`TODO.md` documented **-0.0018**. Both reproduce; they are different fit sets, and
the shipped `__main__` used the one the doc did not describe.

`fusion.py` now takes `--fit {calib,calib+tampered}` (default `calib`) and prints
**both** either way, because reporting one without the other misrepresents the
model whichever you pick:

```
  fit on            ood clean  ood worst  tampered
  general alone        0.9170     0.8849    0.3696
  calib                0.9149     0.8834    0.3780   <- default, saved
  calib+tampered       0.8529     0.8291    0.8503
```

- [x] Flag added, both configurations printed, `fusion.npz` regenerated on the
      default
- [ ] **Tell Adriel the doc needs the second half.** "Parity" is true and
      incomplete: the parity configuration scores **0.3780 on tampered**, barely
      above general's 0.3696. Fusion can match general on the headline *or*
      handle tampered images. It cannot currently do both, and that is the honest
      framing for the error-analysis deliverable
- [ ] Re-check `report()` coverage numbers after any change — AUC on a shrinking
      population is not comparable across variants

### F3 — Needs pixels or someone else's file, so not now

- [ ] Multiple faces per image — only the largest is cropped and cached
- [ ] YuNet detection confidence as a feature — `features.py` discards it
- [ ] Sub-crop consistency within the aligned face — needs pixels

All three need a `features.py` change and a re-embed. Not worth it for a branch
already at 0.94; revisit only if the pixel pass happens for other reasons.

---

## Stage T — Text deformation detection — **PARKED**

> **Not being worked on.** Deprioritised with Adriel on 2026-08-28: face
> improvement is the better use of the time, and text is optional in the brief.
> Kept here because the research is done and the blockers are recorded — if it is
> ever revived, start at T0 rather than rediscovering the gated dataset and the
> id-join failure.

Everything here is additive: no shared file changes except one appended line in
`requirements.txt`, and the two fusion slots (`cal_text`, `text_present`) are
already reserved at neutral fill, so nothing downstream reshapes if it lands or
stays parked.

### T0 — Unblock the two things that need a human

- [ ] **Request TextFake access** — `huggingface.co/datasets/Yuning0123/TextFake`
      is `gated: manual`, so an author approves by hand and it may take days.
      Click first, work around it second. It is the branch's external benchmark,
      not its training set, so nothing below is blocked on it
- [ ] **Tell the team before appending `rapidocr` to `requirements.txt`.** Shared
      file, append-only, never reorder or re-pin other lines

### T1 — The pixel pass (this is the embedding Adriel means)

OCR reads pixels and **there are none on disk** — the streaming pass discarded
every image, which is why face and spectral have caches and text has nothing.
This branch has to produce its own, and it is the one pass that does not touch
CLIP at all.

```bash
python scripts/stream_embed.py --dataset saberzl/So-Fake-OOD --split test_image \
  --source so_fake_ood --assign-split test_ood --n-per-class 5000 \
  --no-embed --save-images
```

**Smoke-tested 2026-08-28 on 20 images, and the join does not work. Do not run
the full pass until this is resolved.**

```
saved images          20
in main.csv            0    (0%, and 0 against ANY source)
```

Ids are 16 hex both sides, so this is not a formatting mismatch. Two separate
problems were found, and only the first is proven:

- [x] **`embed.py:158` is wrong.** The comment reads
      `img.save(f, "JPEG", quality=95)  # q95 again = same bytes, same id`.
      JPEG q95 is **not idempotent** — measured on a 512px image, a second
      round-trip changes **92.6% of pixels**, max delta 43, mean 6.0. So a saved
      image re-hashes to a different id than its own filename, and any consumer
      that reloads and re-normalises can never join. Adriel's file — report, do
      not edit
- [ ] **Unexplained: a fresh stream produces ids absent from the manifest.** The
      filename is written from the pre-save id and *should* be correct, so
      non-idempotence alone does not explain 0/20. Remaining candidate is that
      `image_id` is environment-dependent: it hashes the q95 round-trip, and
      Pillow/libjpeg is unpinned in `requirements.txt`, so a different encoder
      version yields different pixels and therefore different ids for the same
      source image. Cannot be tested from one machine — **ask Adriel**

Until that is answered, the ways forward, in order of preference:

- [ ] **A. Ask Adriel to run the `--save-images` pass on the machine that built
      the cache.** Same environment, so ids match by construction and the join
      works. Cheapest fix, and it is his pass anyway
- [ ] **B. TextFake standalone.** Needs no join at all; blocked only on the gate
- [ ] **C. Own source.** Re-stream under a new `--source`, self-consistent ids,
      text branch evaluated standalone. Cannot feed fusion — different images from
      every other branch — so this is a demo/eval path, not an integration one

- [ ] `data/raw/` is gitignored — confirmed, pixels never enter git

Three reasons this is the right shape:

- **No CLIP, so no fp16 hazard.** `embed.py` has no MPS path and falls back to
  CPU fp32, which would poison a shared cache with mismatched numerics. `--no-embed`
  sidesteps that entirely — this pass is safe to run on the Mac
- **No re-streaming for variants.** `degrade.py` is seeded off `image_id`, so all
  15 variants regenerate offline bit-exact from the saved clean image
- **It joins the manifest.** Features keyed on `(image_id, variant)` reach fusion
  the same way face and spectral do. **TextFake cannot do this** — different
  images, no shared ids — which is why it is the benchmark and not the base

- [ ] `data/raw/` is gitignored. Confirm the pixels never enter git

### T2 — The scorer

`PIPELINE.md` §2.3 freezes the contract: `float32[6]` in, one raw score plus
`text_present` out, `LogisticRegression`. Six features from §6:

| # | feature | note |
|---|---|---|
| 1 | mean OCR confidence | |
| 2 | std of OCR confidence | the variance is the actual signal |
| 3 | dictionary hit rate | multilingual problem, below |
| 4 | mean glyph consistency | same character, same shape? |
| 5 | fraction non-ASCII | multilingual problem, below |
| 6 | region count | |

- [ ] `quorum/detectors/text.py` — extractor + probe
- [ ] **Decide the multilingual question before fitting.** TextFake is 28
      languages (Chinese 15.5%, English 13.6%). Features 3 and 5 are
      English-shaped: an English dictionary hit rate on Chinese scores ~0 for real
      *and* fake, and `frac_nonascii` becomes a language detector rather than an
      artifact detector. Either restrict the dictionary check to detected-Latin
      regions and let `text_present` carry the rest, or swap both for
      script-agnostic stand-ins (confidence distribution shape, stroke-width
      variance). Say which in the README
- [ ] Return `(np.zeros(6, dtype=np.float32), False)` when OCR finds nothing.
      Neutral fill plus presence flag — "no text here" must never read as "the
      text model says real"
- [ ] **Cache OCR output per `(image_id, variant)`.** It is by far the slowest
      thing in this project and the eval grid wants 15 variants per image. Full
      grid was estimated at ~9 hours; clean-only is ~45 minutes. Start clean-only
- [ ] `__main__` block of hard asserts, matching every other module. No pytest

### T3 — Evaluation

- [ ] Per-variant AUC **and `text_present` coverage**, together — same discipline
      as the face branch, and for the same reason: a score over a shrinking
      population is not comparable across variants
- [ ] TextFake as the external benchmark once access lands
- [ ] Fit calibration on `calib_ood`, never `sid_calib` — the carve exists
      precisely because a saturated calibration set manufactures over-confidence
      (`HANDOVER-MODELS.md` §4)

### T4 — Wire into fusion, only if it earns it

- [ ] Swap `cal_text = 0.5` / `text_present = 0` for the real values in
      `fusion.py`. One line each; `COLUMNS` does not change
- [ ] **Gate:** keep it only if fusion with text ≥ fusion without, on the
      family-disjoint carve. Fusion sits at parity with general (-0.0018); a
      weak sixth signal can push it back under. Measure, do not assume

### T5 — README, and the honest limitation

- [ ] **TextFake undercuts our own premise and this must be stated.**
      `PIPELINE.md` §6 calls garbled signage a high-precision signal. TextFake
      finds the inverse ordering: GPT-Image-2 renders text *well* (70% entity OCR
      hit rate) and is the hardest generator to catch, while low-fidelity
      generators are easy. So the feature is high precision with **falling recall
      against frontier generators** — exactly the trade-off the error-analysis
      deliverable asks for
- [ ] Note the text-density curse for the spectral branch: as glyph density
      rises, frequency methods lose 37–43%. `text_present` reaching fusion lets it
      discount the spectral score, which is what that flag is *for*
- [ ] Expect a lopsided raw scorer before Platt — most detectors on this benchmark
      run 90%+ on real, under 35% on fake. That is conservative bias, not a bug

### Guardrails

- Six features and a `LogisticRegression`. `PIPELINE.md` notes a threshold would
  nearly do. **Do not let this become a model**
- Do not touch `quorum/embed.py`, `features.py`, `degrade.py`, or `scripts/*`
- If this starts costing time that face, general or fusion need, cut it again.
  It was optional when it was assigned and it stays optional

---

## Candidate features beyond text — **for discussion, nothing assigned**

Adriel asked what else we could extract if text is not the best use of the time.
Ranked against where the measurements say the holes actually are, not against a
generic forensics checklist. Three facts drive the ranking:

- `general` scores **0.3688** on tampered images — inverted, worse than chance.
  Largest single failure in the system
- `face` abstains on **73%** of images (92% on `organizer_val`). Coverage, not
  accuracy, is its ceiling
- Fusion is at **parity** (-0.0018 vs general). Folding in a signal badly cost 6
  points once already, so the bar is "does it help *in combination*", not "is it
  a real signal"

That favours features which fill a hole over another global score.

### Tier 1 — worth building

**1. Patch-level self-consistency (multi-crop CLIP).** 3x3 split, embed each patch
through the *same* frozen CLIP, take variance across patches, max pairwise
distance, and each patch's distance to the global embedding.

Attacks the tampered hole directly: a locally edited photo is globally authentic
but locally inconsistent, which is exactly why the general probe inverts on it.
Zero new parameters, no cap impact. Relative across patches, so global degradation
shifts all of them together. `PIPELINE.md` §4.5 already calls multi-crop the
cheapest untried upgrade and it sits unstarted on Albert's list. Cost is 9x CLIP
forwards on the same pass.

**2. Provenance — C2PA / EXIF / PNG chunks.** No pixels needed, just file bytes.
`provenance.py` is unbuilt and unowned; `provenance_prior` already sits in the
fusion vector at neutral 0.5.

Scores ~nothing on the graded table, because the grid re-encodes and strips
metadata — say so plainly rather than hiding it. The brief allows "reasonable
deployment assumptions as long as they are stated clearly", and "in production you
check the signed manifest before running a 300M-parameter model" is the strongest
Impact & Relevance argument available to us. Also the best demo material: a verdict
that cites a signature beats one that cites a logit.

### Tier 2 — cheap, plausible, unmeasured

**3. Depth-of-field consistency.** Tile, compute local Laplacian energy, ask
whether sharpness varies *smoothly* across space (real optics) or is uniform or
patchy (generated). Physical prior, orthogonal to CLIP semantics.

**4. Noise-residual spatial moments.** `features.py` already computes a
median-blur residual and then only takes its FFT bands. Kurtosis, skew and
variance-of-local-variance of that same residual are different information at
near-zero extra cost — three more scalars in the existing 8-vector. Cheapest
possible addition in the whole project.

**5. Chromatic aberration.** Real lenses displace R against B at high-contrast
edges; generators rarely simulate it. Cheap, physical, not directly targeted by
the augmentation grid.

### Tier 3 — considered and rejected

| feature | why not |
|---|---|
| JPEG double-quantization / ghost | the grid re-encodes at 4 qualities — self-defeating |
| Demosaicing / CFA periodicity | destroyed by any resize; SID images already re-encoded |
| Colour / illumination histograms | `jitter02` attacks brightness/contrast/sat ±20% directly |
| Lighting & shadow geometry | expensive, fragile, needs its own model |
| Hand / anatomy detector | another model; the face learning curve suggests a fast plateau |

### The constraint that gates all of them

Every pixel-based option here — text included — is blocked on the same thing:
**no pixels on disk, and the pixel pass currently produces `image_id`s that do not
match the manifest** (T1). That one unresolved question gates text,
patch-consistency, DoF, residual moments and CA alike, so getting Adriel's answer
on it outranks choosing between them.

Provenance is the exception: it needs original files rather than normalised
pixels, so it is blocked differently, and needs no new pass at all if source files
are kept on the demo path.

---

## Stage 4 — Evaluation — **Adriel**

- [x] `scripts/eval_grid.py` — all 15 settings x 4 branches, plus the combiner
      comparison. Refits from cache each run so it cannot drift from a stale
      `.npz`, and reports held-out rows only (excludes `calib_ood`)
- [x] **Robustness Evaluation Summary** — `docs/robustness.md`, regenerated.
      general 0.9170/0.8848, face 0.9421/0.9168, spectral 0.6736/0.5471,
      tampered 0.9528/0.8962 (clean/worst)
- [x] Carries the blur caveat inline so the face row cannot be misread as
      robustness — it is the shortcut-learning result from HANDOVER-MODELS §8
- [x] `scripts/try_face.py` — score individual images through the face + general
      probes, `--save-crops` to verify alignment. ~13s model load then ~50ms an
      image, so pass them all in one invocation. Useful template for the demo
      backend: load the model once at import, never per request
- [ ] Reference number on `organizer_val` — **blocked, and hard-blocked.**
      COCO val2017 is 100% real, so the organizer set has no positive class and
      **no AUROC can be computed at all** until WildFake DALL·E Advanced lands.
      This is the only externally-comparable number we get. Chase it.
- [ ] Per-content-bucket AUROC — wild variance means we are partly reading
      semantics, and saying so is a strength
- [ ] Re-run the grid once WildFake lands; `organizer_val` is the only
      externally-comparable number and it is still unscoreable

---

## Stage 5 — Error analysis — **Adriel** (required deliverable)

Feeds Innovation & Problem Insight at 20%. Not an afterthought.

- [ ] 3–5 representative **false positives**, each with a hypothesis
- [ ] 3–5 representative **false negatives**, same
- [x] Two concrete cases already banked: a FLUX_2 face the general probe rates
      0.5438 (a coin flip) and the face branch catches at 0.9567 — the
      complementarity argument in one image; and a TAMPERED image where both
      branches sag (0.63 face / 0.44 general), the documented inversion
- [x] Most of the narrative is already written — HANDOVER-MODELS §11 on why a
      single probe scores 0.91 on unseen generators and 0.37 on edited photos
- [ ] Stated trade-offs: robustness vs clean accuracy, generalisation vs
      in-distribution ceiling, false-positive cost at platform scale

---

## Stage 6 — Demo — **Michael & Valentino**

The judged artifact. Reliability beats features.

**Start now against stubbed scores.** Do not wait for Stage 3. `predict.py` was
built stubbed in Phase 1 for exactly this reason — integration problems are
cheap to find early and expensive to find in the last 48 hours.

### Backend
- [ ] Wrap `predict.py` / fusion behind an API the frontend can call
- [ ] Return the richer internal schema: per-branch signals, content type,
      reliability, degradation estimate
- [ ] Graceful degradation: a missing branch shows "not measured", never "real"
- [ ] Handle upload of arbitrary user images without crashing

### Frontend
- [ ] Feed-style slideshow UI (image-level, in scope)
- [ ] Upload → live verdict
- [ ] Side-by-side clean vs degraded, showing the robustness claim visually
- [ ] Per-branch breakdown so the ensemble is legible to a judge
- [ ] **No TikTok logo or branding anywhere**
- [ ] Test on a fresh clone, on a machine that never trained anything

---

## Stage 7 — Submission — **everyone**

- [ ] `README.md` — overview, setup, reproduction, limitations, per-member
      contributions
- [ ] Limitations section — easy marks; material already exists in SPEC §6.3
      and Stage 5
- [ ] `predict.py` verified on a fresh clone
- [ ] Public GitHub repo, structured and commented
- [ ] Devpost description — approach, tools, models, libraries, datasets
- [ ] Demo video on YouTube, public, end-to-end, **no third-party trademarks**
- [ ] Robustness summary attached (Stage 4)
- [ ] Error analysis note attached (Stage 5)

---

## Critical path

```
Stage 0 (webinar)
   │
Adriel: data ─┬─> Albert: general + spectral ─┐
              └─> Kacey: face ────────────────┴─> Kacey: FUSION
                                                     │
                        Michael & Valentino: demo ───┤ (start stubbed, wire late)
                                                     │
                                          Adriel: eval + error analysis
                                                     │
                                                 submission
```

**Fusion is no longer the bottleneck** — built, measured, and deliberately not
shipped as the scorer. The critical path is now the demo (Stage 6) and the two
required write-ups (Stages 4–5). Text and `provenance.py` are already cut.

**Michael and Valentino are not blocked by anyone.** The demo can be fully built
against stubbed scores and wired to the real model in an afternoon. Treat any
week where the demo has not progressed as a scheduling failure, not a dependency.
