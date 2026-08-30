# Quorum — Sequential TODO

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

### Second push — **DONE 28 Aug**

Was blocking: the remote predated the `calib_ood` carve, so anyone pulling would
have calibrated on `sid_calib` and shipped probabilities ~5x worse on unseen
generators with no error saying so. Verified on the remote — 130 shards,
`main.csv`, 6 models, still private, and it includes Kacey's `--fit calib`
`fusion.npz` because the merge landed first. **Albert and Kacey should re-pull.**

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
- [x] `push_cache.py` again — landed 28 Aug 05:33 UTC
- [ ] Fresh-pull check: `.gitattributes` now pins `*.csv eol=lf` and
      `ShardWriter` writes LF, so a pull should no longer show ~973k phantom
      CSV insertions. Verify once, then trust it
- [ ] **Still open from the first push, and overdue:** invite Albert, Kacey,
      Michael, Valentino with **write** access; delete the stale personal
      `adrieljss/quorum-cache`; **revoke the write token pasted in chat** and
      issue a fresh one (`hf auth login`). It has org admin rights.

### WildFake — **DONE 28 Aug**, by Adriel
- [x] `scripts/fetch_wildfake.py` — reads the 25.6 GB `DALLE.zip` central
      directory over HTTP range requests and inflates only
      `DALLE/Advanced/DALLE3`. ~1.5 GB instead of 25.6 GB, no `modelscope` SDK.
- [x] **The subset is 3,719 images, not 8,843.** WildFake files it as 8,843
      entries; 1,808 basenames repeat with an identical CRC32 and size, so the
      brief's figure is a FILE count. Docs corrected in `DATA_LAYOUT.md`,
      `HANDOVER.md`, `README.md`.
- [x] Both embed passes + `build_manifest.py`; verified
      `{0: 75000, 1: 55785}`, split `test_organizer`, 8,719 unique images
- [x] `push_cache.py` — 1.48 GB on the remote
- [x] `eval_grid.py --source organizer_val` — general **0.9837 / 0.9729**,
      drop 0.0108. First externally-comparable number the project has had.
- [x] Fixed: `eval_grid.py` wrote every source to the same `robustness.md`, so
      the organizer run silently overwrote the So-Fake-OOD deliverable. Output
      path is now per-source, and the blur caveat only prints where it is true
      (face FALLS under blur on organizer_val: 0.9520 -> 0.8887).

---

## Stage 2 — Branch models (Albert and Kacey in parallel)

Both read the same `main.csv`. Neither blocks the other.
Full briefs in `docs/HANDOVER.md` §4 and §5.

### Albert — general probe
- [x] Baseline 0.9124 clean / 0.8798 worst on So-Fake-OOD
- [x] Try MLP head; it underperformed the linear models on OOD
- [x] Compare linear alternatives; RidgeClassifier (`alpha=0.001`,
      `solver="lsqr"`) selected at 0.9221 clean / 0.8981 worst
- [ ] **Multi-crop / patch self-consistency (PIPELINE §4.5) — promoted to the
      top of this list.** It is now the highest-value untried idea in the whole
      project, for three separate reasons and one build:
      1. It is the principled fix for our largest failure — the tampered branch
         flags **a quarter** of real COCO photographs (24.2%, re-measured
         30 Aug; the old "a third" was stale) because it cannot generalise
         "untampered" to unseen photography. Comparing 3x3 patches *against each
         other* never needs to know what real photography looks like globally.
         **§5h confirms the branch ships, so this fix is still wanted** — it is
         the only route to keeping edit-detection without paying for it in
         false positives on unfamiliar photography
      2. It gives us **explainability, which we currently have none of.** A
         per-patch score is a heat map: a verdict that points at a region beats
         one that cites a logit, and it is judged
      3. Zero new parameters, same frozen CLIP, 9x forwards on one pass
- [x] Freeze the winner into `data/models/general.npz`

### Albert — regularity / spectral
- [x] `quorum/detectors/spectral.py` — scorer over the existing 8 features
- [x] Per-variant AUC table — clean 0.7365 / worst 0.5596 (`blur10`)
- [x] Confirmed classifier changes do not materially improve spectral; retain it
      as a complementary low-level signal for fusion

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
- [x] Decide build or cut. **Cut.** Nine hours of OCR for a 7-parameter model
      against fusion being the critical path. Both slots stay in the fusion
      vector at neutral fill, so wiring it back later is one line
- [x] **Attempted anyway, 29-30 Aug (Adriel, spare machine). Still cut, and now
      for measured reasons rather than budget ones.** `HANDOVER.md` §5b.
      Attempt 1, six OCR statistics: cross-dataset transfer **0.4627, below
      chance** — the features track text composition, not deformation, and five
      of six flip sign between datasets. This **falsifies `PIPELINE.md` §6's
      "garbled signage is a high-precision tell"** on our data: SID_Set's AI
      images have *cleaner* text than its reals. Attempt 2, CLIP on warped OCR
      crops (769-d, face-branch shape): genuinely works — transfer 0.8083, logit
      correlation 0.391 with the shipped score — but worth **+0.0022
      [+0.0015, +0.0029]** on organizer_val, and it lifts every content class
      uniformly, so it does **not** close the text-heavy gap it was built for
      (-0.0420 -> -0.0409). Code in `quorum/detectors/text.py`, untracked, wired
      into nothing. Not measured: the 15-variant grid, ~1.5h on a 500-image
      subsample — that is the run that would decide it

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
general narrowed from -0.0112 clean to **-0.0022**, i.e. parity. The +0.0042 from
the random *generator* split does not survive a family-disjoint one; splitting
model siblings was flattering the result.

**Parity has a price tag, and quoting it without the price is wrong.** Fusion has
two fit sets and they are different models — I reported one in prose while
`__main__` shipped the other. Kacey caught it and added `--fit`:

```
fit on            ood clean  ood worst  tampered
general alone        0.9170     0.8848    0.3698
calib                0.9148     0.8833    0.3806   <- default, shipped
calib+tampered       0.8526     0.8288    0.8483
```

The parity model reaches parity by becoming the general probe (`cal_tampered`
weight +0.100). Fusion can match general on the headline **or** detect tampering,
never both. `python -m quorum.fusion` now prints both rows either way.

So `predict.py` stays on `max()` on its own merits: the task is disjunctive and a
linear combiner loses on the pooled full task (0.8511 vs 0.8597 held out — the
0.8440-vs-0.8728 figures this line used to carry predate the leak fix and made
the margin look 3x wider than it is). `HANDOVER.md`
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
- [x] **`predict.py` has an operating point.** It never did — 0.5 was the
      sigmoid default, chosen by nobody, and it cost 0.09 precision while
      flagging 27.6% of COCO photographs as AI. `scripts/pick_threshold.py`
      picks 0.766 on `calib_ood`; the score is shifted so 0.5 *is* that point,
      which leaves AUROC bit-identical. A trade, not a free win — recall
      0.882 → 0.767, F1 0.820 → 0.808, and 0.5 was already near F1-optimal.
      `HANDOVER.md` §5f.6
- [x] `scripts/try_grid.py` — one image through all 15 variants, or `--chain`
      for all 196 composed pairs. The robustness claim on something you can
      look at. Demo material
- [x] **Chained degradation measured** — `scripts/chain_eval.py`. The official
      grid is single transforms only; composed pairs cost ~0.013 AUROC and
      degradation does **not** compound. `HANDOVER.md` §5f.7
- [ ] **Re-run `chain_eval.py --n 100 --out c100.npz`** — the current number is
      n=50, where the AUROC standard error (~0.04) is wider than the entire
      spread of the worst-chain table. ~70 min, one 2.9GB shard download
- [ ] Fold the chained result into `docs/robustness.md` once the 200-image run
      lands. It is the only number in the submission that measures what actually
      happens to images in the wild
- [x] Reference number on `organizer_val` — **DONE 28 Aug.** general
      **0.9837 clean / 0.9729 worst**, drop 0.0108, over 8,719 images.
      `docs/robustness-organizer_val.md`. Two caveats belong with it: the
      shipped `max` is *worse* here than general alone (0.9541/0.8841),
      because there are no tampered images for the tampered branch to catch;
      and DALL·E 3 is an easier target than So-Fake-OOD, so quote 0.9170 as
      the headline, not this.
- [x] Per-content-bucket AUROC — measured. **Text-heavy is the worst class on
      both benchmarks** (so_fake_ood -0.0564, organizer_val -0.0420 against the
      pooled mean), which is what motivated the two text attempts above. Uses
      the free CLIP zero-shot `content_onehot()` over cached embeddings
- [x] **All six figures now benchmark the tampered eval set**, 30 Aug. It was
      absent from five of them: only `robustness.png`'s `tampered` column ever
      loaded `sid_tampered_eval`, and that column was unlabelled so it read as a
      fourth branch on So-Fake-OOD. Now: a recall curve on `threshold.png`, a
      third panel on `separation.png`, a `general (on edited)` column on both
      grids, a fifth column on `generalisation.png`, and a third eval-set pair
      on `benchmarks.png`
- [x] `make_figures.py --no-tampered` -> `docs/figures-no-tampered/`, the
      general-probe-alone counterfactual, threshold re-picked (0.640, not
      0.766). Evidence for Stage 5; **not** a proposal — see the decision below
- [x] Two figure captions asserted numbers instead of computing them (`0.9170`
      as the headline claim, "a quarter of ordinary photographs"). Both now
      derive from the data, so they cannot outlive the result
- [x] `pick_threshold.py` refactor: `shipped(X, names=...)` takes a branch
      subset and the accuracy-plateau rule is now `plateau()`, so the
      counterfactual re-picks its threshold by the *same* rule rather than a
      second copy of it
- [x] **Bug found and fixed in `pick_threshold.py` while re-running it**: its
      "COCO FP" column selected `organizer_val` by `variant` alone, so it
      counted every correctly-caught DALL-E image as a false positive against
      real photography — **56.3% at 0.5 where the truth is 27.6%**. The numbers
      in `predict.py`'s docstring were always right; the script that produced
      them had drifted after WildFake was added to that source. Same root cause
      as the `make_figures` bug fixed the same day, opposite direction.
      `HANDOVER.md` §6
- [x] Re-run the grid once WildFake lands — done; see above

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
- [x] `chain_eval.py` prints the most confident errors by `image_id`, so a run
      hands you the cases directly. The shard is deleted after the pass — without
      the id a false positive found there can never be looked at again
- [x] **The strongest case is banked and written up**: the tampered branch fires
      on 24.2% of real COCO photographs, and the obvious fix (more real-photo
      diversity) made it *worse* — COCO false positives 13.6% → 53.5% while its
      own AUROC rose 0.9528 → 0.9884. Capacity and data are not the lever.
      `HANDOVER.md` §5g
- [x] Stated trade-offs: robustness vs clean accuracy, generalisation vs
      in-distribution ceiling, false-positive cost at platform scale.
      **All four are measured, not asserted** — the threshold trade (§5f.6), the
      fusion trade (§5c), the `max` false-positive cost (§5g), and now the
      tampered branch itself on every axis (§5h)
- [x] **§5g is closed. The tampered branch stays** — decided 30 Aug, Adriel.
      Dropping it wins four of five metrics on synthetic-vs-real and cuts COCO
      false positives 8.9% -> 2.9%, but takes edited-photo AUROC from 0.9035 to
      **0.5286**, a coin flip, and recall on edited images from 74.6% to 10.7%.
      The two readings of "robust" disagree and both are real: the branch makes
      transform-robustness 6x worse on the organizer set (drop 0.0121 ->
      0.0773) and is the only thing that survives editing. `HANDOVER.md` §5h
- [ ] Write the Error Analysis Note itself. **The material is now banked** —
      §5h is most of it, and `docs/figures-no-tampered/` is the six-figure
      counterfactual. This is assembly, not research

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

### Positioning — two arguments that are ours to make

- [ ] **Our evaluation is harder than the public work, and someone else says so.**
      The widely-forked Kaggle notebook `darkmatternet/can-ai-detect-ai-cnn-vs-vit-xai`
      benchmarks on CIFAKE — one generator, one real source, 32x32, and a test
      split that shares its generator with train. Its own "Fork Experiments"
      list opens with *"Add a generator-held-out split rather than a random image
      split"* and closes with *"The strongest extension is not a larger model. It
      is a more difficult, generator-shifted evaluation."* That is exactly the
      `calib_ood` carve. An independent author naming our design decision as
      their recommended next step beats asserting it ourselves.
      **Do not claim we beat them on score** — their benchmark is easier, so
      their number is probably higher. Our in-distribution equivalent is 0.9996;
      the gap to 0.9170 is the cost of being measured honestly
- [ ] **Cost framing — say it the defensible way.** Not "a sub-1000 parameter
      detector" (we run a 304M ViT on every image and a judge will find that in
      ten seconds). Say: *769 trained parameters on a backbone you already run.*
      Where CLIP embeddings already exist, detection costs 0.32 us and a 3.5 KB
      file; the marginal cost of a new branch is zero inference; a new generator
      costs seconds, not GPU-hours. State the frozen-backbone limitation in the
      same breath — it is what makes the efficiency claim credible.
      `HANDOVER.md` §5f.8

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

**The model side is in good shape and is no longer where the risk is.** As of
28 Aug the shipped scorer has a measured operating point, a family-disjoint
eval, a robustness grid, a composed-degradation result, and every major trade
quantified rather than asserted. What it does *not* have is anything to show:
no demo, and no explainability. Both are judged, and one build — patch-level
scoring (Albert, Stage 2) — produces a heat map and fixes our largest failure
at the same time. That is the highest-leverage thing left.
