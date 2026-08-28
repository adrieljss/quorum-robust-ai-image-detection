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
- [x] Decide build or cut. **Cut.** Nine hours of OCR for a 7-parameter model
      against fusion being the critical path. Both slots stay in the fusion
      vector at neutral fill, so wiring it back later is one line

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
