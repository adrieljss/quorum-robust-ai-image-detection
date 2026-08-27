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

### Second push — nobody is blocked on these
- [x] `organizer_val` face + spectral pass — 75,000 spectral / 3,941 face rows
- [x] ~~`faces/` dataset~~ — **CUT.** Face probe saturates at ~500 images
      (500 -> 0.8892 OOD clean, all 4,128 -> 0.8952). 5.4 GB and a `.rar` to
      feed a model that plateaued eight times ago. Coverage/demographic
      diversity is a *limitations* note, not a download.
- [x] Re-run `build_manifest.py` — 330,851 rows, assertions A–F green
- [ ] `push_cache.py` again — `pull_cache.py` picks up the delta

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
- [ ] Explain why face AUC *rises* under blur (0.9382 → 0.9500 at `blur20`).
      Not survivorship: `blur20` retains 99.7% of faces. Cause unknown

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
- [ ] Wire `predict.py` to fusion — **deferred, not cancelled.** Blocked on the
      calibration slice below. Measured on so_fake_ood clean/worst: raw `max()`
      0.9042 / 0.8634, fusion 0.8587 / 0.8340, so swapping it in *today* costs
      ~6 points. `predict.py` keeps `max()` and its docstring records why

### Blocked on Adriel — the one thing that unblocks fusion

- [ ] **Carve a generator-disjoint calibration slice from So-Fake-OOD**, own
      `source` and `--assign-split`. Touches assertions A and B, so it is
      Adriel's call — move this into Stage 1 if that fits better

Fusion beats general only when calibrated on generator-diverse data: +0.0042
clean and +0.0055 worst, positive on all 5 random generator partitions. Fitted on
`sid_calib` it *loses*, because that split shares generators with train and every
branch is saturated there, so Platt fits an extreme slope that manufactures
over-confidence. Measured collapse, sorted by saturation:

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

- [ ] `scripts/eval_grid.py` — clean vs all 14 transforms, per branch and fused
- [ ] **Robustness Evaluation Summary** table (required deliverable)
- [ ] Reference number on `organizer_val` — **blocked, and hard-blocked.**
      COCO val2017 is 100% real, so the organizer set has no positive class and
      **no AUROC can be computed at all** until WildFake DALL·E Advanced lands.
      This is the only externally-comparable number we get. Chase it.
- [ ] Per-content-bucket AUROC — wild variance means we are partly reading
      semantics, and saying so is a strength

---

## Stage 5 — Error analysis — **Adriel** (required deliverable)

Feeds Innovation & Problem Insight at 20%. Not an afterthought.

- [ ] 3–5 representative **false positives**, each with a hypothesis
- [ ] 3–5 representative **false negatives**, same
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

**Kacey's fusion is the bottleneck.** If schedule slips, cut the text branch and
`provenance.py` first — both are already flagged as cut candidates in SPEC.

**Michael and Valentino are not blocked by anyone.** The demo can be fully built
against stubbed scores and wired to the real model in an afternoon. Treat any
week where the demo has not progressed as a scheduling failure, not a dependency.
