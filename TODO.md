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
- [x] Try MLP head; it underperformed the linear models on OOD
- [x] Compare linear alternatives; RidgeClassifier (`alpha=0.001`,
      `solver="lsqr"`) selected at 0.9221 clean / 0.8981 worst
- [ ] Try multi-crop embedding (PIPELINE §4.5) — cheapest untried upgrade
- [x] Freeze the winner into `data/models/general.npz`

### Albert — regularity / spectral
- [x] `quorum/detectors/spectral.py` — scorer over the existing 8 features
- [x] Per-variant AUC table — clean 0.7365 / worst 0.5596 (`blur10`)
- [x] Confirmed classifier changes do not materially improve spectral; retain it
      as a complementary low-level signal for fusion

### Kacey — face probe
- [ ] `quorum/detectors/face.py`, conditioning on `face_px`
- [ ] Per-variant AUC **and coverage** (detector loses 77% under `noise010`)

### Kacey — text
- [ ] Decide build or cut. **Recommendation: cut**, to protect fusion time.

---

## Stage 3 — Calibration and fusion — **Kacey**

The critical path. Nothing downstream works without it, and there is no partial
credit: a demo without fusion is a demo of one probe.

- [ ] Platt scaling per branch, fit on `calib` **only**
- [ ] Reliability diagrams to verify calibration
- [ ] CLIP zero-shot content label (face / animal / object / scene / text-heavy)
      — free, reuses the embedding
- [ ] `degradation_estimate` feature — fusion must distinguish "no artifacts
      found" from "could not measure"
- [ ] `quorum/fusion.py` — logistic regression over the 5-branch input vector
- [ ] Wire `predict.py` to fusion, replacing the `max()` placeholder
- [ ] Verify output contract exactly: `[{"image_path": ..., "pred": 0.87}]`

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
