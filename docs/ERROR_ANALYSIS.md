# Error Analysis Note

Required deliverable #5. Every number here is measured, none is asserted, and
each one names the script that reproduces it.

Scores come from `predict.score_embeddings` — the shipped scorer,
`max(general, 1.25 * tampered)` at `OPERATING_POINT = 0.8523`, so the threshold
below is always 0.5 on the emitted `pred`. Both constants changed on 30 Aug when
the probe was retrained (section 3.1); the figures import them from `predict.py`
rather than pasting them, so a figure cannot outlive the model it illustrates.

---

## 1. Where we stand

| eval set | n | AUROC | ACC | PREC | RECALL | F1 | FPR | FNR |
|---|---|---|---|---|---|---|---|---|
| **So-Fake-OOD clean** *(headline)* | 4,198 | 0.9255 | 0.8204 | 0.9255 | 0.6974 | 0.7954 | 0.0563 | 0.3026 |
| So-Fake-OOD, all 15 variants | 62,970 | 0.9193 | 0.7987 | 0.9276 | 0.6485 | 0.7634 | 0.0508 | 0.3515 |
| organizer set, clean | 8,719 | 0.9719 | 0.9221 | 0.9146 | 0.9016 | 0.9081 | 0.0626 | 0.0984 |
| organizer set, all 15 variants | 130,785 | 0.9589 | 0.8992 | 0.8945 | 0.8658 | 0.8799 | 0.0759 | 0.1342 |
| SID_Set tampered, clean | 3,595 | 0.9135 | 0.8428 | 0.8991 | 0.7018 | 0.7883 | 0.0563 | 0.2982 |
| SID_Set tampered, all 15 | 24,581 | 0.8924 | 0.6573 | 0.9917\* | 0.6306 | 0.7710 | 0.0563 | 0.3694 |
| **FOREIGN tampered, clean** | 5,096 | **0.7260** | 0.5412 | 0.8686 | **0.2600** | 0.4002 | 0.0563 | 0.7400 |
| **FOREIGN tampered, all 15** | 47,096 | **0.7348** | 0.3024 | 0.9905\* | 0.2726 | 0.4275 | 0.0563 | 0.7274 |

\* Those rows pit thousands of edited images against 2,096 reals. Precision and
accuracy there are imbalance artifacts; only AUROC, FPR and recall mean anything.

**Read the last two rows before quoting any of the others.** "FOREIGN tampered"
is So-Fake-OOD's own locally-edited class, embedded 30 Aug. It is the first
cross-dataset test the tampered branch has ever had, and it fails it -- see
section 7.5. Every SID_Set tampered number above is a *same-dataset* number.

**So-Fake-OOD is the honest headline** for the synthetic task. Different
datasets, different generator families, never trained on. The organizer set's
0.9719 is the friendlier number but it is WildFake DALL-E 3 against COCO
val2017 -- one generator against one photo corpus -- and the brief states it
"will not contribute to the final score", so it is a demonstration, not a claim.

**False accusations on real photography**, the number the operating point was
chosen to control:

```
COCO val2017 reals, clean          n= 5,000    FPR  6.26%
COCO val2017 reals, all 15 var     n=75,000    FPR  7.59%
```

Both improved from 8.90% / 10.02% when the probe was retrained on 30 Aug
(section 3.1) and the operating point re-derived by cross-validation.

---

## 2. Failure by transformation

The project's own thesis, so it goes first. AUROC across the organizers' 15-way
grid:

```
So-Fake-OOD     clean 0.9255   worst 0.8900 (noise01)   drop 0.0355
organizer set   clean 0.9719   worst 0.9145 (noise01)   drop 0.0574
```

Two things worth saying out loud:

- **Additive noise is the only transformation that really hurts.** `noise01` is
  worst on both sets, and every noise level ranks in the bottom four. Noise
  attacks the high-frequency band where generator artifacts live, and no
  amount of probe tuning recovers information that has been destroyed.
- **JPEG compression does not hurt at all.** On the organizer set `jpeg30`
  scores **0.9810**, *above* clean's 0.9719, and on So-Fake-OOD 0.9280 against
  0.9255. The transformation everyone assumes kills a detector is the one we are
  most robust to, because CLIP's own pretraining distribution is full of
  recompressed web images.

Full per-variant tables: `docs/robustness.md`, `docs/robustness-organizer_val.md`.
Figures: `docs/figures/robustness.png`, `robustness-organizer_val.png`.

---

## 3. Failure by generator

So-Fake-OOD `test_ood`, each generator family scored against all held-out reals.
The remaining five families (FLUX_2, Flux.1_pro, Ideogram2, Ideogram3,
Recraftv3) are the `calib_ood` carve and appear in no eval number.

| generator | n clean | AUROC | recall (clean) | recall (all 15) | **FNR** |
|---|---|---|---|---|---|
| GPT-image-2 | 123 | 0.8029 | 28.5% | 26.1% | **73.9%** |
| GPT-image-1.5 | 279 | 0.8413 | 40.1% | 35.4% | **64.6%** |
| nano_banana_2 | 85 | 0.8445 | 44.7% | 43.1% | **56.9%** |
| imagen3 | 227 | 0.9218 | 69.6% | 64.4% | 35.6% |
| Seedream3.0 | 280 | 0.9446 | 73.6% | 67.1% | 32.9% |
| seedream4.5 | 101 | 0.9503 | 74.3% | 68.3% | 31.7% |
| Imagen4 | 280 | 0.9570 | 78.6% | 72.0% | 28.0% |
| nano_banana | 136 | 0.9583 | 85.3% | 80.0% | 20.0% |
| GPT4o | 290 | 0.9679 | 84.5% | 79.3% | 20.7% |
| Hidream | 301 | 0.9681 | 86.7% | 83.9% | 16.1% |

**The failure tracks generator recency more than generator family** -- but the
pattern is weaker than it first looked, and the weakening is itself a result.

Two of three same-lab pairs still show the newer model beating us harder:
GPT-image-2 (73.9% FNR) against GPT4o (20.7%), and nano_banana_2 (56.9%) against
nano_banana (20.0%). The third **inverted** after the 30 Aug retrain:
seedream4.5 now scores 0.9503 against Seedream3.0's 0.9446, where before it was
0.8871 against 0.9328.

So the honest claim is narrower than "newer is always harder": *some* recent
generators are much harder, the effect is large where it appears (3-4x the FNR),
and it is not a fixed property of the generator -- more training breadth moved
one of the three across the line entirely. Reproduce: `scratchpad/errors.py`.

### 3.1 It is a distribution problem, not a backbone problem

Two hypotheses fit the table above. Either a frozen 2023 CLIP does not
*represent* a 2026 generator's artifacts — in which case no linear probe on top
can ever find them — or it does, and our probe was simply never shown them.

`calib_ood` is this project's one sanctioned training exception and holds five
generator families (FLUX_2, Flux.1_pro, Ideogram2, Ideogram3, Recraftv3) the
shipped probe has never seen. Adding them to `sid_train` — 2,044 images, variant
density matched so they cannot outvote 16,000 by carrying 3.75x the rows each —
and re-scoring `test_ood`, which is still never trained on:

| generator | shipped | +5 families | Δ |
|---|---|---|---|
| GPT-image-2 | 0.8047 | 0.8381 | **+0.0335** |
| nano_banana_2 | 0.8315 | 0.8935 | **+0.0620** |
| GPT-image-1.5 | 0.8339 | 0.8764 | **+0.0425** |
| seedream4.5 | 0.8856 | 0.9710 | **+0.0854** |
| imagen3 | 0.9382 | 0.9517 | +0.0135 |
| Seedream3.0 | 0.9472 | 0.9627 | +0.0155 |
| Imagen4 | 0.9613 | 0.9740 | +0.0127 |
| nano_banana | 0.9654 | 0.9719 | +0.0065 |
| GPT4o | 0.9687 | 0.9792 | +0.0105 |
| Hidream | 0.9701 | 0.9776 | +0.0075 |

Overall `test_ood` clean **0.9245 → 0.9469**, and the gain lands exactly where
the pain is: **+0.0559 on the four worst generators, +0.0111 on the six best**.

Three findings, each load-bearing:

- **The backbone is not the ceiling.** Train on one `calib_ood` family, test on
  held-out rows of that same family: 0.8385 – 0.9719. CLIP does encode these
  artifacts. The probe had not been shown them.
- **It is free on the false-positive axis.** COCO FP at matched 75% recall went
  **1.2% → 0.6%**, *down*. That is the opposite of §8.5, where adding real-photo
  diversity to the tampered branch took COCO FP from 13.6% to 53.5%. Generator
  diversity and real-photo diversity are different axes and behave differently.
- **Recency was not what fixed it.** The five families added are contemporaneous
  with or older than GPT-image-2 and nano_banana_2. The +0.056 on the newest
  generators came from **diversity alone**, which is the stronger result: the
  probe is learning something transferable rather than memorising fingerprints.

**Shipped 30 Aug** as `python -m quorum.detectors.general --plus`. The cost was
real and was paid: `calibrate()` fitted Platt scaling on `calib_ood` precisely
because it was family-disjoint, so spending it on training left the calibration
without a home. The fix needs no new download -- train on four families,
calibrate on the fifth, rotate, and average the five folded weight vectors.

That rotation also replaced the operating point. `OPERATING_POINT` is now
cross-validated: each fold picks the high end of its accuracy plateau on the one
family that fold's model never saw (0.585 / 0.600 / 0.705 / 0.665 / 0.635), and
the five are averaged to **0.8523**. It reads no evaluation set at all, and it
cut COCO false positives 8.90% -> 6.26%.

`scripts/pick_threshold.py` did NOT survive this and is marked invalid: it picks
on `calib_ood` as if held out, which is now training data.

Numbers in the table above are the **general branch alone** (ridge decision
values, 0.9245 baseline), not the shipped `max()`. Do not mix the two.
Reproduce: `scratchpad/more_data.py`, `scratchpad/newthr.py`.

### 3.2 A second dataset transferred nothing, which qualifies section 3.1

If breadth is what pays, more of it should pay again. WildFake's Midjourney
Advanced subset is a genuinely foreign corpus -- different dataset, different
generator, different capture pipeline. 1,500 images were fetched, embedded and
folded into the same recipe (`--plus --also wildfake_midjourney`).

| | calib_ood only | + Midjourney | Δ |
|---|---|---|---|
| **So-Fake-OOD clean** *(honest cross-dataset)* | 0.9268 | 0.9255 | **−0.0013** |
| **organizer set** | 0.9412 | **0.9719** | **+0.0307** |
| worst variant | 0.8937 | 0.8900 | −0.0037 |
| COCO false positives | 6.34% | 6.26% | −0.08pp |
| GPT-image-2 | 0.7963 | 0.8029 | +0.0066 |
| nano_banana_2 | 0.8493 | 0.8445 | −0.0048 |

**That is the signature of content matching, not artifact learning.** CLIP
zero-shot content buckets on the AI half of each source:

| | sid_train AI | organizer (DALL-E 3) | Midjourney |
|---|---|---|---|
| art / illustration | 43.8% | **72.4%** | **70.1%** |
| photographic | 47.1% | **15.8%** | **19.3%** |

Midjourney's content mix nearly matches the organizer set's, and the organizer
set is what improved. So-Fake-OOD, with a different mix, did not move.

Two consequences, and the second is uncomfortable:

- The +0.031 is on a benchmark the brief says **"will not contribute to the
  final score"**. It is a demonstration number, not a claim.
- **It qualifies section 3.1.** `calib_ood` and `test_ood` are the *same
  dataset* split by generator family, so some unknown fraction of that +0.0183
  was same-dataset transfer -- shared capture pipeline, resolution, JPEG history
  -- rather than pure generator breadth. A genuinely foreign dataset transferred
  **zero**. "Breadth beats recency" is therefore stated more confidently in
  section 3.1 than the evidence now supports.

Kept anyway: it costs nothing measurable and improves the reported benchmark.
But the honest cross-dataset number is unchanged, and the writeup must say so.
Reproduce: `scratchpad/mj_content.py`.

---

## 4. Failure by content

`SPEC.md` Phase 6 asks for per-content-bucket AUROC, on the grounds that a wide
swing means we are partly reading semantics rather than artifacts. Buckets are
CLIP's own zero-shot assignment over eight prompts, not ground truth — that
is fine for this purpose, and stated so nobody mistakes it for a labelled cut.

**Organizer set, clean** — spread 0.8676 … 0.9719, range **0.1043**

| bucket | n | %AI | AUROC | FPR | FNR |
|---|---|---|---|---|---|
| painting / art | 2,135 | 98.4% | 0.8676 | 23.5% | 15.0% |
| product / packaging | 488 | 28.3% | 0.8802 | 12.9% | 26.1% |
| photograph of a scene | 2,603 | 7.0% | 0.9017 | 6.9% | 24.3% |
| 3D render | 306 | 80.7% | 0.9091 | 25.4% | 9.7% |
| illustration / drawing | 1,456 | 40.7% | 0.9209 | 11.7% | 22.1% |
| text / signage heavy | 714 | 26.9% | 0.9437 | 6.7% | 20.8% |
| photo of a person | 555 | 39.1% | 0.9468 | 11.2% | 15.2% |
| landscape / nature | 462 | 10.8% | 0.9719 | 9.0% | 4.0% |

**So-Fake-OOD, clean** — spread 0.8723 … 0.9707, range **0.0985**

| bucket | n | %AI | AUROC | FPR | FNR |
|---|---|---|---|---|---|
| landscape / nature | 930 | 59.1% | 0.8723 | 26.6% | 15.1% |
| **text / signage heavy** | 589 | 48.6% | 0.8787 | 2.3% | **52.4%** |
| product / packaging | 237 | 60.3% | 0.8835 | 13.8% | 29.4% |
| painting / art | 228 | 52.2% | 0.9083 | 13.8% | 19.3% |
| photograph of a scene | 907 | 38.0% | 0.9083 | 7.3% | 29.6% |
| illustration / drawing | 535 | 65.8% | 0.9183 | 12.6% | 22.2% |
| photo of a person | 760 | 39.1% | 0.9707 | 2.4% | 15.2% |

Three readings, all uncomfortable and all worth publishing:

- **A ~0.10 AUROC range across content is real.** We are partly reading
  semantics. A detector purely reading generator artifacts would be flat here.
- **Text-heavy images are our worst false-negative bucket by a wide margin** —
  52.4% FNR on So-Fake-OOD against 15–30% everywhere else, with FPR of only
  2.3%. The model has effectively learned *"lots of text ⇒ real"*. This is the
  measured version of the two garbled-text false negatives in §6, and it
  vindicates the premise behind the text branch even though all three attempts
  to exploit it failed (§8).
- **Faces are our strongest bucket on both sets** (0.9468 / 0.9707). That is a
  quiet argument that a dedicated face branch is redundant, and it agrees with
  the measurement that adding one to `max()` is a wash (`HANDOVER.md` §5e).

Reproduce: `scratchpad/buckets.py`.

---

## 5. Representative false positives

![representative errors](figures/error-cases.png)

Ranked by how **stably** wrong they are — mean shipped score across all 15
degradation variants, not clean score. A borderline clean miss teaches nothing;
a photograph called fake under every transformation teaches a lot. All ten cases
in the figure are wrong in 15 of 15.

Case studies come from the organizer set because it is the only eval set whose
pixels are still on disk. Regenerate with `python scripts/error_cases.py`.

| # | image | pred | general | tampered | hypothesis |
|---|---|---|---|---|---|
| 1 | `000000006460.jpg` | 0.97 | −2.7 | **+5.0** | B&W surf photo with a large **"STB" graphic watermark** and a "© ZACK GINGG" byline composited on |
| 2 | `000000338191.jpg` | 0.96 | +1.5 | **+4.7** | **Nine-photo collage** of fire hydrants with hard black borders |
| 3 | `000000192191.jpg` | 0.92 | −1.3 | **+3.7** | Kitchen photo containing a **printed pizza-box lid** — a flat, saturated, machine-made graphic inside a natural scene |
| 4 | `000000314182.jpg` | 0.92 | −0.6 | **+3.8** | Flash-lit food bowls on white tile. The odd one out: no composite. Likely read as studio/render from the blown highlights and flat ground |
| 5 | `000000435081.jpg` | 0.89 | −0.2 | **+3.5** | **Sixteen-photo collage** of miniature clay food, plus a "PetitPlat" watermark. Subject matter is also genuinely artificial-looking |

**These are one failure mode, not five.** Four of the five contain a region that
did not come from the camera — a watermark, a collage border, a printed
graphic. And the branch attribution is unambiguous:

```
313 / 5,000 = 6.3% of COCO photographs are flagged
  the tampered branch is the higher of the two in 96.5% of them
  the general branch alone would flag only 0.4%
```

**The tampered branch is not malfunctioning. It is answering its question
correctly and we are asking it the wrong one.** It was trained on SID_Set class
2 — a real photograph with a locally edited region — so its learned concept is
*"this frame contains material that was not in the original capture."* A
watermark satisfies that. A collage satisfies that. A photograph of printed
packaging very nearly satisfies that. The gap between *composited* and
*AI-generated* is a labelling gap in the task, not a bug in the model.

This is the deepest problem in the system, and it survived four attempts to fix
it (§8). It is also the reason the branch is kept anyway: without it,
edited-photo AUROC collapses from 0.9035 to **0.5286**, a coin flip
(`HANDOVER.md` §5h).

---

## 6. Representative false negatives

| # | image | pred | hypothesis |
|---|---|---|---|
| 1 | `43575994…` | 0.02 | **B&W line-art comic illustration.** Not a photograph, so the photographic-artifact prior has nothing to grip |
| 2 | `830e6b1c…` | 0.02 | **Extreme blur, no high-frequency content.** Generator artifacts live in the band this image does not have |
| 3 | `97a51249…` | 0.02 | Photorealistic PS3 box shot. The giveaway is **garbled text — "FLY SWATTTER"** — which we do not read |
| 4 | `1ac74c7c…` | 0.02 | Game Boy Color render. Same mode: **"GAME BOYCoLORR", "GAME B_YLOR"** |
| 5 | `9fa67d65…` | 0.03 | **B&W manga illustration.** Same mode as #1 |

Three failure modes, in order of how much they cost us:

1. **Non-photographic style** (#1, #5). The probe was trained to separate
   *synthetic photographs* from *real photographs*. An illustration is neither.
   CLIP places it far from both training clusters and the linear boundary falls
   on the "real" side by default. This is the largest FN bucket in the figure
   and the cheapest to fix — a style gate that routes illustrations to a
   different decision would help, and we have not built one.
2. **Garbled text in a photorealistic render** (#3, #4). A human spots these
   instantly. We measured the bucket at **52.4% FNR** (§4), so this is not two
   unlucky images. Three separate attempts to read text failed (§8), which
   makes this our best-understood and most stubborn miss.
3. **No high-frequency content** (#2). Nothing to read, and nothing to do about
   it. Correctly abstaining would be better than a confident 0.02, and we do not
   currently express that.

---

## 7. Trade-offs, all measured

Four, each a real cost we chose to pay, with the measurement that priced it.

**7.1 False positives against recall.** `OPERATING_POINT` is the only lever that
moves these metrics by more than a rounding error.

| cut | ACC | PREC | RECALL | F1 | COCO FP |
|---|---|---|---|---|---|
| 0.500 (sigmoid default) | 0.8306 | 0.7840 | 0.9134 | 0.8438 | 26.5% |
| **0.8523 (shipped)** | 0.8204 | **0.9255** | 0.6974 | 0.7954 | **6.3%** |

Precision +0.14 and false accusations cut **4.2x**, paid for with −0.22 recall
and −0.048 F1. That is a steeper trade than the previous probe made (+0.11
precision for −0.14 recall), because the cross-validated cut sits further right
than the old hand-picked 0.766. Section 3.1 gives the alternative cut, 0.8057,
which holds the old policy instead and strictly dominates the model it replaced.
The premise is that at a realistic base
rate most uploads are genuine, so a libelled photograph is the expensive error,
and F1 weights a missed fake and a libelled photograph equally. Argue with the
premise, not the constant. `predict.py`, `HANDOVER.md` §5f.6.

**7.2 Transform-robustness against edit-robustness.** The two readings of
"robust" disagree, and both are real. Dropping the tampered branch wins four of
five synthetic-vs-real metrics and cuts COCO FP 8.9% → 2.9%, but takes
edited-photo AUROC 0.9035 → 0.5286 and recall 74.6% → 10.7%. Keeping it makes
the organizer-set degradation drop 6x worse (0.0121 → 0.0773). We kept it,
because a detector that cannot survive editing is not the one this brief asks
for. The full counterfactual is regenerated in `docs/figures-no-tampered/`.
`HANDOVER.md` §5h. **Read 7.5 before quoting any of those numbers** -- they are
all measured on SID_Set, and the branch does not survive leaving it.

**7.3 Generalisation against in-distribution ceiling.** A single probe scores
**0.9987** AUROC on held-out in-domain data (`sid_calib`, clean) and **0.9255**
on unseen generator families — a 0.07 drop for changing dataset alone. We
optimise for the second and publish it as the headline. The corollary is §3: on
the newest generators we are at 0.76–0.80, and no in-distribution number would
have revealed that.

**7.4 `max()` against a learned combiner.** Fusion over the calibrated branch
vector scores lower than `max` on both eval sets and by more on the pooled task,
so `max` stays. A disjunctive task — *either* branch firing means "AI touched
it" — is exactly the shape a linear combiner handles worst. The margin is now
small enough (~0.0014 clean) that this is a judgement call rather than a rout.
`HANDOVER.md` §5c/§5e holds the measurement.

> **Two stale constants in `predict.py`'s docstring**, both found on 30 Aug
> while writing this note. Neither changes a decision; both would embarrass us
> if a judge re-ran them.
>
> | claim in `predict.py` | re-measured 30 Aug |
> |---|---|
> | `max` on So-Fake-OOD clean/worst = 0.9189 / 0.8921 | **0.9085 / 0.8731** (`test_ood`), 0.9069 / 0.8712 incl. `calib_ood` |
> | "the F1 argmax is 0.506" | **0.532**, F1 0.8295 |
>
> The first predates the ridge probe; both models moved together so the `max`
> vs fusion ranking is unaffected. The second is close enough that the
> "0.5 is nearly F1-optimal" argument still stands. Fix the docstring, do not
> quote the old numbers.

---

**7.5 The tampered branch does not transfer, and every number we quote for it
is a same-dataset number.**

This is the most serious limitation in the project and it was invisible until
30 Aug, because the branch trains on SID_Set class 2 and every evaluation we had
was SID_Set's own eval split -- same corpus, same editing pipeline. So-Fake-OOD
carries its own `TAMPERED` class which `stream_embed.py` silently skips unless
`--tampered` is passed, so it had never been embedded. 3,000 of them now are.

Negatives are So-Fake-OOD reals in both columns, so the eval pairing is
within-dataset and only the MODEL is crossing datasets:

| scorer | SID_Set tampered *(in-dataset)* | So-Fake-OOD tampered *(FOREIGN)* |
|---|---|---|
| general alone | 0.5399 | 0.7100 |
| **tampered branch alone** | **0.9528** | **0.6251** |
| max() -- shipped | 0.9135 | 0.7260 |
| recall @ operating point, tampered alone | 70.1% | **7.8%** |
| recall @ operating point, max() | 70.2% | **26.0%** |

Three readings, in descending order of how much they should worry us:

1. **The branch collapses.** 0.9528 -> 0.6251 AUROC, and recall at the shipped
   operating point falls from 70.1% to 7.8%. It learned SID_Set's particular
   inpainting pipeline, not "editing".
2. **The ranking inverts.** On foreign edits the *general* branch (0.7100) beats
   the *tampered* branch (0.6251) -- the branch built for the task is the worse
   of the two at it. The complementarity argument in `HANDOVER-MODELS.md` §11
   holds only inside SID_Set.
3. **It partly reopens 7.2.** On foreign edits `max()` beats general-alone by
   just +0.016, while the tampered branch causes **89.4% of our COCO false
   positives** (section 5). On SID_Set it is still decisive, 0.5399 -> 0.9135.
   So the branch earns its place on one dataset and very nearly nothing on the
   other, and which you believe depends on whether SID_Set or So-Fake-OOD better
   represents real-world editing. We have no basis for that judgement, and
   saying so is more honest than picking the flattering one.

#### Why: it learned a dataset, not a concept

The collapse has a mechanism, and it is more useful than the collapse. Median
branch scores by population, clean images only:

| population | tampered branch | general branch |
|---|---|---|
| SID reals (trained on) | 0.036 | 0.031 |
| SID edits (trained on) | 0.917 | 0.120 |
| SID edits (held out) | 0.924 | 0.136 |
| foreign reals | 0.090 | 0.102 |
| **foreign edits** | **0.176** | **0.412** |
| foreign fully-synthetic | 0.146 | 0.936 |

How far editing moves each branch, *within* each corpus:

| | tampered branch | general branch |
|---|---|---|
| SID reals → SID edits | **+0.888** | +0.105 |
| foreign reals → foreign edits | **+0.087** | +0.310 |

**The branch responds to SID edits ten times more strongly than to foreign
edits.** On So-Fake-OOD it is nearly flat.

**And 0.036 → 0.924 is too clean to be real.** Finding a locally-edited region
in an otherwise-authentic photograph is a hard problem. A linear probe on frozen
CLIP that separates it near-perfectly has not solved it -- it has found a
**construction artifact of SID_Set's tampered split**: one inpainting model, one
re-encode pipeline, some global signature every image in that class shares. The
branch learned *"was this produced by SID_Set's tampering process"*, not *"was
this edited"*.

Two independent corroborations:

- **Foreign fully-synthetic images score 0.146** on the tampered branch, barely
  above foreign reals at 0.090. It is not firing on synthetic content in
  general; it is narrowly tuned to something SID-specific.
- **It finally explains §8.5**, the result that never made sense: retraining the
  branch on more real-photo diversity made COCO false positives *worse*
  (13.6% → 53.5%) while its own AUROC *rose* (0.9528 → 0.9884). If the branch
  fits a dataset signature rather than a concept, more real diversity sharpens
  that signature and generalises nothing. That is precisely what was measured,
  and we had no explanation for it until now.

**The inversion has its own cause.** The general branch moves +0.310 on foreign
edits against only +0.105 on SID edits, because So-Fake-OOD's tampered images
contain visibly *generated* material (median general score 0.412, far above
foreign reals' 0.102) while SID_Set's edits are nearly invisible to a
synthetic-content detector. The two corpora mean different things by "tampered":
So-Fake-OOD pastes in generated content, SID_Set does something subtler that
leaves a pipeline fingerprint.

So this is not a weak edit-detector. It is a strong detector of **one dataset's
editing pipeline**, which happens to be the only pipeline it was ever tested on.

Unverified: SID_Set pixels are not on disk, so we cannot confirm whether the
artifact is a re-encode, a resize, or the inpainting model itself. Re-streaming
a sample of `sid_tampered` with `--save-images` would settle it.
Reproduce: `scratchpad/why_tamp.py`.

**Not a reason to drop it.** `max()` is still the better scorer on both datasets,
and dropping the branch returns the edited task to a coin flip on SID_Set. It IS
a reason to stop quoting 0.9035 and 0.9528 without the qualifier.

Reproduce: `scratchpad/tamp_transfer.py`. The set is `so_fake_tampered_eval`,
3,000 images, all 15 variants, `test_ood` split -- never trained on.

---

## 8. What we tried that did not work

Negative results, kept because they are the evidence that the shipped design is
a decision rather than a default.

| # | attempt | result |
|---|---|---|
| 1 | OCR features (6-d) for garbled text | Transfers at **0.4627**, below chance; five of six features flip sign across datasets |
| 2 | CLIP on warped text crops | Transfers at 0.8083 but worth **+0.0022**; moves the text-heavy gap −0.0420 → −0.0409 |
| 3 | Text crops across the 15-variant grid | Clean 0.8284 → worst **0.5229**; detection collapses to 48.2%, and missingness is label-correlated 3.63:1 (AI text is 60px median, real text 19px, against an 8px OCR floor) |
| 4 | Text consistency, as a concept | 2 of 6 sampled real COCO text regions are photographer watermarks. Any such detector answers *"was this text composited?"*, not *"was this AI-generated?"* — the same confusion as §5 |
| 5 | Retrain tampered on more real-photo diversity | COCO FP got **worse**, 13.6% → 53.5%, while its own AUROC rose 0.9528 → 0.9884. Capacity and data are not the lever. **Explained in §7.5**: the branch fits a SID_Set construction artifact, so more real diversity sharpens the artifact and generalises nothing |
| 6 | Face branch into `max()` | Wash: 0.9161/0.8856 against 0.9153/0.8860 |
| 7 | Face + spectral into `max()` | Clearly worse: 0.8914/0.8135 |
| 8 | Fusion meta-classifier | Loses on both sets (§7.4) |
| 9 | **Patch self-consistency** | Mechanism confirmed, gate failed — below |
| 10 | Per-generator specialist zoo | Five one-generator probes combined by `max()`: **0.9042** against one pooled probe's **0.9444** on identical rows. Loses by 0.040, and loses *most* on GPT-image-2 (0.763 vs 0.848) — the generator specialists were supposed to help |
| 11 | Nonlinear head (MLP-64) on the pooled data | 0.9425 against linear's 0.9444, with double the COCO false positives. Tests the same premise as #10 from the other side: one linear boundary is **not** the bottleneck, data breadth is |
| 12 | A second foreign dataset (Midjourney, 1,500 imgs) | +0.0307 on the organizer set, **−0.0013** on So-Fake-OOD. Content matching, not artifact learning — §3.2 |

**§8.10 and §8.11 together are the answer to "why not a mixture of experts?"**
Splitting by *generator* starves each slice of the shared notion of "synthetic"
that transfers; the pooled probe beats the zoo by 0.040 and beats it by most on
the hardest generator. Split branches by **task** (`general` / `tampered` are two
different questions with different label spaces), never by generator. Reproduce:
`scratchpad/specialists.py`.

**§8.9 in full**, because it is the most recent and the most instructive.
Score the nine cells of a 3×3 grid separately with the existing general probe
and use their disagreement. On composited images the whole-image score is
**0.5189** — blind, a coin flip — while patch max reaches **0.7982**, with zero
new parameters. The mechanism is real. But it lost its gate: at matched recall
it gives 9.1% COCO false positives against the shipped scorer's 6.8%, adding it
to `max()` is a wash, and the hypothesis that patch statistics would *transfer*
better was disproved (0.9830 against 0.9848). It costs 9x inference and is not
shipped in the scorer.

One result from it is worth keeping: **`argmax` over the nine cells finds the
edited cell 73.5% of the time against 11.1% chance** (top-3: 86.1%). That is a
working explainability heat map from the probe we already ship. It belongs in
the demo, off the scoring path, and labelled as illustrative — 73.5% is measured
on hard-edged composites and will be lower on a real inpaint.
Reproduce: `scratchpad/patch_build.py`, `patch_eval.py`.

---

## 9. Limitations of this note

- **Case studies come from the organizer set only.** So-Fake-OOD and SID_Set
  were streamed and embedded without saving pixels, so their errors can be
  counted but not looked at. Sections 3, 4 and 7.5 cover them; 5 and 6 do not.
  This is why the tampered-branch collapse in 7.5 has numbers but no faces.
- **The content buckets in section 4 predate the 30 Aug retrain.** The AUROC
  values there belong to the previous probe. The qualitative findings hold --
  bucket spread is real, text-heavy is the worst FNR bucket -- but do not quote
  those specific numbers alongside section 1's.
- **Content buckets are CLIP's opinion of content**, assigned by zero-shot
  argmax over eight prompts. They are a lens, not labels. Section 3.2 leans on
  them for the Midjourney conclusion, and that conclusion is only as good as the
  lens.
- **The `composite` population in 8.9 is a proxy** for locally-edited images --
  one grid cell of a COCO photo overwritten by a WildFake image — because no
  SID_Set tampered pixels are on disk. Deliberately an upper bound: if the method
  cannot see a hard-pasted foreign cell it will never see an inpaint. Now that
  `so_fake_tampered_eval` exists the patch gate could be re-run against real
  foreign edits, and has not been.
- **The SID_Set artifact in 7.5 is inferred, not identified.** The evidence that
  one exists is strong -- a 10x difference in edit response and near-perfect
  in-corpus separation on a hard task -- but we have not seen the pixels, so we
  cannot say whether it is a re-encode, a resize, or the inpainting model's own
  signature. Re-streaming `sid_tampered` with `--save-images` would settle it and
  has not been done.
- **We cannot say which tampered distribution is representative.** Section 7.5
  reports that the branch works on SID_Set and largely fails on So-Fake-OOD. We
  have no third dataset to break the tie, and no basis for claiming either is
  closer to how photographs are edited in the wild.
- **WildFake contains near-duplicates.** Two of the top five false negatives
  were the same picture at 45KB and 1.4MB, cosine 0.9771, both in eval. The
  figure de-duplicates at 0.95; the aggregate numbers do not.
- **Nothing here is a claim about the organizer set's score.** The brief states
  that subset "will not contribute to the final score"; its 0.9719 is reported
  because it is externally comparable, not because it counts.
