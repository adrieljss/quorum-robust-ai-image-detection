# Quorum — Albert's branch plan

Working plan for the `general-spectral` branch. The general and spectral
branches are measured separately; tampered remains unchanged.

## Status

| branch | selected model | status |
|---|---|---|
| general | RidgeClassifier, `alpha=0.001`, `solver="lsqr"` | **merged 29 Aug**, Platt folded into the weights |
| spectral | LogisticRegression over 8 existing features | **merged**, and measured OUT of the combiner |
| tampered | existing logistic probe | unchanged |

## General probe

The MLP and PCA variants did not improve transfer to So-Fake-OOD. Ridge was the
best single-branch model tested:

```
general Ridge   clean 0.9246   worst 0.9014 (noise002)   drop 0.0231
```

The baseline quoted here as `0.9125 / 0.8798` is stale -- it predates the
`calib_ood` leak fix. Main's logistic probe was **0.9170 / 0.8848**, so the real
gain is +0.0076 clean and +0.0165 worst, not +0.0121 / +0.0216. Still a gain,
and the drop improves 0.0321 -> 0.0232, which is the headline metric.

**The separate threshold policy was never implemented, and that broke the
deliverable.** `predict.py` shifts its sigmoid by a fixed `OPERATING_POINT`,
so it reads the saved weights as log-odds. Ridge decision values are not
log-odds. Shipped raw, this probe scored **0.5143 accuracy and 0.0561 recall**.

Fixed on merge by folding Platt scaling into the coefficients
(`general.calibrate`), which is exact because Platt of a linear model is
linear. `predict.py` and `OPERATING_POINT` are unchanged. Result at the same
0.766 cut: **0.9085 / 0.8731 AUROC, acc 0.8247, prec 0.8816** -- better than
the logistic probe on all four. HANDOVER.md 4c has the alternatives.

## Spectral branch

The feature extraction already exists in `quorum/features.py`; the scorer is
implemented in `quorum/detectors/spectral.py` and evaluated through
`scripts/eval_grid.py`.

Current held-out result:

```
spectral        clean 0.6739   worst 0.5471 (noise01)   drop 0.1268
```

Classifier comparisons (logistic, scaled logistic/SVM/Ridge, and shrinkage
LDA) all remain near the same result. Therefore the limitation is the feature
signal, not the classifier. Blur, resize, and noise destroy the high-frequency
information by design. That conclusion is right and it is worth stating as a
*result*: classical frequency forensics do not survive the transforms real
images go through, which is the argument for the CLIP probe.

**But it is not a complementary fusion feature -- it is subtractive.** Adding
it costs 0.076 worst-case under `max`, and it still loses under a learned
combiner fitted on the carve, i.e. given its best shot:

```
max(general, tampered)              clean 0.8997   worst 0.8532
max(general, tampered, spectral)    clean 0.8868   worst 0.7770
LR(general, tampered)               clean 0.9148   worst 0.8819
LR(general, tampered, spectral)     clean 0.9073   worst 0.8729
```

`predict.py` does not call it, and should not. HANDOVER.md 4d.

Also fixed on merge: `evaluate()` bypassed the `calib_ood` carve and read
0.7362 clean against a true 0.6736, and the 25 all-zero feature vectors in
`spec_so_fake_ood` are now counted by the module instead of sitting open.

## Open work

- [x] Regenerate and save the final general model to `data/models/general.npz`
      -- done on merge, with the Platt fold applied.
- [x] Provide general/spectral branch scores to the fusion owner --
      `fusion.fit_branches()` now calls `fit_general`, so the combiner table no
      longer compares fusion against a probe nobody ships.
- [ ] ~~Multi-crop on H100~~ -- **do not.** Measured 29 Aug and it does not
      work: patch self-consistency scores 0.5220 alone and makes the general
      probe worse in combination (0.7516 vs 0.7542). HANDOVER.md 4b.
- [ ] `max` now beats fusion by only **+0.0014** (0.9189/0.8921 vs
      0.9175/0.8905), down from +0.0086. The better general branch narrowed it.
      Re-check before anyone repeats "max beats fusion" as settled.
