# Quorum — Albert's branch plan

Working plan for the `general-spectral` branch. The general and spectral
branches are measured separately; tampered remains unchanged.

## Status

| branch | selected model | status |
|---|---|---|
| general | RidgeClassifier, `alpha=0.001`, `solver="lsqr"` | active/final candidate |
| spectral | LogisticRegression over 8 existing features | complete, complementary |
| tampered | existing logistic probe | unchanged |

## General probe

The MLP and PCA variants did not improve transfer to So-Fake-OOD. Ridge was the
best single-branch model tested:

```
general Ridge   clean 0.9246   worst 0.9014 (noise002)   drop 0.0231
```

Compared with the original logistic baseline (`0.9125 / 0.8798`), Ridge
improves both clean and worst-case AUROC. Smaller alpha values were effectively
tied; `alpha=0.001` is retained as the stable choice with the `lsqr` solver.

The thresholded metrics use a separate policy: the threshold is the 99th
percentile of clean-real `sid_calib` scores, corresponding to approximately 1%
calibration false-positive rate. AUROC remains the primary comparison metric.

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
information by design. Keep spectral as a complementary fusion feature rather
than optimizing it as a standalone detector.

## Open work

- [ ] Regenerate and save the final general model to `data/models/general.npz`.
- [ ] Optional: run multi-crop only on H100; it is not required for the selected
      result and remains blocked by the local CUDA/cuDNN environment.
- [ ] Provide general/spectral branch scores to the fusion owner.
