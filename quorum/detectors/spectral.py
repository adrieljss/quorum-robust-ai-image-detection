"""Spectral artifact scorer over cached spectral feature vectors.

Eight frequency-domain features from quorum/features.py, one logistic probe,
nine trained parameters. It is the weakest branch and that is the point: it
reaches 0.6736 clean and collapses to 0.5471 under noise01, where the frozen
CLIP probe holds 0.9245/0.9013 on the same grid. Classical frequency forensics
do not survive the transforms real images go through -- which is the argument
for the rest of the system, not a branch that underperformed.

**It must not enter the combiner.** Measured on So-Fake-OOD, held out:

    max(general, tampered)              clean 0.8997   worst 0.8532
    max(general, tampered, spectral)    clean 0.8868   worst 0.7770
    LR(general, tampered)               clean 0.9148   worst 0.8819
    LR(general, tampered, spectral)     clean 0.9073   worst 0.8729

It loses under max() -- which inherits the worst branch by construction, and a
near-chance branch just pushes real photographs upward -- and it still loses
under a learned combiner fitted on the carve, i.e. given its best shot.
predict.py does not call this, deliberately.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from quorum.detectors.general import auc_by_variant, fit, load

CALIB_SPLIT = "calib_ood"


def train(train_source="spec_sid_train"):
    X, rows = load(train_source)
    return fit(X, rows.label.values), rows


def evaluate(eval_source="spec_so_fake_ood", train_source="spec_sid_train"):
    """Held out. so_fake_ood carries the calib_ood rows the calibrators are fitted
    on; scoring them here reads 0.7362 clean against a true 0.6736."""
    clf, _ = train(train_source)
    X, rows = load(eval_source)
    keep = (rows.split != CALIB_SPLIT).values
    return auc_by_variant(clf, X[keep], rows[keep].reset_index(drop=True))


def zero_rows(source="spec_so_fake_ood"):
    """Feature vectors that came out all zero -- 25 in spec_so_fake_ood, every
    one a real image. features.py returns them silently, so count them rather
    than discover them as a shrug in the AUC."""
    X, rows = load(source)
    return rows[np.abs(X).sum(axis=1) == 0]


if __name__ == "__main__":
    scores = evaluate()
    print(scores.to_string(float_format="%.4f"))
    print(f"\nclean {scores['clean']:.4f}  worst {scores.min():.4f} "
          f"({scores.idxmin()})  drop {scores['clean'] - scores.min():.4f}")

    z = zero_rows()
    print(f"all-zero feature vectors: {len(z)}  labels {sorted(set(z.label))}"
          if len(z) else "all-zero feature vectors: none")

    # The check: this branch is only ever reported, never combined. If someone
    # wires it into the combiner these numbers say what it costs.
    assert scores["clean"] < 0.75, f"clean {scores['clean']:.4f} -- carve leaking again?"
    assert scores.min() > 0.5, f"worst {scores.min():.4f} is at or below chance"
    print("ok")
