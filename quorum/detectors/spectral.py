"""Spectral artifact scorer over cached spectral feature vectors."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quorum.detectors.general import auc_by_variant, fit, load


def train(train_source="spec_sid_train"):
    X, rows = load(train_source)
    return fit(X, rows.label.values), rows


def evaluate(eval_source="spec_so_fake_ood", train_source="spec_sid_train"):
    clf, _ = train(train_source)
    X, rows = load(eval_source)
    return auc_by_variant(clf, X, rows)


if __name__ == "__main__":
    scores = evaluate()
    print(scores.to_string(float_format="%.4f"))
    print(f"\nclean {scores['clean']:.4f}  worst {scores.min():.4f} "
          f"({scores.idxmin()})  drop {scores['clean'] - scores.min():.4f}")
