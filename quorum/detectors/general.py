"""General probe: logistic regression on frozen CLIP embeddings.

Trains on sid_train, reports AUC per degradation setting on any eval source.
The whole point of the project is the *spread* of those numbers, not the mean.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from quorum.embed import load_source

MODELS = Path(__file__).resolve().parents[2] / "data" / "models"
MODEL = MODELS / "general.npz"
MODEL_TAMPERED = MODELS / "tampered.npz"


def load(source: str):
    """(X, rows) with re-embedded duplicates dropped -- restarted runs re-draw
    the same shuffle order, so the same image can land in two shards."""
    X, R = load_source(source)
    keep = ~R.duplicated(subset=["image_id", "variant"])
    return X[keep.values].astype(np.float32), R[keep].reset_index(drop=True)


def fit(X, y):
    return LogisticRegression(max_iter=2000).fit(X, y)


def auc_by_variant(clf, X, R):
    p = clf.decision_function(X)
    out = {v: roc_auc_score(R.label[m], p[m])
           for v, m in ((v, (R.variant == v).values) for v in R.variant.unique())
           if R.label[m].nunique() == 2}
    return pd.Series(out).sort_values()


def save(clf, path=MODEL):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, w=clf.coef_, b=clf.intercept_)


def train_tampered():
    """Tampered vs REAL only -- never sees a fully-synthetic image.

    Kept separate on measurement, not taste: folding tampered into the general
    probe costs it 0.9124 -> 0.7920 on So-Fake-OOD while only reaching 0.7831
    on tampered, where a dedicated probe reaches 0.9521. One linear boundary
    cannot serve both tasks. Fusion combines them instead.
    """
    Xa, Ra = load("sid_train")
    Xb, _ = load("sid_tampered")
    m = (Ra.label.values == 0)
    X = np.concatenate([Xa[m], Xb])
    y = np.r_[np.zeros(m.sum()), np.ones(len(Xb))]
    return fit(X, y), X, y


if __name__ == "__main__":
    Xtr, Rtr = load("sid_train")
    print(f"general: train {Xtr.shape}  {Rtr.label.value_counts().to_dict()}")
    clf = fit(Xtr, Rtr.label.values)
    save(clf)

    try:
        tclf, Xt, yt = train_tampered()
        save(tclf, MODEL_TAMPERED)
        Xe, Re = load("sid_tampered_eval")
        Xo, Ro = load("so_fake_ood")
        real = Xo[(Ro.variant == "clean").values & (Ro.label.values == 0)]
        pe = tclf.decision_function(Xe[(Re.variant == "clean").values])
        pr = tclf.decision_function(real)
        auc = roc_auc_score(np.r_[np.ones(len(pe)), np.zeros(len(pr))], np.r_[pe, pr])
        print(f"tampered: train {Xt.shape}  held-out vs real AUC {auc:.4f}")
    except FileNotFoundError as e:
        print(f"tampered probe skipped: {e}")   # run --tampered passes first

    for src in ("sid_calib", "so_fake_ood"):
        X, R = load(src)
        a = auc_by_variant(clf, X, R)
        print(f"\n=== {src}  ({R.image_id.nunique()} imgs) ===")
        print(a.to_string(float_format="%.4f"))
        print(f"clean {a.get('clean', float('nan')):.4f}  "
              f"worst {a.min():.4f} ({a.idxmin()})  drop {a.get('clean', 0) - a.min():.4f}")
