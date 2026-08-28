"""General probe: logistic regression on frozen CLIP embeddings.

Trains on sid_train, reports AUC per degradation setting on any eval source.
The whole point of the project is the *spread* of those numbers, not the mean.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import roc_auc_score

from quorum.embed import load_source

MODELS = Path(__file__).resolve().parents[2] / "data" / "models"
MODEL = MODELS / "general.npz"
MODEL_TAMPERED = MODELS / "tampered.npz"


MANIFEST = MODELS.parent / "manifests" / "main.csv"


@lru_cache(maxsize=1)
def _resolved():
    """image_id -> the split build_manifest.py resolved it to.

    The manifest applies a priority rule (eval beats train) when one image turns
    up in two sources. The cache does not: it keeps whatever each shard wrote.
    SID_Set ships 43 images in both its tampered train and validation splits, so
    a loader that trusts the shard trains a probe on 80 rows of its own eval set.
    Small -- it inflated the tampered probe 0.9440 -> 0.9464 -- but it is exactly
    the contamination that does not look like a bug.
    """
    if not MANIFEST.exists():
        print(f"WARNING: no {MANIFEST} -- cross-split dedupe is OFF; "
              f"run scripts/build_manifest.py before trusting any number")
        return {}
    # by name, not position: usecols returns columns in FILE order, so a
    # positional zip here silently builds the map backwards and filters nothing.
    m = pd.read_csv(MANIFEST, usecols=["image_id", "source", "split"])
    m = m.drop_duplicates("image_id")
    return dict(zip(m.image_id, zip(m.source, m.split)))


def load(source: str):
    """(X, rows) with re-embedded duplicates and cross-split leaks dropped.

    Two distinct problems: a restarted run re-draws the same shuffle order, so
    one image can land in two shards of the SAME source; and an image can appear
    in two sources on opposite sides of the train/eval line. The manifest owns
    the second, so defer to it rather than re-deriving the priority rule here.
    """
    X, R = load_source(source)
    keep = ~R.duplicated(subset=["image_id", "variant"])

    resolved = _resolved()
    if resolved:
        # face_/spec_ are branch caches of the same images; the manifest holds one
        # row per image under the bare source name.
        base = source.removeprefix("face_").removeprefix("spec_")
        owner = R.image_id.map(resolved)
        held = owner.map(lambda v: v[0] if isinstance(v, tuple) else None)
        keep &= (held.isna() | (held == base)).values        # unseen id -> keep

        # Adopt the manifest's split rather than the shard's. The shard records
        # what the embedding pass was told; the manifest records what the split
        # actually resolved to after dedupe and the calib_ood carve. When they
        # disagree the manifest is right, and silently keeping the stale label is
        # how a calibration slice ends up back in the eval set.
        R = R.copy()
        R["split"] = owner.map(lambda v: v[1] if isinstance(v, tuple) else None).fillna(R.split)
    return X[keep.values].astype(np.float32), R[keep].reset_index(drop=True)


def fit(X, y):
    return LogisticRegression(max_iter=2000).fit(X, y)


def fit_general(X, y):
    """Selected general-probe model; spectral and tampered use ``fit``."""
    return RidgeClassifier(alpha=0.001, solver="lsqr").fit(X, y)


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
    # The check that would have caught the 43-image SID_Set leak: no image may
    # sit on both sides of the train/eval line once load() has filtered.
    seen = {}
    for s in ("sid_train", "sid_tampered", "sid_calib", "so_fake_ood",
              "sid_tampered_eval", "organizer_val"):
        seen[s] = set(load(s)[1].image_id)
    for a in seen:
        for b in seen:
            if a < b:
                assert not (seen[a] & seen[b]),                     f"{len(seen[a] & seen[b])} images shared by {a} and {b}"
    print(f"splits disjoint: {sum(map(len, seen.values())):,} images across {len(seen)} sources")

    Xtr, Rtr = load("sid_train")
    print(f"general: train {Xtr.shape}  {Rtr.label.value_counts().to_dict()}")
    clf = fit_general(Xtr, Rtr.label.values)
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
