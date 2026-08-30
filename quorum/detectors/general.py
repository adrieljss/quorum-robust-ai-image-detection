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
    """Selected general-probe model; spectral and tampered use ``fit``.

    Ridge beats logistic on rank quality here -- 0.9245/0.9013 against
    0.9170/0.8848 on So-Fake-OOD, and a 0.0232 drop against 0.0321. But its
    decision values are NOT log-odds, and predict.py sigmoids them against a
    fixed operating point. Shipped raw, this probe scores 0.5143 accuracy and
    0.0561 recall: it stops detecting. Always calibrate() before save().
    """
    return RidgeClassifier(alpha=0.001, solver="lsqr").fit(X, y)


def calibrate(clf, X, y):
    """Fold Platt scaling into the weights. Returns (w, b) in log-odds.

    Platt is p = sigmoid(A*z + B) and z = x@w + b, so the calibrated logit is
    x@(A*w) + (A*b + B) -- still linear. The calibration collapses into the
    saved coefficients, so predict.py's scoring path needs no change and no new
    file has to ship beside the probe.

    Fitted on calib_ood, the generator-FAMILY-disjoint carve, for the same
    reason calibrate.py uses it: sid_calib is a set this probe scores 0.9996 on,
    and a slope fitted there manufactures over-confidence on unseen generators.

    Measured against a plain moment-match onto the old logistic scale, which
    needs no labels: Platt 0.9085/0.8731 vs 0.9057/0.8683, with better precision
    (0.8816 vs 0.8724) and fewer COCO false positives. Matching onto the
    TAMPERED branch instead -- the intuitive move, since that is what max()
    compares against -- is much worse, 0.8309/0.7722; that branch is
    over-confident and makes a bad reference.
    """
    w, b = clf.coef_.ravel(), float(np.ravel(clf.intercept_)[0])
    z = (X @ w + b).reshape(-1, 1)
    lr = LogisticRegression(max_iter=2000).fit(z, y)
    A, B = float(lr.coef_.ravel()[0]), float(lr.intercept_[0])
    return A * w, A * b + B


def calib_rows(source="so_fake_ood", split="calib_ood"):
    X, R = load(source)
    m = (R.split == split).values
    assert m.any(), f"no {split} rows in {source} -- run scripts/build_manifest.py"
    return X[m], R.label.values[m]


def auc_by_variant(clf, X, R):
    p = clf.decision_function(X)
    out = {v: roc_auc_score(R.label[m], p[m])
           for v, m in ((v, (R.variant == v).values) for v in R.variant.unique())
           if R.label[m].nunique() == 2}
    return pd.Series(out).sort_values()


def save(clf, path=MODEL):
    """Accepts an estimator, or a (w, b) pair already folded by calibrate()."""
    path.parent.mkdir(parents=True, exist_ok=True)
    w, b = clf if isinstance(clf, tuple) else (clf.coef_, clf.intercept_)
    np.savez(path, w=np.asarray(w).reshape(1, -1), b=np.asarray(b).reshape(1))


def train_tampered(extra_reals=()):
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
    for src in extra_reals:
        # More REAL photographs as negatives. Opt-in, because 5g tried this with
        # a different real source and made COCO false positives 4x WORSE
        # (13.6% -> 53.5%) while the branch's own AUROC rose. 7.5 explains why:
        # the branch fits a SID_Set construction artifact, so generic real
        # diversity sharpens the artifact instead of teaching the concept.
        # COCO train2017 is a narrower bet -- it carries the exact false-positive
        # mode section 5 found (watermarks, collages, printed packaging).
        Xr, Rr = load(src)
        k = (Rr.label.values == 0) & Rr.variant.isin(KEEP).values
        assert k.any(), f"{src} has no real rows in KEEP variants"
        X = np.concatenate([X, Xr[k]])
        y = np.r_[y, np.zeros(int(k.sum()))]
        print(f"  + {src}: {int(k.sum()):,} real rows as negatives")
    return fit(X, y), X, y


KEEP = ("clean", "jpeg50", "blur10", "noise005")     # sid_train's variant density


class Linear:
    """(w, b) wearing an estimator's face, so auc_by_variant can score it.

    train_general_plus returns folded weights rather than a fitted object, and
    without this the __main__ tables silently described a DIFFERENT probe from
    the one that had just been saved.
    """

    def __init__(self, w, b):
        self.w, self.b = np.asarray(w).ravel(), float(np.ravel(b)[0])

    def decision_function(self, X):
        return X @ self.w + self.b


def train_general_plus(extra="so_fake_ood", split="calib_ood", also=()):
    """Spend the calibration set on TRAINING, and rotate calibration across it.

    calib_ood exists to be generator-family-disjoint from test_ood, which makes
    it the only held-out data we may fit on. It has been doing one job -- Platt
    scaling -- and measurement says it is worth more as training data: adding its
    five families lifts the four worst test_ood generators +0.0559 and LOWERS
    COCO false positives. See docs/ERROR_ANALYSIS.md 3.1.

    So do both. Train on four families, calibrate on the fifth, rotate, and mean
    the five calibrated weight vectors. Averaging is legitimate here because
    calibrate() folds Platt into the coefficients: all five live in the same
    768-d space and differ only in which family tuned the slope, so their mean
    is the ensemble's linear equivalent at 1x inference cost.

    The rotation is not bookkeeping. The shipped operating point was picked on
    ONE calibration set with no evidence it would survive another; across these
    five folds the accuracy-optimal threshold ranges 0.575-0.695, which says
    calibration-set composition matters and is worth knowing.
    """
    Xs, Rs = load("sid_train")
    for src in also:
        # Extra generator families, downloaded rather than streamed. Trimmed to
        # KEEP so a source with a different variant density cannot outvote
        # sid_train by carrying more rows per image.
        Xa, Ra = load(src)
        k = Ra.variant.isin(KEEP).values
        Xs = np.concatenate([Xs, Xa[k]])
        Rs = pd.concat([Rs, Ra[k]], ignore_index=True)
        print(f"  + {src}: {int(k.sum()):,} rows, "
              f"{Ra.label[k].value_counts().to_dict()}")
    Xc, Rc = load(extra)
    cal = (Rc.split == split).values
    assert cal.any(), f"no {split} rows in {extra}"
    fams = sorted(Rc.generator[cal & (Rc.label == 1)].unique())

    # Reals here carry generator "unknown", so they cannot be folded by family.
    # Fold them by IMAGE, or a photo lands in train and calibration at once.
    rng = np.random.default_rng(0)
    ids = np.unique(Rc.image_id[cal & (Rc.label == 0)])
    fold = dict(zip(ids, rng.integers(0, len(fams), len(ids))))

    W, B = [], []
    for i, f in enumerate(fams):
        hold = (cal & (Rc.generator == f).values) | (
            cal & (Rc.label.values == 0)
            & np.array([fold.get(v, -1) == i for v in Rc.image_id]))
        tr = cal & ~hold & Rc.variant.isin(KEEP).values
        clf = fit_general(np.concatenate([Xs, Xc[tr]]),
                          np.r_[Rs.label.values, Rc.label.values[tr]])
        w, b = calibrate(clf, Xc[hold], Rc.label.values[hold])
        W.append(w); B.append(b)
        print(f"  fold calib={f:12s} train {int(tr.sum()):6,d} extra rows")
    return np.mean(W, 0), float(np.mean(B))


SOURCES = ("sid_train", "sid_tampered", "sid_calib", "so_fake_ood",
           "sid_tampered_eval", "organizer_val", "so_fake_tampered_eval",
           "wildfake_midjourney", "coco_train_reals", "real_holdout_laion")


def check_disjoint(sources=SOURCES):
    """The check that would have caught the 43-image SID_Set leak: no image may
    sit on both sides of the train/eval line once load() has filtered.

    Split out of __main__ so it can be run WITHOUT retraining. Everything below
    overwrites data/models/general.npz and tampered.npz, which is not something a
    test run should do to the shipped weights.
    """
    seen = {s: set(load(s)[1].image_id) for s in sources}
    for a in seen:
        for b in seen:
            if a < b:
                assert not (seen[a] & seen[b]), (
                    f"{len(seen[a] & seen[b])} images shared by {a} and {b}")
    print(f"splits disjoint: {sum(map(len, seen.values())):,} images "
          f"across {len(seen)} sources")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="split-disjointness only; do NOT retrain or overwrite the "
                         "shipped .npz files")
    ap.add_argument("--tampered-reals", nargs="*", default=[],
                    help="extra REAL sources as negatives for the tampered "
                         "branch, e.g. coco_train_reals. See ERROR_ANALYSIS 7.5 "
                         "before trusting the result -- 5g says this can backfire")
    ap.add_argument("--also", nargs="*", default=[],
                    help="extra training sources for --plus, e.g. wildfake_midjourney")
    ap.add_argument("--plus", action="store_true",
                    help="train the general probe on sid_train PLUS calib_ood's "
                         "generator families, rotating calibration across them "
                         "(docs/ERROR_ANALYSIS.md 3.1). Overwrites general.npz.")
    a = ap.parse_args()
    if a.check:
        check_disjoint()
        raise SystemExit

    check_disjoint()

    Xtr, Rtr = load("sid_train")
    print(f"general: train {Xtr.shape}  {Rtr.label.value_counts().to_dict()}")
    if a.plus:
        w, b = train_general_plus(also=a.also)
        save((w, b))                        # already calibrated, already folded
        clf = Linear(w, b)                  # tables must describe what was SAVED
        print("  NOTE: pair this probe with predict.TAMPERED_SCALE = 1.25")
    else:
        clf = fit_general(Xtr, Rtr.label.values)
        Xc, yc = calib_rows()
        save(calibrate(clf, Xc, yc))        # log-odds, or predict.py misreads it

    try:
        tclf, Xt, yt = train_tampered(a.tampered_reals)
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
