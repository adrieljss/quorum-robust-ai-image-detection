"""Platt scaling per branch, plus the calib split that keeps fusion honest.

Every branch emits a raw, uncalibrated score on its own scale -- a CLIP probe's
decision_function runs to +-8, the 8-feature spectral one barely leaves +-2.
Fusion cannot weigh those against each other until they all mean P(AI), so each
one gets its own sigmoid fit here.

The split matters as much as the fit. Calibrators and the fusion LR must not see
the same rows: a calibrator is exact on the rows it was fitted on, so fusion
trained there would learn from probabilities it will never see again and
over-trust every branch. sid_calib's 3,996 images halve into calib_a (fit the
calibrators) and calib_b (fit fusion), split by image_id so all 15 variants of
one image stay on the same side.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

MODELS = Path(__file__).resolve().parents[1] / "data" / "models"
CALIBRATORS = MODELS / "calibrators.npz"
# Adriel, per HANDOVER-MODELS 6: the calibration set is now a generator-disjoint
# carve out of So-Fake-OOD, not sid_calib. Fitting Platt on sid_calib fitted it on
# a branch that scores 0.9996 there, and the resulting slope manufactured
# over-confidence on every unseen generator -- ECE out 0.1026 vs 0.0217 for
# general, 0.1665 vs 0.0333 for face. The carve is by generator FAMILY (see
# scripts/build_manifest.py) so Ideogram2/Ideogram3 cannot straddle the boundary.
CALIB_SOURCE = "so_fake_ood"
CALIB_SPLIT = "calib_ood"
EVAL_SPLIT = "test_ood"


def half(R, which: str, bit: int = 0) -> np.ndarray:
    """Boolean mask for an a/b split on image_id, not on rows.

    Splitting on rows would put jpeg30 of an image in one half and its clean in
    the other -- near-duplicate features across the boundary, and the leakage
    this split exists to prevent walks straight back in.

    `bit` picks which 8 hex characters of the id to read, so a set already
    halved on bit 0 can be halved again independently. Re-using bit 0 on an
    already-split frame returns everything or nothing, silently.
    """
    assert which in ("a", "b"), which
    lo = 8 * bit
    v = R.image_id.map(lambda s: int(s[lo:lo + 8], 16) & 1).values
    return v == (0 if which == "a" else 1)


def fit_platt(s, y):
    """(a, b) so that sigmoid(a*s + b) is calibrated. Platt scaling.

    C is large because this is a fit, not a model: regularisation shrinks `a`
    toward zero, which flattens every probability toward 0.5 and reads as
    "uncertain" when the branch was in fact confident.
    """
    s = np.asarray(s, dtype=np.float64).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(s, np.asarray(y))
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def platt(s, a, b):
    return 1.0 / (1.0 + np.exp(-(a * np.asarray(s, dtype=np.float64) + b)))


def reliability(p, y, bins: int = 10):
    """Per-bin confidence vs observed frequency -- the reliability diagram as a
    table. A calibrated branch has conf ~= freq in every row."""
    p, y = np.asarray(p), np.asarray(y)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return pd.DataFrame({
        "n": np.bincount(idx, minlength=bins),
        "conf": np.bincount(idx, weights=p, minlength=bins),
        "freq": np.bincount(idx, weights=y, minlength=bins),
    }).assign(conf=lambda d: d.conf / d.n.clip(lower=1),
              freq=lambda d: d.freq / d.n.clip(lower=1))


def ece(p, y, bins: int = 10) -> float:
    """Expected calibration error: mean |conf - freq|, weighted by bin count."""
    r = reliability(p, y, bins)
    return float((r.n / r.n.sum() * (r.conf - r.freq).abs()).sum())


def plot_reliability(panels: dict, path, bins: int = 10):
    """One panel per branch, one curve per source. Required by TODO stage 3.

    The gap between the curves is the whole point: a calibrator fitted on
    same-generator data sits on the diagonal there and leaves it entirely on a
    generator it has not seen. Bin markers are sized by count so the sparse tails
    cannot be mistaken for the same evidence as the middle.
    """
    import matplotlib
    matplotlib.use("Agg")          # no display on a headless run
    import matplotlib.pyplot as plt

    cols = min(len(panels), 2)
    rows = -(-len(panels) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 4.6 * rows),
                             squeeze=False)
    colours = ("#2B4A9B", "#B4462F")

    for ax, (name, curves) in zip(axes.ravel(), panels.items()):
        ax.plot([0, 1], [0, 1], ls=(0, (4, 3)), lw=1.0, c="#9AA3AF", zorder=1)
        for (label, (p, y)), c in zip(curves.items(), colours):
            r = reliability(p, y, bins)
            keep = r.n > 0
            ax.plot(r.conf[keep], r.freq[keep], "-", lw=1.6, c=c, zorder=3,
                    label=f"{label}  ECE {ece(p, y, bins):.4f}")
            ax.scatter(r.conf[keep], r.freq[keep], zorder=4, c=c,
                       s=8 + 90 * r.n[keep] / r.n.max())
        ax.set_title(name, fontsize=12, fontweight="bold", loc="left")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_xlabel("predicted P(AI)"); ax.set_ylabel("observed frequency")
        ax.legend(fontsize=8, loc="upper left", frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.18, lw=0.6)

    for ax in axes.ravel()[len(panels):]:
        ax.set_visible(False)

    fig.suptitle("Branch calibration: fitted in-distribution, measured out",
                 fontsize=13, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save(cals: dict, path=CALIBRATORS):
    """cals maps branch name -> (a, b)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{k: np.array(v, dtype=np.float64) for k, v in cals.items()})


def load(path=CALIBRATORS) -> dict:
    with np.load(path) as z:
        return {k: (float(z[k][0]), float(z[k][1])) for k in z.files}


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # a branch that is informative but badly scaled: raw scores 20x too wide,
    # which is exactly what an unregularised decision_function looks like.
    y = rng.integers(0, 2, 20_000)
    s = rng.normal(y * 2.0 - 1.0, 1.0) * 20.0

    tr, te = np.zeros(len(y), bool), np.zeros(len(y), bool)
    tr[: len(y) // 2], te[len(y) // 2:] = True, True

    a, b = fit_platt(s[tr], y[tr])
    p = platt(s[te], a, b)
    assert ((p >= 0) & (p <= 1)).all(), "probabilities out of range"

    raw = 1.0 / (1.0 + np.exp(-s[te]))          # what fusion would see uncalibrated
    assert ece(p, y[te]) < ece(raw, y[te]), "calibration made it worse"
    assert ece(p, y[te]) < 0.02, f"ECE still {ece(p, y[te]):.4f}"
    print(f"platt: ECE {ece(raw, y[te]):.4f} -> {ece(p, y[te]):.4f}  (a={a:.4f}, b={b:.4f})")

    # the split: disjoint, exhaustive, and never splits one image across halves
    R = pd.DataFrame({"image_id": [f"{i:016x}" for i in rng.integers(0, 2**60, 5000)]})
    R = pd.concat([R.assign(variant=v) for v in ("clean", "jpeg30", "blur20")])
    ma, mb = half(R, "a"), half(R, "b")
    assert (ma | mb).all() and not (ma & mb).any(), "halves overlap or miss rows"
    assert 0.45 < ma.mean() < 0.55, f"lopsided split: {ma.mean():.3f}"
    spread = R.assign(a=ma).groupby("image_id").a.nunique()
    assert spread.max() == 1, "an image landed in both halves"
    print(f"split: calib_a {ma.sum():,} rows / calib_b {mb.sum():,} rows, no image in both")

    r = reliability(p, y[te])
    assert np.isclose(r.n.sum(), te.sum()), "reliability dropped rows"
    print(f"calibrate.py ok: {len(r)} bins, worst |conf-freq| {(r.conf - r.freq).abs().max():.4f}")
