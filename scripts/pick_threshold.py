"""INVALID since 30 Aug -- do not trust this script's number.

It picks the operating point on calib_ood, which was held out when this was
written. `python -m quorum.detectors.general --plus` moved calib_ood into the
TRAINING set (docs/ERROR_ANALYSIS.md 3.1), so this now scores a model on its own
training data and reports a threshold that is too low.

The live rule is the cross-validated one in predict.py: pick per fold on the one
generator family that fold's model never saw, then average. Port it here or
delete this file; leaving it runnable and wrong is the worst of the three.

Pick predict.py's operating point on calib_ood. Prints the constant to paste.

    python scripts/pick_threshold.py

predict.py ships max(P_general, P_tampered) as a raw sigmoid, and 0.5 was never
chosen -- it is just the sigmoid default. Measured on held-out test_ood, 0.5
costs 0.08 precision and flags 25% of COCO photographs as AI. This picks the
accuracy-optimal cut on calib_ood (family-disjoint, never test_ood) and reports
what it does on every held-out set, so the trade is visible before it is shipped.

Re-run this after retraining either probe: the constant is not transferable.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.metrics import roc_auc_score

from quorum.detectors.general import MODELS, load


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def shipped(X, names=("general", "tampered")):
    """Exactly what predict.py computes: raw sigmoids, tampered scaled, then max.

    `names` is a knob for asking what a subset of the branches would do -- a
    one-branch call is still a valid scorer, so reduce rather than maximum(a, b).
    """
    from predict import TAMPERED_SCALE
    out = []
    for n in names:
        z = np.load(MODELS / f"{n}.npz")
        # The tampered branch carries an explicit scale since 30 Aug. max() is
        # NOT invariant to rescaling one argument, so omitting it here makes this
        # a different scorer from the one that ships -- which is precisely the
        # drift predict.self_check exists to catch, and did.
        a = TAMPERED_SCALE if n == "tampered" else 1.0
        out.append(sigmoid(a * (X @ z["w"].ravel() + z["b"].ravel()[0])))
    return np.maximum.reduce(out)


# Accuracy is FLAT across roughly 0.60-0.73 (within 0.01 on both calib and every
# held-out set), so argmax alone picks noise.
TOL = 0.01


def plateau(p, y, tol=TOL):
    """(highest, lowest) threshold within `tol` of peak accuracy, plus the curve.

    On a plateau prefer the end that fires LESS: every extra positive there is a
    real photograph accused of being synthetic. Precision and FPR are the
    tiebreak, chosen before looking at test_ood.
    """
    grid = np.linspace(0.02, 0.98, 481)
    acc = np.array([((p >= t) == (y == 1)).mean() for t in grid])
    keep = np.flatnonzero(acc >= acc.max() - tol)
    return float(grid[keep[-1]]), float(grid[keep[0]]), acc


def stats(y, p, thr):
    hat = p >= thr
    tp = int((hat & (y == 1)).sum()); fp = int((hat & (y == 0)).sum())
    tn = int((~hat & (y == 0)).sum()); fn = int((~hat & (y == 1)).sum())
    return {"acc": float((hat == (y == 1)).mean()),
            "prec": tp / (tp + fp) if tp + fp else float("nan"),
            "rec": tp / (tp + fn) if tp + fn else float("nan"),
            "fpr": fp / (fp + tn) if fp + tn else float("nan")}


def main():
    Xo, Ro = load("so_fake_ood")
    Xc, Rc = load("organizer_val")
    Xt, Rt = load("sid_tampered_eval")

    # All variants, not just clean: predict.py sees degraded images too, and
    # 2,044 clean rows is thin for a threshold. Held out from test_ood by family.
    cal = (Ro.split == "calib_ood").values
    ev = ((Ro.split == "test_ood") & (Ro.variant == "clean")).values
    pk, yk = shipped(Xo[cal]), Ro.label.values[cal]

    thr, lo, acc = plateau(pk, yk)
    # index of 0.5 on plateau()'s grid, derived rather than counted: len//2 - 1
    # lands on 0.498 and the line below then labels it "at 0.5".
    at_half = int(np.argmin(np.abs(np.linspace(0.02, 0.98, 481) - 0.5)))
    print(f"accuracy plateau (within {TOL}): {lo:.3f}-{thr:.3f}")

    y, p = Ro.label.values[ev], shipped(Xo[ev])
    # label == 0 matters: organizer_val is COCO reals AND WildFake DALL-E 3.
    # Without it the "COCO FP" column counted every correctly-caught fake as a
    # false positive and read 56.3% at 0.5 against a true 27.6%.
    pc = shipped(Xc[(Rc.variant == "clean").values & (Rc.label.values == 0)])
    pt = shipped(Xt[(Rt.variant == "clean").values])
    print(f"picked on calib_ood ({cal.sum():,} rows, all variants): {thr:.4f}")
    print(f"  calib accuracy {max(acc):.4f} vs {acc[at_half]:.4f} at 0.5")
    print(f"\nheld-out test_ood AUROC {roc_auc_score(y, p):.4f} "
          f"(unchanged -- a threshold cannot move it)\n")

    h = f"{'':10}{'acc':>7}{'prec':>7}{'recall':>8}{'FPR':>7}{'COCO FP':>9}{'tamp rec':>10}"
    print(h + "\n" + "-" * len(h))
    for tag, t in (("@0.5", 0.5), (f"@{thr:.3f}", thr)):
        s = stats(y, p, t)
        print(f"{tag:10}{s['acc']:>7.3f}{s['prec']:>7.3f}{s['rec']:>8.3f}"
              f"{s['fpr']:>7.3f}{(pc >= t).mean():>9.1%}{(pt >= t).mean():>10.3f}")

    # predict.py emits a score, not a verdict, so shift the scale instead of
    # exporting a threshold: 0.5 on the output becomes this operating point.
    # Monotone, so AUROC is untouched and rank-based grading sees no change.
    # This rule maximises ACCURACY and nothing else. Accuracy is nearly flat
    # across the plateau while the COCO false-positive rate is not, so the
    # suggestion below can be worse on the axis this project actually chose.
    # Compare the COCO column above before pasting.
    from predict import OPERATING_POINT as SHIPPED
    print(f"\naccuracy-optimal on calib_ood: {thr:.4f}")
    if abs(thr - SHIPPED) > 1e-4:
        print(f"predict.py ships:          {SHIPPED:.4f}  <- NOT the same, and that "
              f"may be deliberate.\n"
              f"  The plateau is flat in accuracy and steep in false positives on "
              f"real\n  photography. 0.766 was kept over 0.692 because it holds COCO "
              f"at 8.9%\n  instead of 13.7% for +0.002 accuracy. See predict.py.")
    else:
        print(f"predict.py ships:          {SHIPPED:.4f}  (agrees)")


if __name__ == "__main__":
    main()
