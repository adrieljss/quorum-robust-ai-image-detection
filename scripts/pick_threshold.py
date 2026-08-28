"""Pick predict.py's operating point on calib_ood. Prints the constant to paste.

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


def shipped(X):
    """Exactly what predict.py computes: raw sigmoids, no Platt, then max."""
    out = []
    for n in ("general", "tampered"):
        z = np.load(MODELS / f"{n}.npz")
        out.append(sigmoid(X @ z["w"].ravel() + z["b"].ravel()[0]))
    return np.maximum(*out)


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

    # Accuracy is FLAT across roughly 0.60-0.73 (within 0.01 on both calib and
    # every held-out set), so argmax alone picks noise. Take the highest
    # threshold still within TOL of the best: on a plateau, prefer the end that
    # fires less, because every extra positive here is a real photograph
    # accused of being synthetic. Precision and FPR are the tiebreak, chosen
    # before looking at test_ood.
    TOL = 0.01
    grid = np.linspace(0.02, 0.98, 481)
    acc = np.array([((pk >= t) == (yk == 1)).mean() for t in grid])
    thr = float(grid[np.flatnonzero(acc >= acc.max() - TOL)[-1]])
    print(f"accuracy plateau (within {TOL}): "
          f"{grid[np.flatnonzero(acc >= acc.max() - TOL)[0]]:.3f}-{thr:.3f}")

    y, p = Ro.label.values[ev], shipped(Xo[ev])
    pc = shipped(Xc[(Rc.variant == "clean").values])
    pt = shipped(Xt[(Rt.variant == "clean").values])
    print(f"picked on calib_ood ({cal.sum():,} rows, all variants): {thr:.4f}")
    print(f"  calib accuracy {max(acc):.4f} vs {acc[len(grid)//2 - 1]:.4f} at 0.5")
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
    print(f"\npaste into predict.py:\n\n    OPERATING_POINT = {thr:.4f}")


if __name__ == "__main__":
    main()
