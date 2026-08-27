"""Robustness Evaluation Summary -- AUROC per branch per degradation setting.

Required submission deliverable. The *spread* between clean and worst is the
result; the mean is not. Each branch is refitted from cache here (seconds) so the
table never drifts from a stale .npz on someone's disk.

    python scripts/eval_grid.py                    # so_fake_ood, the headline
    python scripts/eval_grid.py --source sid_calib # in-distribution ceiling
    python scripts/eval_grid.py --no-combiners     # skip the slow section

Branches with no cache are skipped, so this still runs with a partial cache.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import numpy as np
import pandas as pd

from quorum.detectors.face import design, px_stats
from quorum.detectors.general import load, fit, auc_by_variant

OUT = Path(__file__).resolve().parents[1] / "docs" / "robustness.md"

# branch -> cache prefix. Same probe, same 15-variant grid, different features.
BRANCHES = {"general": "", "face": "face_", "spectral": "spec_"}

# Reported next to the face row rather than buried: the branch's clean AUC is
# partly a shortcut, so its degraded numbers must not be read as robustness.
BLUR_CAVEAT = """
> **Read the `face` row with care.** Its AUC *rises* under blur (0.9382 clean ->
> 0.9500 at `blur20`). That is not robustness and not survivorship -- on the
> identical 1,624 images present at both settings it still rises 0.8951 ->
> 0.9264. Blur hurts on generators we trained on (-0.0047) and helps on unseen
> ones (+0.0314), concentrated in small upsampled faces and anti-correlated with
> clean AUC (r = -0.685, p = 0.007). The clean number is inflated by
> pipeline-specific texture CLIP writes into the embedding. Working:
> `docs/HANDOVER-MODELS.md` section 8.
"""


def held_out(X, R):
    """Drop the calibration carve. so_fake_ood now contains calib_ood rows that
    calibrators and fusion are fitted on -- scoring them here would report a
    number nobody held out."""
    if "calib_ood" not in set(R.split):
        return X, R
    m = (R.split != "calib_ood").values
    return X[m], R[m].reset_index(drop=True)


def grid(prefix, eval_source, train_source="sid_train"):
    Xtr, Rtr = load(prefix + train_source)
    X, R = held_out(*load(prefix + eval_source))
    if prefix == "face_":                       # Kacey's 769th feature, not the 768-d baseline
        st = px_stats(Rtr)
        Xtr, X = design(Xtr, Rtr, st), design(X, R, st)
    return auc_by_variant(fit(Xtr, Rtr.label.values), X, R)


def tampered_grid(real_source):
    """Tampered eval has no negatives of its own -- borrow the reals from the
    eval source so every variant is scored on a matched pair."""
    from quorum.detectors.general import train_tampered
    clf, _, _ = train_tampered()
    Xe, Re = load("sid_tampered_eval")
    Xr, Rr = held_out(*load(real_source))
    m = (Rr.label.values == 0)
    return auc_by_variant(clf, np.concatenate([Xe, Xr[m]]),
                          pd.concat([Re, Rr[m]], ignore_index=True))


def build(source):
    cols = {}
    for name, prefix in BRANCHES.items():
        try:
            cols[name] = grid(prefix, source)
        except FileNotFoundError:
            print(f"  skip {name}: no {prefix or '<general>'} cache")
    try:
        cols["tampered"] = tampered_grid(source)
    except FileNotFoundError:
        print("  skip tampered: run stream_embed.py --tampered first")
    if not cols:
        raise SystemExit("no branch caches found -- run the embedding passes first")
    df = pd.DataFrame(cols)
    return df.reindex(["clean"] + sorted(i for i in df.index if i != "clean")).dropna(how="all")


def combiners(source):
    """How the branches are actually combined, on the FULL task.

    so_fake_ood alone understates every combiner that handles tampering, because
    it contains no tampered images. Pooling both eval sets is the only honest
    comparison: a detector that aces synthetic images and inverts on edited ones
    has not solved the problem.
    """
    from sklearn.linear_model import LogisticRegression
    from quorum.calibrate import CALIB_SOURCE, half
    from quorum.fusion import (COLUMNS, assemble, auc_by_variant as fauc,
                               fit_branches, fit_calibrators, raw_scores,
                               tampered_holdout)

    M = fit_branches()
    cal = raw_scores(CALIB_SOURCE, M)
    a, b = half(cal, "a"), half(cal, "b")
    tam = tampered_holdout(M)
    ta, tb = half(tam, "a", bit=1), half(tam, "b", bit=1)
    cals = fit_calibrators(cal, a, tam[ta])

    rows = pd.concat([cal[b], tam[tb]], ignore_index=True)
    clf = LogisticRegression(max_iter=2000).fit(assemble(rows, cals), rows.label.values)

    ood = raw_scores(source, M)
    ood = ood[ood.split != "calib_ood"].reset_index(drop=True)
    te = raw_scores("sid_tampered_eval", M)
    full = pd.concat([ood, te], ignore_index=True)
    Xf, Xo = assemble(full, cals), assemble(ood, cals)
    g, t = COLUMNS.index("cal_general"), COLUMNS.index("cal_tampered")

    def row(pf, po):
        vf, vo = fauc(pf, full), fauc(po, ood)
        return {"FULL avg": vf.mean(), "FULL worst": vf.min(),
                f"{source} clean": vo["clean"], f"{source} worst": vo.min()}

    return pd.DataFrame({
        "general alone":  row(Xf[:, g], Xo[:, g]),
        "max(gen,tamp)":  row(np.maximum(Xf[:, g], Xf[:, t]), np.maximum(Xo[:, g], Xo[:, t])),
        "fusion LR":      row(clf.predict_proba(Xf)[:, 1], clf.predict_proba(Xo)[:, 1]),
    }).T


def main(source, do_comb):
    df = build(source)
    summary = pd.DataFrame({"clean": df.loc["clean"], "worst": df.min(),
                            "worst_variant": df.idxmin(),
                            "drop": df.loc["clean"] - df.min()})
    assert df.notna().any().all(), "a branch produced no scoreable variant"

    comb = combiners(source) if do_comb else None
    fence = "\n```\n"
    parts = [
        f"# Robustness Evaluation Summary\n\n_Generated by scripts/eval_grid.py "
        f"on `{source}`._\n\nAUROC per branch under each degradation setting. The "
        f"clean-to-worst **drop** is the robustness claim; a high mean with a large "
        f"drop is a fragile detector.\n",
        "\n## Per-branch summary\n", fence, summary.to_string(float_format="%.4f"), fence,
        BLUR_CAVEAT,
        "\n## Full grid\n", fence, df.to_string(float_format="%.4f"), fence,
    ]
    if comb is not None:
        parts += [
            "\n## Combiners, on the full task\n\n`FULL` pools So-Fake-OOD "
            "(fully-synthetic) with sid_tampered_eval (locally edited) against the "
            "same real pool. The general probe is *inverted* on tampering, so a "
            "single-branch score that looks strong on one column collapses on the "
            "other.\n", fence, comb.to_string(float_format="%.4f"), fence,
            "\nThe task is **disjunctive** -- \"AI touched this\" = fully-synthetic "
            "OR locally edited -- so `max` beats a linear combiner in log-odds "
            "space, which is forced into one additive trade-off across two "
            "complementary detectors. `predict.py` ships `max` on this "
            "measurement. Fitted on a *generator-disjoint* calibration slice, "
            "fusion does win (`docs/HANDOVER-MODELS.md` section 4); building that "
            "slice is the open data task.\n",
        ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(parts), encoding="utf-8")

    print(df.to_string(float_format="%.4f"), "\n")
    print(summary.to_string(float_format="%.4f"))
    if comb is not None:
        print("\n" + comb.to_string(float_format="%.4f"))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="so_fake_ood")
    p.add_argument("--no-combiners", action="store_true")
    a = p.parse_args()
    main(a.source, not a.no_combiners)
