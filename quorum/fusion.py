"""Fusion: five branches, one verdict.

The meta-classifier is a logistic regression over calibrated branch scores plus
the context needed to read them. Input order is frozen in HANDOVER 5c:

    [ cal_general, cal_tampered, cal_face, face_present,
      cal_spectral, cal_text, text_present,
      content_onehot(5), degradation_estimate, provenance_prior ]

Two slots are placeholders on purpose. `text` was cut -- 9 hours of OCR for a
7-parameter model, against fusion being the critical path -- and provenance.py
is unbuilt and unowned. Both hold the neutral fill and keep their column, so
wiring either one in later is a one-line change and not a reshape.

Splits: branch probes are fitted on `train`, calibrators on calib_a, this LR on
calib_b. Fitting fusion on the calibrators' own rows would train it on
probabilities no future image ever produces.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from quorum.calibrate import (CALIB_SOURCE, CALIB_SPLIT, EVAL_SPLIT, fit_platt,
                              half, platt, plot_reliability)
from quorum.detectors.face import design, px_stats
from quorum.detectors.general import MODELS, fit, fit_general, load

MODEL = MODELS / "fusion.npz"
CONTENT_VECS = MODELS / "content_prompts.npz"
FIGURES = Path(__file__).resolve().parents[1] / "docs" / "figures"

CONTENT = ("face", "animal", "object", "scene", "text")
PROMPTS = {
    "face":   ["a photo of a person's face", "a portrait of a person"],
    "animal": ["a photo of an animal", "a photo of a pet"],
    "object": ["a photo of an object", "a close-up photo of a product"],
    "scene":  ["a photo of a landscape", "a photo of an outdoor scene"],
    "text":   ["a photo containing written text", "a screenshot with writing"],
}
# Order of the assembled matrix. Frozen -- predict.py and the demo index it.
COLUMNS = (["cal_general", "cal_tampered", "cal_face", "face_present",
            "cal_spectral", "cal_text", "text_present"]
           + [f"content_{c}" for c in CONTENT]
           + ["degradation_estimate", "provenance_prior"])

# A branch that did not fire must not read as "real" -- but inside THIS model
# that is a readability choice, not a performance one. Every absent branch also
# gets a `*_present` indicator, so its column is constant and the LR absorbs the
# fill into the indicator weight. Measured on so_fake_ood, filling cal_face with
# 0.0 instead: 0.9377/0.9052 clean/worst against 0.9378/0.9053. Same model.
# The choice DOES matter where no fitted weight sits in between -- predict.py's
# max(), where 0 is the correct fill, and anything the demo displays, where 0
# claims the branch ran and found the image authentic.
NEUTRAL = 0.5
TAMPERED_FIT = "a"     # probe fits this half of sid_tampered, fusion sees the other


def content_vectors() -> np.ndarray:
    """(5, 768) L2-normed CLIP text embeddings, one per content class.

    Cached: the text tower runs once for 10 prompts and never again. Averaging
    two templates per class then renormalising is the standard zero-shot recipe
    -- a single template swings several points on ambiguous classes.
    """
    if CONTENT_VECS.exists():
        with np.load(CONTENT_VECS) as z:
            return z["v"]

    import open_clip
    from quorum.embed import BACKBONE, Embedder

    emb = Embedder()
    tok = open_clip.get_tokenizer(BACKBONE)
    out = []
    with emb.torch.inference_mode():
        for c in CONTENT:
            t = emb.model.encode_text(tok(PROMPTS[c]).to(emb.device)).float()
            t = t / t.norm(dim=-1, keepdim=True)
            t = t.mean(0)
            out.append((t / t.norm()).cpu().numpy())
    v = np.stack(out).astype(np.float32)
    CONTENT_VECS.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CONTENT_VECS, v=v)
    return v


def content_onehot(X: np.ndarray) -> np.ndarray:
    """(N, 5) one-hot of the nearest content class. Free -- reuses the embedding.

    Content type is a fusion feature, not a branch: SPEC 6.3 records that we
    deliberately did not build per-content specialists.
    """
    k = np.argmax(X @ content_vectors().T, axis=1)
    return np.eye(len(CONTENT), dtype=np.float32)[k]


def fit_branches():
    """Every branch probe, fitted on `train` only. Seconds, against cache.

    The tampered probe is fitted on half of sid_tampered rather than all of it.
    The other half is the only data in the project where the general probe is
    not merely wrong but inverted (AUC 0.37 -- a locally-edited photo is
    globally authentic), and fusion has to see that case to learn it. Fitting
    the probe on all of sid_tampered would make those scores in-sample and
    teach fusion to over-trust the branch instead.
    """
    Xg, Rg = load("sid_train")
    Xf, Rf = load("face_sid_train")
    Xs, Rs = load("spec_sid_train")
    Xt, Rt = load("sid_tampered")
    stats = px_stats(Rf)

    real = Rg.label.values == 0
    ht = half(Rt, TAMPERED_FIT)
    tampered = fit(np.concatenate([Xg[real], Xt[ht]]),
                   np.r_[np.zeros(real.sum()), np.ones(ht.sum())])
    return {
        # fit_general, not fit: this must be the probe predict.py ships, or
        # the combiner table compares fusion against a model nobody runs.
        "general": fit_general(Xg, Rg.label.values),
        "tampered": tampered,
        "face": fit(design(Xf, Rf, stats), Rf.label.values),
        "spectral": fit(Xs, Rs.label.values),
        # "how degraded does this look" from the spectral vector alone. Fusion
        # needs it to tell "no artifacts found" from "could not measure"
        # (PIPELINE 7.1) -- the spectral branch goes to chance under resize, and
        # without this feature fusion cannot know that is what happened.
        "degradation": fit(Xs, (Rs.variant != "clean").astype(int).values),
        "px_stats": stats,
    }


def raw_scores(source: str, M: dict) -> pd.DataFrame:
    """One row per (image_id, variant) of `source`, with every branch's raw score.

    The general source is the spine because every image has an embedding; face
    and spectral left-join onto it and are allowed to be absent.
    """
    Xg, Rg = load(source)
    df = Rg[["image_id", "variant", "label", "split"]].copy()
    df["general"] = M["general"].decision_function(Xg)
    df["tampered"] = M["tampered"].decision_function(Xg)
    df[[f"content_{c}" for c in CONTENT]] = content_onehot(Xg)

    Xs, Rs = load("spec_" + source)
    s = Rs[["image_id", "variant"]].copy()
    s["spectral"] = M["spectral"].decision_function(Xs)
    s["degradation_estimate"] = M["degradation"].predict_proba(Xs)[:, 1]
    df = df.merge(s, on=["image_id", "variant"], how="left")

    Xf, Rf = load("face_" + source)
    f = Rf[["image_id", "variant"]].copy()
    f["face"] = M["face"].decision_function(design(Xf, Rf, M["px_stats"]))
    df = df.merge(f, on=["image_id", "variant"], how="left")
    return df


def tampered_holdout(M) -> pd.DataFrame:
    """The half of sid_tampered its probe never saw. All label 1.

    Out-of-sample for every branch: general and face and spectral were fitted on
    sid_train, which shares no images with sid_tampered.
    """
    df = raw_scores("sid_tampered", M)
    return df[half(df, "b" if TAMPERED_FIT == "a" else "a")].reset_index(drop=True)


def fit_calibrators(cal, mask, tam) -> dict:
    """Platt per branch on calib_a, skipping rows the branch missed.

    tampered is the exception. sid_calib is real + full_synthetic with zero
    tampered images, so calibrating there would fit the branch's score against
    a question it was not built to answer. It gets the held-out tampered rows
    against calib_a's reals instead.
    """
    cals = {}
    for b in ("general", "face", "spectral"):
        m = mask & cal[b].notna().values
        cals[b] = fit_platt(cal[b].values[m], cal.label.values[m])

    real = mask & (cal.label.values == 0)
    cals["tampered"] = fit_platt(
        np.r_[cal.tampered.values[real], tam.tampered.values],
        np.r_[np.zeros(real.sum()), np.ones(len(tam))])
    return cals


def assemble(df, cals) -> np.ndarray:
    """The frozen input vector. Missing branch -> presence 0 and a neutral fill."""
    out = pd.DataFrame(index=df.index)
    out["cal_general"] = platt(df.general.values, *cals["general"])
    out["cal_tampered"] = platt(df.tampered.values, *cals["tampered"])

    face = platt(df.face.values, *cals["face"])
    out["cal_face"] = np.where(df.face.notna().values, face, NEUTRAL)
    out["face_present"] = df.face.notna().values.astype(np.float32)

    spec = platt(df.spectral.values, *cals["spectral"])
    out["cal_spectral"] = np.where(df.spectral.notna().values, spec, NEUTRAL)

    out["cal_text"] = NEUTRAL          # branch cut, slot kept
    out["text_present"] = 0.0
    for c in CONTENT:
        out[f"content_{c}"] = df[f"content_{c}"].values
    out["degradation_estimate"] = df.degradation_estimate.fillna(NEUTRAL).values
    out["provenance_prior"] = NEUTRAL  # provenance.py unbuilt and unowned

    return out[list(COLUMNS)].to_numpy(dtype=np.float32)


def save(clf, cals, stats, path=MODEL):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, w=clf.coef_, b=clf.intercept_, columns=np.array(COLUMNS),
             px_mu=stats[0], px_sd=stats[1],
             **{f"cal_{k}": np.array(v) for k, v in cals.items()})


def auc_by_variant(p, df):
    out = {v: roc_auc_score(df.label[m], p[m])
           for v, m in ((v, (df.variant == v).values) for v in df.variant.unique())
           if df.label[m].nunique() == 2}
    return pd.Series(out).sort_values()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fit", dest="a_fit", default="calib",
                    choices=["calib", "calib+tampered"],
                    help="rows the fusion LR is fitted on. 'calib+tampered' buys "
                         "tampered coverage and costs ~6 points on so_fake_ood. "
                         "Both are printed either way; this only picks what is saved.")
    a_fit = ap.parse_args().a_fit

    M = fit_branches()

    # One pass over the source, then split. calib_ood and test_ood are disjoint by
    # generator family AND by image, enforced in scripts/build_manifest.py.
    scored = raw_scores(CALIB_SOURCE, M)
    cal = scored[scored.split == CALIB_SPLIT].reset_index(drop=True)
    a, b = half(cal, "a"), half(cal, "b")
    assert not (a & b).any() and (a | b).all(), "calib split is not a partition"

    # Split the holdout again on an independent bit: these rows have to fit the
    # tampered calibrator AND fusion, and one set doing both is the leak the
    # calib_a/calib_b split exists to prevent.
    tam = tampered_holdout(M)
    ta, tb = half(tam, "a", bit=1), half(tam, "b", bit=1)
    cals = fit_calibrators(cal, a, tam[ta])

    # calib_ood has no tampered images, so fusion fitted on it alone never sees the
    # general probe fail and leaves cal_tampered near zero. Adding the held-out
    # tampered rows fixes that and costs ~6 points on so_fake_ood -- a real trade,
    # so it is a flag and BOTH numbers print either way. Reporting one without the
    # other misrepresents the model whichever one you pick.
    fits = {"calib": cal[b],
            "calib+tampered": pd.concat([cal[b], tam[tb]], ignore_index=True)}
    ood = scored[scored.split == EVAL_SPLIT].reset_index(drop=True)
    Xo = assemble(ood, cals)
    # tampered_eval brings no negatives of its own; borrow the eval reals, same
    # pairing as eval_grid.tampered_grid.
    both = pd.concat([raw_scores("sid_tampered_eval", M),
                      ood[ood.label.values == 0]], ignore_index=True)
    Xt, yt = assemble(both, cals), both.label.values

    models = {}
    for name, rows in fits.items():
        Xb, yb = assemble(rows, cals), rows.label.values
        assert Xb.shape[1] == len(COLUMNS) == 14, Xb.shape
        assert ((Xb >= 0) & (Xb <= 1)).all(), "a fusion feature left [0,1]"
        models[name] = (LogisticRegression(max_iter=2000).fit(Xb, yb), len(rows))

    clf, n_fit = models[a_fit]
    save(clf, cals, M["px_stats"])
    print(f"fusion: calib_a {a.sum():,} + tampered {ta.sum():,} -> calibrators")
    print(f"saved --fit {a_fit} ({n_fit:,} rows). The trade, both ways:\n")
    gen_c = auc_by_variant(platt(ood.general.values, *cals["general"]), ood)
    print(f"  {'fit on':16s} {'ood clean':>10s} {'ood worst':>10s} {'tampered':>9s}")
    print(f"  {'general alone':16s} {gen_c['clean']:10.4f} {gen_c.min():10.4f} "
          f"{roc_auc_score(yt, platt(both.general.values, *cals['general'])):9.4f}")
    for name, (m, _) in models.items():
        av = auc_by_variant(m.predict_proba(Xo)[:, 1], ood)
        mark = " <- saved" if name == a_fit else ""
        print(f"  {name:16s} {av['clean']:10.4f} {av.min():10.4f} "
              f"{roc_auc_score(yt, m.predict_proba(Xt)[:, 1]):9.4f}{mark}")
    p = clf.predict_proba(Xo)[:, 1]
    fus = auc_by_variant(p, ood)
    singles = {n: auc_by_variant(platt(ood[n].fillna(0.0).values, *cals[n]), ood)
               for n in ("general", "spectral")}
    # face is scored on face-present rows ONLY. Across the whole population the
    # number measures the 0.5 fill on the 73% with no face, not the branch --
    # 0.66 instead of its actual 0.94.
    fm = ood.face.notna().values
    singles["face"] = auc_by_variant(platt(ood.face.values[fm], *cals["face"]), ood[fm])

    t = pd.DataFrame({"fusion": fus, **singles}).sort_values("fusion")
    print("\n=== so_fake_ood: AUC per variant (face = face-present rows only) ===")
    print(t.to_string(float_format="%.4f"))

    print(f"\nfusion   clean {fus['clean']:.4f}  worst {fus.min():.4f} ({fus.idxmin()})"
          f"  drop {fus['clean'] - fus.min():.4f}")
    g = singles["general"]
    print(f"general  clean {g['clean']:.4f}  worst {g.min():.4f} ({g.idxmin()})"
          f"  drop {g['clean'] - g.min():.4f}")

    # The case a single general probe cannot do at all: tampered vs real, where it
    # scores BELOW chance. Population built above; `both`/`yt` are reused here.
    print(f"\ntampered_eval vs reals   fusion {roc_auc_score(yt, clf.predict_proba(Xt)[:, 1]):.4f}   "
          f"general {roc_auc_score(yt, platt(both.general.values, *cals['general'])):.4f}   "
          f"tampered {roc_auc_score(yt, platt(both.tampered.values, *cals['tampered'])):.4f}")

    print("\nweights:")
    for name, w in sorted(zip(COLUMNS, clf.coef_[0]), key=lambda kv: -abs(kv[1])):
        print(f"  {name:22s} {w:+.3f}")

    # calib_b is held out of the calibrators, so this is an honest
    # in-distribution curve rather than the fit reflecting itself.
    panels = {}
    for n in ("general", "face", "spectral"):
        mc, mo = b & cal[n].notna().values, ood[n].notna().values
        panels[n] = {
            "calib_b (same generators)": (platt(cal[n].values[mc], *cals[n]),
                                          cal.label.values[mc]),
            "so_fake_ood (unseen)": (platt(ood[n].values[mo], *cals[n]),
                                     ood.label.values[mo]),
        }

    # tampered gets its own populations: calib holds no tampered images, so
    # scoring it there measures the branch against a question it was not
    # calibrated for and reads as a calibration failure that is not one.
    realb = b & (cal.label.values == 0)
    panels["tampered"] = {
        "tampered holdout (same)": (
            platt(np.r_[tam.tampered.values[tb], cal.tampered.values[realb]], *cals["tampered"]),
            np.r_[np.ones(tb.sum()), np.zeros(realb.sum())]),
        "tampered_eval (unseen)": (
            platt(both.tampered.values, *cals["tampered"]), yt),
    }
    print(f"\nreliability -> {plot_reliability(panels, FIGURES / 'reliability.png')}")
