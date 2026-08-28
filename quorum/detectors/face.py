"""Face probe: logistic regression on CLIP embeddings of aligned face crops.

Preprocessing is already cached by quorum/features.py -- YuNet detect, Umeyama
5-point align to the ArcFace template, 224px crop, the shared CLIP. This trains
the probe and nothing else.

AUC alone lies about this branch, so every AUC here prints next to the coverage
that produced it -- degradation kills detection on the small and marginal faces
first, and a number computed on the survivors is a number on an easier set.

That is not why AUC RISES under blur (0.9421 clean -> 0.9512 at blur20).
Survivorship was the obvious read and it is wrong: on the same 1,624 images the
score still goes up. It is shortcut learning -- the probe partly reads
resampling texture in upscaled crops, and blur wipes it out. Costs 0.0047
in-distribution, pays 0.0314 out of it. HANDOVER-MODELS.md section 8.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from quorum.detectors.general import MODELS, load, fit, auc_by_variant

MODEL = MODELS / "face.npz"
MANIFEST = Path(__file__).resolve().parents[2] / "data" / "manifests" / "main.csv"


def px_stats(R):
    """Mean/std of log2(face_px) on the training rows, saved with the model.

    log2 because the effect is a ratio: a 64px face upscaled to 224 carries
    ~2.8x the effective degradation of a 181px one downscaled to it.
    """
    z = np.log2(R.face_px.values.astype(np.float32))
    return float(z.mean()), float(z.std()) or 1.0


def design(X, R, stats):
    """768 embedding dims + 1 scaled face_px -> 769 parameters.

    Standardised because the embeddings are L2-normed and each component is
    ~0.036: a raw 64-181 column would swamp the L2 penalty and the probe would
    fit box size instead of provenance.
    """
    mu, sd = stats
    z = (np.log2(R.face_px.values.astype(np.float32)) - mu) / sd
    return np.hstack([X, z[:, None]]).astype(np.float32)


def coverage(face_source: str, base_source: str):
    """Per variant: how many images the branch can speak about at all.

    Numerator is face rows -- features.py writes one only when a face was
    found. Denominator is every image of that source in the manifest. `retain`
    is coverage against clean coverage, the number that shows the detector
    dying -- noise01 is the worst variant on so_fake_ood, holding 0.84.

    retain divides the RATIOS, never the raw counts. Train sources carry clean
    plus 3 sampled variants, so a variant holds ~1/14 of clean's images and a
    count ratio reads as catastrophic detector failure: sid_train noise01
    scored 0.18 that way against a true 0.85.
    """
    _, R = load(face_source)
    m = pd.read_csv(MANIFEST, usecols=["source", "image_id", "variant"])
    m = m[m.source == base_source]

    out = pd.DataFrame({"faces": R.groupby("variant").image_id.nunique(),
                        "images": m.groupby("variant").image_id.nunique()})
    out = out.fillna(0)
    out["cover"] = out.faces / out.images
    out["retain"] = out.cover / out.cover.get("clean", np.nan)
    return out


def save(clf, stats, path=MODEL):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, w=clf.coef_, b=clf.intercept_, px_mu=stats[0], px_sd=stats[1])


def report(clf, stats, face_source: str, base_source: str):
    """The deliverable: AUC and coverage in one table, sorted by AUC.

    calib_ood is excluded: Adriel's generator-disjoint carve (HANDOVER.md 5d)
    moved 2,044 So-Fake images into the calibration split, and reporting the
    headline face number on rows the calibrators are fitted to would not be a
    held-out number. The probe itself never trains on them either way.
    """
    X, R = load(face_source)
    keep = (R.split != "calib_ood").values
    X, R = X[keep], R[keep].reset_index(drop=True)
    a = auc_by_variant(clf, design(X, R, stats), R)
    t = pd.DataFrame({"auc": a}).join(coverage(face_source, base_source))
    return t.sort_values("auc")


if __name__ == "__main__":
    Xtr, Rtr = load("face_sid_train")
    stats = px_stats(Rtr)
    ytr = Rtr.label.values
    print(f"face: train {Xtr.shape}  {Rtr.label.value_counts().to_dict()}  "
          f"face_px {Rtr.face_px.min():.0f}-{Rtr.face_px.max():.0f}px")

    D = design(Xtr, Rtr, stats)
    assert D.shape == (len(Xtr), Xtr.shape[1] + 1), D.shape
    assert np.isfinite(D).all(), "non-finite design matrix"

    clf = fit(D, ytr)
    save(clf, stats)

    t = report(clf, stats, "face_so_fake_ood", "so_fake_ood")
    assert (t.cover <= 1.0).all(), "more face rows than images -- dedupe broke"
    assert t.images.nunique() == 1, "so_fake_ood is not a full grid"

    # regression guard on retain: it divides coverage, not counts. Dividing the
    # raw counts read 0.18 on sid_train noise01 against a true 0.85, because a
    # train variant holds ~1/14 of clean's images and the ratio is meaningless.
    c = coverage("face_sid_train", "sid_train")
    assert c.images.nunique() > 1, "sid_train should be a sampled grid"
    assert c.retain.drop("clean").min() > 0.5, c.retain.round(3).to_dict()

    print("\n=== face_so_fake_ood ===")
    print(t.to_string(float_format="%.4f"))
    print(f"clean {t.auc.get('clean', float('nan')):.4f}  "
          f"worst {t.auc.min():.4f} ({t.auc.idxmin()})  "
          f"drop {t.auc.get('clean', 0) - t.auc.min():.4f}")

    # The point of conditioning: does face_px actually buy anything? Report it
    # rather than assume it -- 769 params vs 768 is not free if it does not.
    Xe, Re = load("face_so_fake_ood")
    plain = auc_by_variant(fit(Xtr, ytr), Xe, Re)
    print(f"\nface_px conditioning  clean {t.auc['clean']:.4f} vs {plain['clean']:.4f}"
          f"   worst {t.auc.min():.4f} vs {plain.min():.4f}")
