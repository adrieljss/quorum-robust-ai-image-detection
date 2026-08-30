"""Required deliverable: directory of images -> JSON of {image_path, pred}.

    python predict.py --input-dir path/to/images --output preds.json

`pred` is P(AI-generated) in [0,1]. Two fields, no more -- the demo's richer
schema stays in the demo.

The score is shifted so that **0.5 is the operating point**. It was not before:
0.5 is the sigmoid's default, nobody chose it, and on held-out test_ood it cost
0.09 precision and flagged 27.6% of COCO photographs as AI-generated. The shift
is monotone, so AUROC is identical (0.8997) and rank-based grading sees no
change; only a threshold-based read of `pred` moves.

It is a TRADE, not a free win, and both halves belong here:

    test_ood clean          acc    prec  recall      F1   COCO FP  tamp rec
      0.8523 (ships)      0.821   0.926   0.698   0.796      6.3%     0.704
      0.8057 (policy)     0.839   0.906   0.758   0.825      8.9%     0.754
    the model this REPLACED, at its own operating point
      0.766               0.825   0.882   0.751   0.811      8.9%     0.746

The probe underneath changed on 30 Aug (general --plus, docs/ERROR_ANALYSIS.md
3.1) and AUROC went 0.9085 -> 0.9268, which no threshold can move. So the two
rows above are the same better model at two defensible cuts:

  0.8523 SHIPS. Derived by cross-validation over calib_ood's five generator
    families -- each fold's threshold is picked on the one family that fold's
    model never saw, then averaged (0.585/0.600/0.705/0.665/0.635). No model
    scores its own training data, so this reads NO evaluation set at all.
    It cuts false accusations on real photography 8.9% -> 6.3%, a 30% drop.

  0.8057 is the alternative: hold the OLD policy (8.9% COCO FP) constant across
    the model change. It strictly dominates the model it replaced on every
    axis -- same false positives, +0.7pp recall, +0.025 precision, +0.015
    accuracy, +0.015 F1, +0.8pp tampered recall. Legitimate, but it READS COCO
    to place the cut, so it is a preserved decision rather than an independent
    derivation. Use it if this is ever scored on F1.

Chosen anyway: at a realistic base rate most uploads are genuine, so a false
accusation is the expensive error, and F1 weights a missed fake and a libelled
photograph equally. That premise is the decision -- argue with it, not the code.

The general probe is a RidgeClassifier whose Platt scaling is FOLDED INTO its
saved weights (`general.calibrate`). Ridge decision values are not log-odds;
shipped raw through the sigmoid below they scored 0.514 accuracy and 0.056
recall. The fold is linear, so it lives in `w`/`b` and nothing here changes.

ponytail: fusion.py exists and this still does not call it, deliberately.
Measured on so_fake_ood, clean / worst: max 0.9189 / 0.8921, fusion
0.9175 / 0.8905. The margin is now only +0.0014, so this is much closer than it
was -- but `max` still wins on both, and on the pooled task by more (0.8733 vs
0.8674). See HANDOVER.md 5c/5e for the two fit sets and why `max` beats a
learned combiner on a disjunctive task.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from quorum.detectors.general import MODEL, MODEL_TAMPERED
from quorum.embed import Embedder

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
BATCH = 64
# Cross-validated over calib_ood's five generator families: each fold picks the
# high end of its accuracy plateau on the family that fold's model never saw,
# and the five are averaged. Accuracy is flat across the plateau, so its high end
# is taken -- every extra positive there is a real photograph accused of being
# fake.
#
# WARNING: scripts/pick_threshold.py DISAGREES and is currently wrong. It picks
# on calib_ood as if that were held out, and since `general --plus` that set is
# TRAINING data. Its number is contaminated. Fix it or ignore it; do not paste
# its output here.
OPERATING_POINT = 0.8523
SHIFT = float(np.log(OPERATING_POINT / (1 - OPERATING_POINT)))

# How loudly the tampered branch speaks inside max(). NOT a free parameter that
# was fitted -- a policy dial, like OPERATING_POINT above, and it was hiding.
#
# max(sigmoid(g), sigmoid(t)) is not invariant to rescaling g, so the general
# branch's Platt SLOPE silently decides how often the tampered branch wins the
# max. Across five calibration folds that moved COCO false positives from 3.8%
# to 12.9% AT FIXED RECALL -- a balance nobody chose, inherited from whichever
# generator families happened to sit in the calibration set. Naming it here makes
# it a decision instead of an accident.
#
# Measured on the --plus probe, threshold held at 75% So-Fake-OOD recall:
#     alpha   COCO FP   tampered recall
#     0.6        2.3%        53.0%
#     1.0        6.2%        69.5%
#     1.25       8.5%        74.6%   <- holds the shipped policy
#     1.5       10.5%        77.4%
#
# 1.0 is a no-op and pairs with the DEFAULT probe. 1.25 pairs with the --plus
# probe and is what SHIPS. The two are calibrated together: revert one without
# the other and the operating point moves silently.
TAMPERED_SCALE = 1.25


def _probe(path):
    d = np.load(path)
    return d["w"].ravel(), d["b"][0]


def load_probes():
    """[(w, b)] per branch, tampered pre-scaled by TAMPERED_SCALE.

    Scaling here rather than in score_embeddings so every caller -- the demo,
    the figures, pick_threshold -- inherits the same balance without knowing the
    dial exists. The scale is linear, so it folds into (w, b) like Platt does.
    """
    probes = [_probe(MODEL)]
    if MODEL_TAMPERED.exists():
        w, b = _probe(MODEL_TAMPERED)
        probes.append((TAMPERED_SCALE * w, TAMPERED_SCALE * b))
    return probes


def score_embeddings(v, probes=None) -> np.ndarray:
    """(n, 768) CLIP embeddings -> P(AI). THE definition of the shipped score.

    Every evaluation must call this rather than re-deriving it. scripts/ has
    twice grown a private copy that silently drifted -- one used Platt-calibrated
    branches, which is a different model, and reported it as the shipped one.
    """
    probes = load_probes() if probes is None else probes
    # max in LOGIT space, then one sigmoid. Identical to max-of-sigmoids because
    # sigmoid is monotone, but nothing saturates to exactly 0 or 1 first, so the
    # shift cannot hit an infinity.
    z = np.max([v @ w + b for w, b in probes], axis=0)
    return 1 / (1 + np.exp(-(z - SHIFT)))


def find_images(paths_root) -> list:
    """Every supported image under `paths_root`, sorted, recursive.

    rglob because judges may hand over a nested tree; sorted because the output
    order is the only thing pairing a prediction with its image in a diff.
    """
    return sorted(p for p in Path(paths_root).rglob("*") if p.suffix.lower() in EXTS)


def to_records(paths, scores) -> list:
    """THE required output shape: two fields, posix paths, 4dp. Nothing else.

    as_posix so a Windows run and a Linux run emit the same strings -- judges may
    diff these files.
    """
    return [{"image_path": Path(p).as_posix(), "pred": round(float(v), 4)}
            for p, v in zip(paths, scores)]


def score_all(paths) -> np.ndarray:
    """max(P_synthetic, P_tampered) -- either one firing means AI touched it.

    ponytail: max, not a learned combiner, and measurement says keep it that
    way for now -- a fusion LR over the calibrated branch vector scored lower on
    both so_fake_ood and tampered. Far better than the general probe alone,
    which scores tampered images BELOW real ones (AUC 0.37) because a
    locally-edited photo is globally authentic.
    """
    probes = load_probes()
    emb, out = Embedder(), []
    for i in range(0, len(paths), BATCH):
        imgs = [Image.open(p).convert("RGB") for p in paths[i:i + BATCH]]
        out.append(score_embeddings(emb.embed_batch(imgs), probes))
        print(f"  {min(i + BATCH, len(paths))}/{len(paths)}")
    return np.concatenate(out)


def main(a):
    # a typo'd path and a genuinely empty one are different problems; without
    # this they produce the same "no images under" line and look like the same one
    root = Path(a.input_dir)
    if not root.is_dir():
        raise SystemExit(f"not a directory: {a.input_dir}")
    paths = find_images(root)
    if not paths:
        raise SystemExit(f"no images under {a.input_dir}")
    preds = to_records(paths, score_all(paths))
    Path(a.output).write_text(json.dumps(preds, indent=2))
    print(f"{len(preds)} predictions -> {a.output}")


def self_check():
    """The deliverable's contract, on synthetic data. No cache, no GPU, ~1s.

    Two things are checked that nothing else in the repo checks: the operating
    point actually lands on 0.5 after the shift, and the private copies of the
    scoring path in scripts/ still agree with score_embeddings(). The docstring
    above records that the second one has already drifted twice.
    """
    import sys
    import tempfile
    sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

    d = 768
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # -- the operating point. A raw combined score of exactly OPERATING_POINT must
    # come out at exactly 0.5; that equality IS the shift, and every threshold
    # claim in the docstring above rests on it.
    flat = [(np.zeros(d), float(np.log(OPERATING_POINT / (1 - OPERATING_POINT))))]
    v = np.random.default_rng(0).normal(size=(5, d))
    assert np.allclose(score_embeddings(v, flat), 0.5), score_embeddings(v, flat)

    # -- monotone, so AUROC is untouched. The whole "rank-based grading sees no
    # change" claim is this line.
    w = np.zeros(d); w[0] = 1.0
    ramp = np.zeros((7, d)); ramp[:, 0] = np.linspace(-30, 30, 7)
    s = score_embeddings(ramp, [(w, 0.0)])
    assert (np.diff(s) > 0).all(), s
    assert ((s > 0) & (s < 1)).all(), f"saturated to a hard 0/1: {s}"
    assert np.isfinite(s).all()

    # -- max in logit space == max of sigmoids, which is what the comment in
    # score_embeddings claims and what lets the shift be applied once.
    two = [(w, 0.0), (-w, 0.5)]
    both = score_embeddings(ramp, two)
    each = np.maximum(*[sigmoid(ramp @ a + b - SHIFT) for a, b in two])
    assert np.allclose(both, each), np.abs(both - each).max()

    # -- the shipped probes: right shape, right range, both branches present.
    probes = load_probes()
    assert len(probes) == 2, f"{len(probes)} probes -- tampered.npz missing?"
    assert all(pw.shape == (d,) for pw, _ in probes), [pw.shape for pw, _ in probes]
    real = score_embeddings(v / np.linalg.norm(v, axis=1, keepdims=True), probes)
    assert real.shape == (5,) and ((real >= 0) & (real <= 1)).all(), real

    # -- no private copy has drifted. pick_threshold scores the UNSHIFTED value on
    # purpose (it is picking the cut), so the two differ by exactly SHIFT in logit
    # space and by nothing else.
    from pick_threshold import shipped as pt_shipped
    u = v / np.linalg.norm(v, axis=1, keepdims=True)
    q = np.clip(pt_shipped(u), 1e-12, 1 - 1e-12)
    assert np.allclose(sigmoid(np.log(q / (1 - q)) - SHIFT),
                       score_embeddings(u, probes)), (
        "pick_threshold.shipped has drifted from score_embeddings")
    import make_figures
    assert make_figures.NEW_THR == OPERATING_POINT, (
        f"make_figures.NEW_THR {make_figures.NEW_THR} != predict {OPERATING_POINT}")

    # -- file discovery: nested, every supported extension, case-insensitive,
    # sorted, and nothing else picked up. README claims all of this.
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        (root / "sub" / "deep").mkdir(parents=True)
        want = [root / "sub" / "deep" / "d.BMP", root / "sub" / "b.jpeg",
                root / "a.jpg", root / "c.PNG", root / "e.webp"]
        for f in want:
            f.write_bytes(b"")
        for f in (root / "notes.txt", root / "anim.gif", root / "noext"):
            f.write_bytes(b"")
        got = find_images(root)
        assert got == sorted(want), [p.name for p in got]
        assert len(got) == len(EXTS), f"{len(got)} of {len(EXTS)} extensions found"

        # -- the output records. Two fields, posix separators even on Windows.
        r = to_records(got, np.linspace(0, 1, len(got)))
        assert all(set(x) == {"image_path", "pred"} for x in r), r[0]
        assert all("\\" not in x["image_path"] for x in r), r
        assert all(0.0 <= x["pred"] <= 1.0 for x in r), r
        assert to_records([Path("a/b.jpg")], [0.123456])[0]["pred"] == 0.1235
        assert json.loads(json.dumps(r)) == r, "not JSON-round-trippable"

    print(f"predict.py ok: operating point {OPERATING_POINT} -> 0.5 "
          f"(shift {SHIFT:.4f}), {len(probes)} probes, {len(EXTS)} extensions")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir")
    p.add_argument("--output", default="preds.json")
    p.add_argument("--self-check", action="store_true",
                   help="contract + scoring checks, no cache and no GPU")
    a = p.parse_args()
    if a.self_check:
        self_check()
    elif not a.input_dir:
        p.error("--input-dir is required (or pass --self-check)")
    else:
        main(a)
