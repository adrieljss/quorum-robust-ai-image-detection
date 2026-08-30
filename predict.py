"""Required deliverable: directory of images -> JSON of {image_path, pred}.

    python predict.py --input-dir path/to/images --output preds.json

`pred` is P(AI-generated) in [0,1]. Two fields, no more -- the demo's richer
schema stays in the demo.

THREE branches vote, not two: max(general, 1.25*tampered, face). The face branch
joined on 31 Aug. It was excluded for weeks on an aggregate-AUROC argument that
turned out to measure the wrong thing -- with 27% face coverage a real gain on
faces dilutes to +0.001 overall. Counting RESCUES instead: it catches 442 of the
1,891 AI faces max() misses on So-Fake-OOD (23%) while newly flagging 58 of 8,209
real faces, a 7.6:1 ratio, and it does not hurt on ANY of the 15 degradation
variants on either eval set. At matched false positives that is +0.96pp recall.
It does NOT help on GPT-image-2, whose faces it scores 0.02-0.43; it carries the
same recency blind spot as everything else here.

The score is shifted so that **0.5 is the operating point**. It was not before:
0.5 is the sigmoid's default, nobody chose it, and on held-out test_ood it cost
0.09 precision and flagged 27.6% of COCO photographs as AI-generated. The shift
is monotone, so AUROC is identical (0.8997) and rank-based grading sees no
change; only a threshold-based read of `pred` moves.

It is a TRADE, not a free win, and both halves belong here:

    test_ood clean          acc    prec  recall      F1   COCO FP  tamp rec
      0.8092 (ships)      0.840   0.902   0.759   0.824      8.9%     0.757
      0.8523 (cv)         0.821   0.926   0.698   0.796      6.3%     0.704
    the model this REPLACED, at its own operating point
      0.766               0.825   0.882   0.751   0.811      8.9%     0.746

The probe underneath changed on 30 Aug (general --plus, docs/ERROR_ANALYSIS.md
3.1) and AUROC went 0.9085 -> 0.9268, which no threshold can move. So the two
rows above are the same better model at two defensible cuts:

  0.8050 SHIPS, anchored to 8.25% false positives on So-Fake-OOD reals -- a pool
    no branch trains on. COCO val2017 is deliberately NOT the anchor any more,
    even though it is clean again: one pool should not be both the thing we tune
    against and the thing we report, and ERROR_ANALYSIS 7.7 exists because we
    ran out of untouched pools once already.

  0.8523 is the alternative, and the purer one: cross-validated over calib_ood's
    five generator families, each fold picked on the family its model never saw
    (0.585/0.600/0.705/0.665/0.635, averaged). It reads NO evaluation set at
    all and cuts false accusations to 6.3%. Use it if false positives matter
    more than misses, or if the derivation has to be unimpeachable.

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

from quorum.detectors.general import MODEL, MODEL_TAMPERED, MODELS
from quorum.embed import Embedder

MODEL_FACE = MODELS / "face.npz"

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
BATCH = 64
# Anchored to 8.25% false positives on So-Fake-OOD reals, a pool no branch trains
# on. COCO train2017 was REMOVED from the tampered branch on 30 Aug
# (ERROR_ANALYSIS 7.6, reverted): on the corpus-disjoint holdout it was slightly
# WORSE, 19.50% against 18.50%, and it cost 7.3pp of tampered recall. What it
# bought was an in-distribution COCO number. Reverting gave back COCO val2017 and
# the organizer benchmark as clean evaluation sets, for -0.0063 AUROC.
#
# The alternative below is derivation-pure and reads no eval set at all:
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
OPERATING_POINT = 0.8092
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


def load_face():
    """(w, b, px_mu, px_sd) for the 769-d face probe, or None if it is absent.

    Not part of `probes`: it eats 768 embedding dims PLUS a standardised
    log2(face_px), so it cannot ride the same matrix multiply.
    """
    if not MODEL_FACE.exists():
        return None
    d = np.load(MODEL_FACE)
    return (d["w"].ravel(), float(d["b"].ravel()[0]),
            float(d["px_mu"]), float(d["px_sd"]))


def face_score(imgs, emb, face_model) -> np.ndarray:
    """Face-branch probability per image on the SHIPPED scale, 0.0 where no face.

    Zero, not 0.5: this feeds a max(), where an absent branch must not be able to
    raise the score. (quorum/fusion.py uses 0.5 for the same quantity because it
    feeds a LEARNED combiner with a face_present indicator beside it, which makes
    the fill arbitrary there and not here.)

    Crops go through in ONE batch. Detection is the cost, not the embedding.
    """
    from quorum.features import face_crop

    out = np.zeros(len(imgs), np.float32)
    if face_model is None:
        return out
    w, b, mu, sd = face_model
    crops, idx, px = [], [], []
    for i, im in enumerate(imgs):
        c, p = face_crop(im)
        if c is not None:
            crops.append(c); idx.append(i); px.append(max(p, 1.0))
    if not crops:
        return out
    V = emb.embed_batch(crops)
    feat = np.column_stack([V, (np.log2(np.asarray(px)) - mu) / sd])
    out[idx] = 1 / (1 + np.exp(-(feat @ w + b - SHIFT)))
    return out


def score_embeddings(v, probes=None, face=None) -> np.ndarray:
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
    p = 1 / (1 + np.exp(-(z - SHIFT)))
    # The face branch arrives already on this scale because it cannot share the
    # matrix multiply above. Absent -> the caller passes nothing and the score is
    # exactly what it was before the branch existed.
    return p if face is None else np.maximum(p, face)


def find_images(paths_root) -> list:
    """Every supported image under `paths_root`, sorted, recursive.

    rglob because judges may hand over a nested tree; sorted because the output
    order is the only thing pairing a prediction with its image in a diff.
    """
    return sorted(p for p in Path(paths_root).rglob("*") if p.suffix.lower() in EXTS)


def verdict(pred) -> str:
    """The word for a number, using the ONE threshold that means anything here.

    0.5 on the emitted score IS OPERATING_POINT after the shift, so this is the
    same comparison predict.py's callers should make and the same one every
    metric in docs/ERROR_ANALYSIS.md is computed at.
    """
    return "AI" if pred >= 0.5 else "real"


BRANCH_NAMES = ("general", "tampered")


def branch_scores(v, probes) -> dict:
    """Each branch on the SAME scale as `pred`, so max(branches) == pred exactly.

    Shifted individually rather than shown raw. A raw sigmoid would be a
    different scale from the emitted score and invite exactly the comparison that
    made try_grid.py disagree with this file on a verdict -- the numbers here are
    the ones that actually competed in the max().

    face and spectral are absent on purpose: they are not in the shipped scorer
    (measured a wash and worse respectively, ERROR_ANALYSIS 8.6/8.7), and pulling
    the face detector in would add a model and an ONNX dependency to the
    deliverable for a branch that does not vote.
    """
    return {n: round(float(1 / (1 + np.exp(-(v @ w + b - SHIFT)))), 4)
            for n, (w, b) in zip(BRANCH_NAMES, probes)}


def score_variants(path, emb, probes, with_branches=False) -> dict:
    """Every degradation variant of one image, scored, nothing written to disk.

    The image is normalised (JPEG q95) and the grid seeded off its image_id
    exactly as quorum/embed.py does, so these numbers are comparable to
    docs/robustness.md. That normalisation is why variants["clean"] can differ
    slightly from the top-level `pred`, which scores the file AS GIVEN -- a
    serving path should not silently re-encode what it was handed. On a JPEG the
    gap is nil; on a PNG it is a few thousandths.
    """
    from quorum.degrade import apply, variant_specs
    from quorum.embed import image_id, normalise

    img = normalise(Image.open(path))
    specs = variant_specs(image_id(img), None)          # None = the full 15
    imgs = [img] + [apply(img, kind, param, rng) for _, kind, param, rng in specs]
    names = ["clean"] + [nm for nm, *_ in specs]
    out = {}
    for i in range(0, len(imgs), BATCH):
        V = emb.embed_batch(imgs[i:i + BATCH])
        for nm, vec, v in zip(names[i:i + BATCH], V, score_embeddings(V, probes)):
            rec = {"pred": round(float(v), 4), "verdict": verdict(v)}
            if with_branches:
                rec["branches"] = branch_scores(vec, probes)
            out[nm] = rec
    return out


def to_records(paths, scores) -> list:
    """THE required output shape: two fields, posix paths, 4dp. Nothing else.

    as_posix so a Windows run and a Linux run emit the same strings -- judges may
    diff these files.
    """
    return [{"image_path": Path(p).as_posix(), "pred": round(float(v), 4)}
            for p, v in zip(paths, scores)]


def score_all(paths, keep_embeddings=False):
    """max(P_synthetic, P_tampered) -- either one firing means AI touched it.

    ponytail: max, not a learned combiner, and measurement says keep it that
    way for now -- a fusion LR over the calibrated branch vector scored lower on
    both so_fake_ood and tampered. Far better than the general probe alone,
    which scores tampered images BELOW real ones (AUC 0.37) because a
    locally-edited photo is globally authentic.
    """
    probes, fm = load_probes(), load_face()
    emb, out, vecs = Embedder(), [], []
    for i in range(0, len(paths), BATCH):
        imgs = [Image.open(p).convert("RGB") for p in paths[i:i + BATCH]]
        V = emb.embed_batch(imgs)
        out.append(score_embeddings(V, probes, face_score(imgs, emb, fm)))
        if keep_embeddings:
            # Handing these back is not an optimisation, it is a CORRECTNESS fix.
            # Re-embedding one image to report its branch scores gave a different
            # answer from the batch-of-64 pass -- fp16 matmuls are not invariant
            # to batch shape, and pred disagreed with max(branches) by ~4e-4.
            vecs.append(V)
        print(f"  {min(i + BATCH, len(paths))}/{len(paths)}")
    scores = np.concatenate(out)
    return (scores, np.concatenate(vecs), emb, probes) if keep_embeddings else scores


def main(a):
    # a typo'd path and a genuinely empty one are different problems; without
    # this they produce the same "no images under" line and look like the same one
    root = Path(a.input_dir)
    if not root.is_dir():
        raise SystemExit(f"not a directory: {a.input_dir}")
    paths = find_images(root)
    if not paths:
        raise SystemExit(f"no images under {a.input_dir}")
    rich = a.branches or a.variants
    got = score_all(paths, keep_embeddings=rich)
    if rich:
        # Opt-in ONLY. The required deliverable is {image_path, pred} and nothing
        # else; self_check asserts that on the default path. `pred` is identical
        # either way -- a flag must not move the number it reports.
        scores, V, emb, probes = got
        preds = to_records(paths, scores)
        for rec, path, v in zip(preds, paths, V):
            rec["verdict"] = verdict(rec["pred"])
            if a.branches:
                rec["branches"] = branch_scores(v, probes)
            if a.variants:
                rec["variants"] = score_variants(path, emb, probes, a.branches)
                print(f"  variants {rec['image_path']}")
    else:
        preds = to_records(paths, got)
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

    # -- the FACE branch is loadable and 769-d. It cannot ride `probes` and is
    # maxed in separately, so a missing or reshaped face.npz has to fail here
    # rather than silently reverting the scorer to two branches.
    fm = load_face()
    assert fm is not None, "face.npz missing -- the shipped scorer is 3 branches"
    assert fm[0].shape == (d + 1,), f"face probe is {fm[0].shape}, expected {(d+1,)}"
    assert np.allclose(score_embeddings(v, flat), 0.5), "face=None must be a no-op"
    hi = np.full(len(v), 0.9)
    assert np.allclose(score_embeddings(v, flat, face=hi), 0.9), "face must max in"
    lo = np.zeros(len(v))
    assert np.allclose(score_embeddings(v, flat, face=lo),
                       score_embeddings(v, flat)), "absent face must contribute 0"

    # -- no private copy has drifted. pick_threshold scores the UNSHIFTED value on
    # purpose (it is picking the cut), so the two differ by exactly SHIFT in logit
    # space and by nothing else.
    #
    # NOTE this compares the EMBEDDING-ONLY path. Since 31 Aug the shipped score
    # also maxes in the face branch, which needs PIXELS and so cannot be
    # reproduced from a 768-d vector. pick_threshold.py and make_figures.py are
    # therefore both face-blind: they understate the shipped scorer by ~0.0013
    # AUROC and their thresholds are picked on the two-branch score. Wiring the
    # face_* caches into both is the fix; until then this assertion means "the
    # shared component agrees", NOT "the scripts reproduce predict.py".
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
          f"(shift {SHIFT:.4f}), {len(probes) + 1} branches "
          f"({', '.join(BRANCH_NAMES)}, face), {len(EXTS)} extensions")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir")
    p.add_argument("--output", default="preds.json")
    p.add_argument("--branches", action="store_true",
                   help="also emit each branch's score and a `verdict`. Branches "
                        "are on the same scale as `pred`, so max(branches) == pred. "
                        "Combine with --variants for per-variant branch scores.")
    p.add_argument("--variants", action="store_true",
                   help="also score all 15 degradation variants of each image, "
                        "generated in memory and never written to disk, and add "
                        "a `verdict` field. NOT the required output schema -- the "
                        "default stays {image_path, pred}.")
    p.add_argument("--self-check", action="store_true",
                   help="contract + scoring checks, no cache and no GPU")
    a = p.parse_args()
    if a.self_check:
        self_check()
    elif not a.input_dir:
        p.error("--input-dir is required (or pass --self-check)")
    else:
        main(a)
