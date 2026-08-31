"""Text branch: OCR features -> logistic probe. PIPELINE.md section 2.3.

Six hand-built features, standardised, one logistic regression: 7 parameters.
The OCR model itself is RapidOCR PP-OCRv4 (~10M, ONNX, CPU) and is frozen --
it reads glyphs, it does not decide anything.

**Why this branch exists**, measured before it was built rather than assumed:
text-heavy images are the shipped detector's WORST content class on both
benchmarks, by CLIP zero-shot content label.

    so_fake_ood    text 0.8603 clean vs 0.9167 for the other four   -0.0564
    organizer_val  text 0.9094 clean vs 0.9514 for the other four   -0.0420

That gap is what this branch exists to close, and `evaluate()` reports against
it rather than against a standalone AUROC that answers nothing.

**Two honest limits, both known going in:**

1. TextFake finds GPT-Image-2 renders text WELL (70% entity OCR hit rate) and is
   the hardest generator to catch, while low-fidelity generators are easy. So
   this feature is high precision with FALLING recall against frontier models --
   it decays as generators improve. TODO-FACE.md T5.
2. The multilingual question (T2). An English dictionary hit rate scores ~0 on
   Chinese for real AND fake, and `frac_nonascii` becomes a language detector
   rather than an artifact detector. Resolved by making feature 3
   script-agnostic; feature 5 is kept but is only honest on Latin-dominant sets,
   which ours are. Say so in the README.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from functools import lru_cache

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from quorum.detectors.general import MODELS, MANIFEST

MODEL = MODELS / "text.npz"
OCR_CACHE = MODELS.parent / "cache" / "ocr"
IMAGES = MODELS.parent / "raw" / "images"

# Order is frozen: the saved model's standardisation stats are positional.
FEATURES = ("conf_mean", "conf_std", "token_plausibility",
            "glyph_consistency", "frac_nonascii", "n_regions")
N_FEAT = len(FEATURES)


@lru_cache(maxsize=1)
def _ocr():
    from rapidocr_onnxruntime import RapidOCR
    return RapidOCR()


def text_features(img):
    """(float32[6], text_present). Zeros + False when OCR finds nothing.

    Neutral fill plus a presence flag -- "no text here" must never reach a
    combiner as "the text model says real".
    """
    res, _ = _ocr()(np.asarray(img.convert("RGB")))
    if not res:
        return np.zeros(N_FEAT, dtype=np.float32), False

    # RapidOCR rows are [box(4x2), text, confidence]; confidence arrives as str.
    conf = np.array([float(r[2]) for r in res], dtype=np.float32)
    words = [str(r[1]) for r in res]
    boxes = [np.asarray(r[0], dtype=np.float32) for r in res]

    # 3. Token plausibility, NOT an English dictionary. Garbled generated text
    # decodes to single characters and punctuation soup; real signage decodes to
    # multi-character alphanumeric tokens. Works in any script, which an English
    # word list does not -- see the multilingual note in the module docstring.
    plausible = [w for w in words if len(w.strip()) >= 2
                 and all(c.isalnum() or c.isspace() for c in w)]
    token_plausibility = len(plausible) / len(words)

    # 4. Glyph consistency. RapidOCR gives region boxes, not per-character boxes,
    # so "same char, same shape" is not directly observable. Per-character
    # ASPECT RATIO is: a real font renders at a consistent width:height across
    # every region of a scene, and generated lettering does not. Reported as a
    # coefficient of variation folded to [0,1], where 1 is perfectly consistent.
    ar = []
    for b, w in zip(boxes, words):
        n = max(len(w.strip()), 1)
        hw = np.linalg.norm(b[1] - b[0]) / n          # per-character width
        hh = np.linalg.norm(b[3] - b[0])              # region height
        if hh > 1e-3:
            ar.append(hw / hh)
    ar = np.asarray(ar, dtype=np.float32)
    if len(ar) < 2 or ar.mean() <= 1e-6:
        glyph_consistency = 1.0                       # one region cannot disagree
    else:
        glyph_consistency = float(1.0 / (1.0 + ar.std() / ar.mean()))

    joined = "".join(words)
    frac_nonascii = (sum(ord(c) > 127 for c in joined) / len(joined)) if joined else 0.0

    # 6. log1p, not the raw count. The contract is six floats, not six raw
    # statistics, and a 1-200 column next to five [0,1] columns would dominate a
    # linear model's gradient before standardisation ever sees it.
    return np.array([conf.mean(), conf.std(), token_plausibility,
                     glyph_consistency, frac_nonascii,
                     np.log1p(len(res))], dtype=np.float32), True


# --------------------------------------------------------------- cache ------
def manifest_labels(source):
    """image_id -> label for one source. Manifest-first: never trust a path."""
    m = pd.read_csv(MANIFEST, usecols=["image_id", "source", "label"])
    m = m[m.source == source].drop_duplicates("image_id")
    return dict(zip(m.image_id, m.label))


def build_cache(source, paths, ids=None):
    """OCR every path once, keyed by image_id. By far the slowest thing here.

    `ids` supplies the image_id per path when the filename is not already one.
    Saved images from stream_embed.py --save-images ARE named by their id and
    must be trusted rather than re-hashed: JPEG q95 is not idempotent, so a
    second round-trip moves 13-53% of pixels and the join silently empties.
    """
    OCR_CACHE.mkdir(parents=True, exist_ok=True)
    out = OCR_CACHE / f"{source}.parquet"
    done = set()
    if out.exists():
        done = set(pd.read_parquet(out).image_id)

    ids = ids or [Path(p).stem for p in paths]
    todo = [(i, p) for i, p in zip(ids, paths) if i not in done]
    print(f"{source}: {len(done)} cached, {len(todo)} to OCR")

    def commit(rows):
        """Fold `rows` into the parquet on disk. Called periodically, not once
        at the end: OCR is the slowest thing in this project and an hour of it
        must not die with the process. Same reason ShardWriter flushes."""
        if not rows:
            return
        df = pd.DataFrame(rows)
        if out.exists():
            df = pd.concat([pd.read_parquet(out), df], ignore_index=True)
        df.drop_duplicates("image_id").to_parquet(out, index=False)

    rows = []
    for n, (iid, p) in enumerate(todo, 1):
        try:
            f, present = text_features(Image.open(p))
        except Exception as e:                       # a corrupt file must not
            print(f"  skip {p}: {type(e).__name__}", flush=True)  # kill the pass
            continue
        rows.append({"image_id": iid, "text_present": present,
                     **dict(zip(FEATURES, f.tolist()))})
        if n % 500 == 0:
            commit(rows)
            rows = []
            print(f"  {n}/{len(todo)}", flush=True)
    commit(rows)
    if out.exists():
        print(f"  -> {out} ({len(pd.read_parquet(out))} rows)", flush=True)
    return out


def source_paths(source):
    """(paths, ids) for the two on-disk pixel layouts.

    sid_train comes from stream_embed.py --save-images, so its filenames ARE the
    ids. organizer_val is the original download, so its ids have to be recomputed
    the same way the embedding pass did -- normalise() then hash. Verified 8/8
    against the manifest on both halves before this was relied on.
    """
    if source == "organizer_val":
        from quorum.embed import image_id, normalise
        root = MODELS.parent / "raw" / "organizer_val"
        paths = sorted(p for d in ("coco_val2017", "wildfake_dalle_adv")
                       for p in (root / d).rglob("*")
                       if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
        return paths, [image_id(normalise(Image.open(p))) for p in paths]
    paths = sorted((IMAGES / source).glob("*.jpg"))
    return paths, [p.stem for p in paths]


def load(source, present_only=True):
    """(X[n,6], rows) joined to manifest labels. Only rows OCR found text in.

    present_only because a probe fitted across text-free images learns "does
    this image contain text", which is a content classifier, not a detector.
    """
    df = pd.read_parquet(OCR_CACHE / f"{source}.parquet")
    lab = manifest_labels(source)
    df = df[df.image_id.isin(lab)].copy()
    df["label"] = df.image_id.map(lab)
    if present_only:
        df = df[df.text_present]
    df = df.reset_index(drop=True)
    return df[list(FEATURES)].to_numpy(np.float32), df


# ---------------------------------------------------------------- probe -----
def stats(X):
    """Standardisation stats, saved with the model. Same pattern as face.py's
    px_stats: the columns have wildly different scales and the L2 penalty would
    otherwise fit whichever one is largest."""
    return X.mean(0), X.std(0) + 1e-6


def fit(X, y):
    st = stats(X)
    return LogisticRegression(max_iter=2000).fit((X - st[0]) / st[1], y), st


def score(clf, st, X):
    return clf.predict_proba((X - st[0]) / st[1])[:, 1]


def save(clf, st, path=MODEL):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, w=clf.coef_.reshape(1, -1), b=np.asarray(clf.intercept_).reshape(1),
             mu=st[0], sd=st[1])


# ============================================================================
# Attempt 2: CLIP on text crops. OCR is a DETECTOR here, never a sensor.
#
# Attempt 1 (above) failed because six hand-built statistics about what OCR
# *reported* tracked text composition, not text deformation -- five of six
# features reverse direction between SID_Set and the organizer set, so the probe
# transfers at 0.4627, below chance. Nothing about the decoded string or its
# confidence is used below; OCR only says WHERE the glyphs are.
#
# This is the face branch's recipe, which is the one pattern in this repo known
# to transfer across datasets: detect region -> crop -> shared CLIP -> linear
# probe on 768 + 1 standardised size column. Face holds 0.9421/0.9168 on
# so_fake_ood and 0.9520/0.8887 on organizer_val, no sign flip.
# ============================================================================
# ATTEMPT 2's model, and a deliberately different filename from MODEL above.
# Attempt 1 transfers at 0.4627 -- BELOW CHANCE -- so the two must never be
# confused for one another by a loader reaching for "the text probe".
CROP_MODEL = MODELS / "text_crop.npz"
CROPS = MODELS.parent / "raw" / "text_crops"
CROP_H = 224          # CLIP's input side; the strip's HEIGHT, not its width
MAX_TILES = 3


def _order(pts):
    """4 points -> [top-left, top-right, bottom-right, bottom-left].

    RapidOCR's point order is not guaranteed across detector versions, and a
    mis-ordered quad makes getPerspectiveTransform mirror or rotate the crop.
    Derive the order from geometry instead of trusting the input.
    """
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.array([pts[s.argmin()], pts[d.argmin()],
                     pts[s.argmax()], pts[d.argmax()]], dtype=np.float32)


def text_strip(img):
    """(warped RGB strip, region height in source px) or (None, 0.0).

    Largest region only -- PIPELINE.md section 5's face rule, for the same
    reason: pooling many regions by averaging would wash out the one that is
    malformed.

    Warped to a fixed HEIGHT with aspect ratio preserved, never squashed to a
    square. A 200x30 strip forced into 224x224 distorts every glyph, which is
    precisely the signal this branch exists to read.
    """
    import cv2

    rgb = np.asarray(img.convert("RGB"))
    res, _ = _ocr()(rgb)
    if not res:
        return None, 0.0

    pts = max((np.asarray(r[0], dtype=np.float32) for r in res),
              key=lambda q: cv2.contourArea(_order(q)))
    b = _order(pts)
    w = max(np.linalg.norm(b[1] - b[0]), np.linalg.norm(b[2] - b[3]))
    h = max(np.linalg.norm(b[3] - b[0]), np.linalg.norm(b[2] - b[1]))
    if h < 8.0 or w < 8.0:                      # too small to carry a glyph
        return None, 0.0

    W = int(np.clip(round(CROP_H * w / h), CROP_H // 4, CROP_H * MAX_TILES))
    dst = np.array([[0, 0], [W, 0], [W, CROP_H], [0, CROP_H]], dtype=np.float32)
    out = cv2.warpPerspective(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                              cv2.getPerspectiveTransform(b, dst), (W, CROP_H),
                              flags=cv2.INTER_LINEAR)
    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)), float(h)


def tiles(strip):
    """Strip -> up to MAX_TILES square 224 crops, evenly spaced.

    Square tiles keep the aspect ratio CLIP was trained on. A strip narrower
    than one tile is padded rather than stretched.
    """
    w, h = strip.size
    if w <= CROP_H:
        pad = Image.new("RGB", (CROP_H, CROP_H), (255, 255, 255))
        pad.paste(strip, ((CROP_H - w) // 2, 0))
        return [pad]
    n = min(MAX_TILES, max(1, round(w / CROP_H)))
    xs = np.linspace(0, w - CROP_H, n).round().astype(int)
    return [strip.crop((x, 0, x + CROP_H, CROP_H)) for x in xs]


def build_crops(source, paths, ids=None):
    """OCR every image once and save the warped strip. Slowest step; resumable.

    Strips are saved rather than embeddings so the GPU pass can be re-run --
    retiling or swapping the pooling rule must not cost another OCR sweep.
    """
    d = CROPS / source
    d.mkdir(parents=True, exist_ok=True)
    meta = CROPS / f"{source}.parquet"
    done = set(pd.read_parquet(meta).image_id) if meta.exists() else set()

    ids = ids or [Path(p).stem for p in paths]
    todo = [(i, p) for i, p in zip(ids, paths) if i not in done]
    print(f"{source}: {len(done)} cached, {len(todo)} to OCR", flush=True)

    def commit(rows):
        if not rows:
            return
        df = pd.DataFrame(rows)
        if meta.exists():
            df = pd.concat([pd.read_parquet(meta), df], ignore_index=True)
        df.drop_duplicates("image_id").to_parquet(meta, index=False)

    rows = []
    for n, (iid, p) in enumerate(todo, 1):
        try:
            strip, px = text_strip(Image.open(p))
        except Exception as e:
            print(f"  skip {p}: {type(e).__name__}", flush=True)
            continue
        if strip is not None:
            strip.save(d / f"{iid}.jpg", "JPEG", quality=95)
        rows.append({"image_id": iid, "text_present": strip is not None,
                     "text_px": px})
        if n % 500 == 0:
            commit(rows); rows = []
            print(f"  {n}/{len(todo)}", flush=True)
    commit(rows)
    print(f"  -> {meta}", flush=True)
    return meta


def embed_crops(source, batch=64):
    """Tile every saved strip, embed with the SHARED CLIP, mean-pool, renorm.

    Mean over tiles, not max: the tiles are mechanical slices of ONE text
    region, not independent subjects, so the face branch's never-average rule
    does not apply here -- that rule is about multiple faces.
    """
    from quorum.embed import Embedder

    meta = pd.read_parquet(CROPS / f"{source}.parquet")
    meta = meta[meta.text_present].reset_index(drop=True)
    emb, out, keep = Embedder(), [], []
    for i in range(0, len(meta), batch):
        chunk = meta.iloc[i:i + batch]
        flat, spans = [], []
        for iid in chunk.image_id:
            f = CROPS / source / f"{iid}.jpg"
            if not f.exists():
                spans.append(0)
                continue
            t = tiles(Image.open(f).convert("RGB"))
            spans.append(len(t))
            flat.extend(t)
        if not flat:
            continue
        V = emb.embed_batch(flat)
        k = 0
        for iid, n in zip(chunk.image_id, spans):
            if n == 0:
                continue
            v = V[k:k + n].mean(0)
            out.append(v / (np.linalg.norm(v) + 1e-9))
            keep.append(iid)
            k += n
        print(f"  {min(i + batch, len(meta))}/{len(meta)}", flush=True)
    X = np.stack(out).astype(np.float32)
    np.save(CROPS / f"{source}_emb.npy", X)
    pd.DataFrame({"image_id": keep}).to_parquet(CROPS / f"{source}_emb.parquet",
                                                index=False)
    print(f"  -> {X.shape}", flush=True)
    return X


def load_crops(source):
    """(X[n,769], rows). 768 CLIP dims + standardised log2(text_px).

    The size column is the face branch's face_px argument transplanted: a 20px
    strip upscaled to 224 carries far harsher effective degradation than a 200px
    one downscaled to it, and without the column the probe cannot separate them.
    """
    X = np.load(CROPS / f"{source}_emb.npy")
    ids = pd.read_parquet(CROPS / f"{source}_emb.parquet").image_id.values
    meta = pd.read_parquet(CROPS / f"{source}.parquet").set_index("image_id")
    lab = manifest_labels(source)
    px = meta.loc[ids, "text_px"].values.astype(np.float32)
    df = pd.DataFrame({"image_id": ids, "text_px": px})
    df["label"] = df.image_id.map(lab)
    m = df.label.notna().values
    return X[m], df[m].reset_index(drop=True)


def crop_design(X, rows, st):
    z = (np.log2(np.maximum(rows.text_px.values.astype(np.float32), 1.0)) - st[0]) / st[1]
    return np.hstack([X, z[:, None]]).astype(np.float32)


def crop_px_stats(rows):
    z = np.log2(np.maximum(rows.text_px.values.astype(np.float32), 1.0))
    return float(z.mean()), float(z.std()) or 1.0


def save_crops(train_source="sid_train", path=CROP_MODEL):
    """Fit attempt 2 on one corpus and save it. DISPLAY ONLY -- see __main__.

    Same shape as face.npz (w[1,769], b, px_mu, px_sd) because it is the same
    design: 768 CLIP dims plus one standardised log2(size) column. Fitted on
    sid_train alone, which is the configuration evaluate_crops() measures
    transferring at 0.8083 to organizer_val -- so the saved probe is the one
    with a published out-of-corpus number, not a better-looking pooled fit
    nobody has a transfer figure for.
    """
    X, R = load_crops(train_source)
    st = crop_px_stats(R)
    clf = LogisticRegression(max_iter=2000).fit(crop_design(X, R, st),
                                                R.label.values.astype(int))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, w=clf.coef_.reshape(1, -1), b=np.asarray(clf.intercept_).reshape(1),
             px_mu=np.float32(st[0]), px_sd=np.float32(st[1]))
    return clf, st


def evaluate_crops(train_source="sid_train", eval_source="organizer_val"):
    """The gate: does a probe fitted on one dataset transfer to the other?

    Attempt 1 scored 0.6789 in-distribution and 0.4627 across, because its
    features tracked composition. That comparison is run FIRST here, before any
    degradation grid or combiner test, because it is the cheap check that
    decides whether anything else is worth measuring.
    """
    from sklearn.model_selection import cross_val_predict

    from predict import score_embeddings
    from quorum.detectors.general import load as load_emb
    from quorum.fusion import CONTENT, content_onehot

    Xa, Ra = load_crops(train_source)
    Xb, Rb = load_crops(eval_source)
    st = crop_px_stats(Ra)
    A, B = crop_design(Xa, Ra, st), crop_design(Xb, Rb, st)
    ya, yb = Ra.label.values.astype(int), Rb.label.values.astype(int)

    def cv(X, y):
        p = cross_val_predict(LogisticRegression(max_iter=2000), X, y, cv=5,
                              method="predict_proba")[:, 1]
        return roc_auc_score(y, p)

    print(f"train {train_source}: {len(A):,} text crops ({ya.mean():.1%} AI)")
    print(f"eval  {eval_source}: {len(B):,} text crops ({yb.mean():.1%} AI)\n")
    print(f"  in-distribution CV, {train_source:14s} {cv(A, ya):.4f}")
    print(f"  in-distribution CV, {eval_source:14s} {cv(B, yb):.4f}")

    clf = LogisticRegression(max_iter=2000).fit(A, ya)
    p = clf.predict_proba(B)[:, 1]
    print(f"  CROSS-DATASET transfer            {roc_auc_score(yb, p):.4f}"
          f"   <- the gate (attempt 1: 0.4627)")

    # Does the size column earn its place, as face_px does for the face branch?
    c768 = LogisticRegression(max_iter=2000).fit(Xa, ya)
    print(f"  cross-dataset, 768 only (no size) "
          f"{roc_auc_score(yb, c768.predict_proba(Xb)[:, 1]):.4f}\n")

    # Same question attempt 1 was asked: does it close the content gap?
    Xe, Re = load_emb(eval_source)
    m = (Re.variant == "clean").values
    Xe, Re = Xe[m], Re[m].reset_index(drop=True)
    Re = Re.assign(shipped=score_embeddings(Xe),
                   content=np.array(CONTENT)[content_onehot(Xe).argmax(1)])
    D = Rb.assign(text_score=p).merge(Re[["image_id", "shipped", "content"]],
                                      on="image_id", how="inner")

    def lg(q):
        q = np.clip(q, 1e-6, 1 - 1e-6)
        return np.log(q / (1 - q))

    rows = {}
    for name, k in (("all text-bearing", np.ones(len(D), bool)),
                    ("CLIP content == text", (D.content == "text").values)):
        y = D.label.values[k].astype(int)
        if pd.Series(y).nunique() < 2:
            continue
        both = np.maximum(lg(D.shipped.values[k]), lg(D.text_score.values[k]))
        rows[name] = {"n": int(k.sum()),
                      "text alone": roc_auc_score(y, D.text_score.values[k]),
                      "shipped": roc_auc_score(y, D.shipped.values[k]),
                      "max(both)": roc_auc_score(y, both)}
    out = pd.DataFrame(rows).T
    out["delta"] = out["max(both)"] - out["shipped"]
    return out


def evaluate(train_source="sid_train", eval_source="organizer_val"):
    """Does this branch close the content gap it was built for?

    Reports AUROC **and coverage** together, the same discipline the face branch
    uses and for the same reason: a score computed over a shrinking population
    is not comparable to one computed over all of it.

    Clean variant only. The 15-variant grid is another OCR pass over 15x the
    images (~9h); measure whether the branch is worth anything first.
    """
    from predict import score_embeddings
    from quorum.detectors.general import load as load_emb
    from quorum.fusion import CONTENT, content_onehot

    Xtr, Rtr = load(train_source)
    clf, st = fit(Xtr, Rtr.label.values)

    X, R = load(eval_source)
    R = R.copy()
    R["text_score"] = score(clf, st, X)

    # The shipped score on the SAME images, from cached embeddings.
    Xe, Re = load_emb(eval_source)
    m = (Re.variant == "clean").values
    Xe, Re = Xe[m], Re[m].reset_index(drop=True)
    Re = Re.assign(shipped=score_embeddings(Xe),
                   content=np.array(CONTENT)[content_onehot(Xe).argmax(1)])
    D = R.merge(Re[["image_id", "shipped", "content"]], on="image_id", how="inner")

    print(f"train {train_source}: {len(Xtr):,} text-bearing images "
          f"({Rtr.label.mean():.1%} AI)")
    print(f"eval  {eval_source}: {len(D):,} of {m.sum():,} clean images carry "
          f"text  (coverage {len(D) / m.sum():.1%})\n")

    def auc(y, s):
        return roc_auc_score(y, s) if pd.Series(y).nunique() == 2 else np.nan

    rows = {}
    for name, k in (("all text-bearing", np.ones(len(D), bool)),
                    ("CLIP content == text", (D.content == "text").values)):
        y = D.label.values[k]
        if pd.Series(y).nunique() < 2:
            continue
        # max in logit space, the shipped combiner's rule, so the comparison is
        # against how this would actually be wired rather than a better one.
        lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        both = np.maximum(lg(D.shipped.values[k]), lg(D.text_score.values[k]))
        rows[name] = {"n": int(k.sum()), "AI%": float(y.mean()),
                      "text alone": auc(y, D.text_score.values[k]),
                      "shipped": auc(y, D.shipped.values[k]),
                      "max(both)": auc(y, both)}
    out = pd.DataFrame(rows).T
    out["delta"] = out["max(both)"] - out["shipped"]
    return out


# --------------------------------------------------------------------------
# The 15-variant grid. THE measurement that decides whether this branch can
# ship: everything above is clean-only, and OCR is a fragile front end. If
# detection collapses under blur20/noise01 the branch goes neutral exactly when
# the image is degraded, which is when the brief is testing us.
#
# Two numbers per variant, and the first matters more than the second:
#   detection rate -- of images with text when CLEAN, how many still yield a
#                     strip after the transform. A branch that cannot SEE the
#                     text has no opinion, however good its AUROC on survivors.
#   AUROC          -- on the survivors only, so it is conditional on detection.
GRID_SOURCE = "organizer_val_grid"
GRID_SEP = "__"


def grid_ids(n=500, source="organizer_val", seed=0):
    """n image_ids that HAVE text when clean, label-stratified.

    Sampled from the clean crop cache rather than the manifest: an image with no
    text when clean can only stay absent under degradation, so it would measure
    nothing and cost 15 OCR calls to say so.
    """
    meta = pd.read_parquet(CROPS / f"{source}.parquet")
    meta = meta[meta.text_present].copy()
    lab = manifest_labels(source)
    meta["label"] = meta.image_id.map(lab)
    meta = meta[meta.label.notna()]
    rng = np.random.default_rng(seed)
    take = []
    for v, g in meta.groupby("label"):                 # stratified, not just head
        take.append(g.iloc[rng.permutation(len(g))[:n // 2]])
    out = pd.concat(take).image_id.tolist()
    print(f"{len(out)} ids ({n // 2} per label) from {len(meta)} text-present")
    return out


def build_crops_grid(ids, source="organizer_val"):
    """OCR all 15 variants of every id. Resumable; commits every 200 rows.

    Strips land in CROPS/GRID_SOURCE keyed `{image_id}__{variant}`, which is the
    shape embed_crops() already expects, so the GPU pass needs no new code.
    """
    from quorum.degrade import all_variants

    paths, all_ids = source_paths(source)
    by_id = dict(zip(all_ids, paths))
    d = CROPS / GRID_SOURCE
    d.mkdir(parents=True, exist_ok=True)
    meta = CROPS / f"{GRID_SOURCE}.parquet"
    done = set(pd.read_parquet(meta).image_id) if meta.exists() else set()

    todo = [i for i in ids if f"{i}{GRID_SEP}clean" not in done]
    print(f"grid: {len(done)} rows cached, {len(todo)} images x 15 to OCR",
          flush=True)

    def commit(rows):
        if not rows:
            return
        df = pd.DataFrame(rows)
        if meta.exists():
            df = pd.concat([pd.read_parquet(meta), df], ignore_index=True)
        df.drop_duplicates("image_id").to_parquet(meta, index=False)

    rows = []
    for n, iid in enumerate(todo, 1):
        src = by_id.get(iid)
        if src is None:
            print(f"  skip {iid}: not on disk", flush=True)
            continue
        try:
            img = Image.open(src).convert("RGB")
            variants = all_variants(img, iid)
        except Exception as e:
            print(f"  skip {iid}: {type(e).__name__}", flush=True)
            continue
        for vname, vimg in variants:
            key = f"{iid}{GRID_SEP}{vname}"
            try:
                strip, px = text_strip(vimg)
            except Exception as e:
                print(f"  skip {key}: {type(e).__name__}", flush=True)
                continue
            if strip is not None:
                strip.save(d / f"{key}.jpg", "JPEG", quality=95)
            rows.append({"image_id": key, "text_present": strip is not None,
                         "text_px": px})
        if n % 20 == 0:
            commit(rows); rows = []
            print(f"  {n}/{len(todo)} images", flush=True)
    commit(rows)
    print(f"  -> {meta}", flush=True)
    return meta


def evaluate_grid(train_source="sid_train"):
    """Per-variant detection rate and AUROC, probe fitted on CLEAN sid_train."""
    Xtr, rtr = load_crops(train_source)
    st = crop_px_stats(rtr)
    clf = LogisticRegression(max_iter=2000).fit(crop_design(Xtr, rtr, st),
                                                rtr.label.values.astype(int))

    meta = pd.read_parquet(CROPS / f"{GRID_SOURCE}.parquet")
    parts = meta.image_id.str.split(GRID_SEP, n=1, expand=True)
    meta["base"], meta["variant"] = parts[0], parts[1]
    lab = manifest_labels("organizer_val")
    meta["label"] = meta.base.map(lab)
    meta = meta[meta.label.notna()].reset_index(drop=True)

    X = np.load(CROPS / f"{GRID_SOURCE}_emb.npy")
    eid = pd.read_parquet(CROPS / f"{GRID_SOURCE}_emb.parquet").image_id.values
    pos = {i: k for k, i in enumerate(eid)}

    n_base = meta[meta.variant == "clean"].shape[0]
    print(f"{'variant':12s}{'detected':>10s}{'rate':>8s}{'AUROC':>9s}{'n_ai':>7s}{'n_real':>8s}")
    print("-" * 54)
    rows = []
    for v, g in meta.groupby("variant"):
        seen = g[g.text_present & g.image_id.isin(pos)]
        rate = len(seen) / max(n_base, 1)
        if seen.label.nunique() < 2:
            print(f"{v:12s}{len(seen):10d}{rate:8.1%}{'--':>9s}")
            continue
        idx = [pos[i] for i in seen.image_id]
        d = crop_design(X[idx], seen.assign(text_px=seen.text_px.values), st)
        a = roc_auc_score(seen.label.values.astype(int), clf.decision_function(d))
        rows.append((v, rate, a))
        print(f"{v:12s}{len(seen):10d}{rate:8.1%}{a:9.4f}"
              f"{int((seen.label == 1).sum()):7d}{int((seen.label == 0).sum()):8d}")

    if rows:
        cl = next((r for r in rows if r[0] == "clean"), None)
        worst = min(rows, key=lambda r: r[2])
        wd = min(rows, key=lambda r: r[1])
        print(f"\nclean AUROC {cl[2]:.4f}" if cl else "")
        print(f"worst AUROC {worst[2]:.4f} ({worst[0]}), "
              f"drop {cl[2] - worst[2]:.4f}" if cl else "")
        print(f"worst DETECTION {wd[1]:.1%} ({wd[0]}) -- "
              f"the branch is silent on {1 - wd[1]:.0%} of images there")
    return rows


if __name__ == "__main__":
    if "--save" in sys.argv:
        # ATTEMPT 2 ONLY. Attempt 1 (the 6 OCR statistics, MODEL above) is not
        # saved and must not be: it transfers at 0.4627, below chance, with five
        # of six features flipping sign across datasets. A displayed number that
        # is ANTI-correlated with the truth on unseen data is worse than no
        # number, which is the whole reason there are two attempts here.
        clf, st = save_crops()
        print(f"text crop probe -> {CROP_MODEL}  "
              f"({clf.coef_.size + 1} parameters, "
              f"{CROP_MODEL.stat().st_size / 1024:.1f} KB, "
              f"log2(text_px) mu={st[0]:.2f} sd={st[1]:.2f})")
        print("DISPLAY ONLY. predict.py does not load this; it is worth +0.0022")
        print("in the scorer and collapses to 0.5229 under the degradation grid.")
        raise SystemExit

    from PIL import Image, ImageDraw

    # --- the extractor, on images that need no dataset -----------------------
    blank = Image.new("RGB", (320, 160), (255, 255, 255))
    f, present = text_features(blank)
    assert not present and f.shape == (N_FEAT,) and f.dtype == np.float32, f
    assert not f.any(), "blank image must return neutral fill, not a score"

    lettered = Image.new("RGB", (320, 160), (255, 255, 255))
    ImageDraw.Draw(lettered).text((20, 60), "STOP AHEAD 100", fill=(0, 0, 0))
    g, present2 = text_features(lettered)
    assert np.isfinite(g).all(), g
    if present2:                       # tiny default font; detection is not certain
        assert 0.0 <= g[0] <= 1.0 and 0.0 <= g[2] <= 1.0 and 0.0 <= g[3] <= 1.0, g
        assert g[5] > 0, "text present but zero regions"
    print(f"text.py ok: blank -> {present}, lettered -> {present2} {g.round(3).tolist()}")
