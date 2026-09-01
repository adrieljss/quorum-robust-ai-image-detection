"""Pixel-branch features: aligned face crops and spectral stats.

Computed inline during the streaming pass -- each image is in memory once and
never again, same one-pass contract as embed_variants(). CLIP threw away the
high-frequency detail these branches exist to recover (PIPELINE 7.2), so they
must read pixels, not embeddings.
"""
import numpy as np
from PIL import Image

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
YUNET = ROOT / "data" / "models" / "yunet.onnx"
MIN_FACE = 64          # a 40px face upscaled to 224 teaches "blurry => fake"

# ArcFace 5-point template at 112, x2 for a 224 crop.
CANONICAL_5 = np.array([
    [38.29, 51.69], [73.53, 51.50], [56.02, 71.74],
    [41.55, 92.37], [70.73, 92.20]], dtype=np.float32) * 2.0

DET_MAX = 512    # detect small, warp from native -- see face_crop
_local = __import__("threading").local()


def _umeyama(src, dst):
    """Least-squares similarity transform (rotation + uniform scale + shift)."""
    sm, dm = src.mean(0), dst.mean(0)
    sc, dc = src - sm, dst - dm
    U, S, Vt = np.linalg.svd(dc.T @ sc / len(src))
    D = np.diag([1.0, -1.0 if np.linalg.det(U @ Vt) < 0 else 1.0])
    R = U @ D @ Vt
    scale = float((S * np.diag(D)).sum() / sc.var(0).sum())
    return np.hstack([scale * R, (dm - scale * R @ sm)[:, None]]).astype(np.float32)


def _detector(w, h):
    """One detector per thread -- FaceDetectorYN holds mutable input-size state,
    so a shared instance races under the embed pool."""
    import cv2
    d = getattr(_local, "det", None)
    if d is None:
        d = _local.det = cv2.FaceDetectorYN.create(str(YUNET), "", (w, h))
    d.setInputSize((w, h))
    return d


def _detect(img: Image.Image):
    """(bgr native, faces in native coords) or (bgr native, None).

    Shared detect step for face_crop() and face_crop_all() -- detection does
    not need native resolution, only the crop does, so this detects on a
    shrunk copy and scales landmarks back rather than resizing the image
    twice.
    """
    import cv2
    bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    k = min(1.0, DET_MAX / max(h, w))
    small = cv2.resize(bgr, (int(w * k), int(h * k))) if k < 1.0 else bgr
    _, faces = _detector(small.shape[1], small.shape[0]).detect(small)
    if faces is None or not len(faces):
        return bgr, None
    return bgr, faces / k     # back to native coords


def _warp(bgr, f):
    """One YuNet detection row (native coords) -> (aligned 224x224 PIL, box_px).

    Alignment is the whole point: landmarks warp every face into one coordinate
    frame, so 'is this eye consistent with that eye' becomes a fixed-position
    comparison instead of a vision problem.
    """
    import cv2
    # YuNet names its landmarks from the subject's point of view ("right eye"),
    # but emits them image-left first -- verified 13/13 on COCO. That already
    # matches the template, so do NOT swap rows (PIPELINE 5.2 says otherwise
    # and is wrong for cv2 5.x; swapping mirrors every face).
    lm = f[4:14].reshape(5, 2).astype(np.float32)
    M = _umeyama(lm, CANONICAL_5)   # LMEDS/RANSAC fit a subset of 5 points; this
                                    # is least squares over all of them

    out = cv2.warpAffine(bgr, M, (224, 224), flags=cv2.INTER_LINEAR)
    # Box size travels with the crop: a 64px face upscaled to 224 carries far
    # harsher effective degradation than a 181px one downscaled to it (measured
    # 2.8x spread on COCO). Without this the probe cannot tell them apart.
    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)), float(min(f[2], f[3]))


def face_crop(img: Image.Image):
    """(aligned 224x224 PIL, box_px) or (None, 0.0) -- the single LARGEST face.

    This is the function the training pipeline (extract_variants below) and
    face.npz's cache are built on: one row per image, matching how the probe
    was fit. Do not change its selection rule -- see face_crop_all() for the
    multi-face variant used elsewhere.
    """
    bgr, faces = _detect(img)
    if faces is None:
        return None, 0.0
    f = max(faces, key=lambda r: r[2] * r[3])
    if min(f[2], f[3]) < MIN_FACE:
        return None, 0.0
    return _warp(bgr, f)


MAX_FACES = 8    # a cap on cost AND on the multiplicity bias: the shipped
                 # scorer takes a max over these, and max over N draws rises
                 # with N even when every draw is from the same distribution,
                 # so uncapped a stadium crowd out-scores a portrait for no
                 # reason but its size.


def face_crop_all(img: Image.Image, max_faces: int = MAX_FACES):
    """[(aligned 224x224 PIL, box_px, bbox_native_px), ...], largest first.

    Same detector and alignment as face_crop(), but keeps every face at or
    above MIN_FACE instead of only the largest. face.npz is still a
    single-face-per-crop probe (see face_crop()'s docstring) -- this just
    lets a caller run that same probe over every face in an image and combine
    the results itself (e.g. max() across faces), rather than silently
    dropping every face but the largest. Not used by the training pipeline or
    its cache, so it cannot affect anything face.npz was fit on.

    bbox is {"x", "y", "width", "height"} in native pixel coords, the same
    shape as every other region box.
    """
    bgr, faces = _detect(img)
    if faces is None:
        return []
    keep = [f for f in faces if min(f[2], f[3]) >= MIN_FACE]
    keep.sort(key=lambda r: r[2] * r[3], reverse=True)
    out = []
    for f in keep[:max_faces]:
        crop, px = _warp(bgr, f)
        x, y, bw, bh = f[:4]
        out.append((crop, px, {
            "x": int(round(float(x))), "y": int(round(float(y))),
            "width": int(round(float(bw))), "height": int(round(float(bh))),
        }))
    return out


FFT_N = 512      # fixed window, NATIVE resolution -- see spectral_features
_BANDS = None


def _band_index(n=FFT_N):
    """Radial band id per pixel, built once. Rebuilding the mask per image cost
    more than the FFT did."""
    global _BANDS
    if _BANDS is None:
        c = n // 2
        yy, xx = np.ogrid[:n, :n]
        r = np.hypot(yy - c, xx - c) / c
        b = np.clip((r * 5).astype(np.int32), 0, 4).ravel()
        _BANDS = (b, np.bincount(b, minlength=5), (r > 0.25).ravel())
    return _BANDS


def spectral_features(img: Image.Image) -> np.ndarray:
    """float32[8]: 5 radial band energies + peak/median + grid peak + rolloff.

    Upsampling and transposed convolution leave periodic traces that a
    high-pass residual exposes. Dies under heavy JPEG -- that is expected, and
    why fusion must also see degradation_estimate (PIPELINE 7.1).

    Measured on a CENTRE CROP at native resolution, never a resize: downscaling
    destroys exactly the high frequencies this branch exists to read. The fixed
    window also makes cost independent of image size (1024px SID_Set images
    cost the same as 640px COCO ones).
    """
    import cv2
    g = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = g.shape
    if h < FFT_N or w < FFT_N:                       # small/cropped variants
        g = cv2.copyMakeBorder(g, 0, max(0, FFT_N - h), 0, max(0, FFT_N - w),
                               cv2.BORDER_REFLECT)
        h, w = g.shape
    y0, x0 = (h - FFT_N) // 2, (w - FFT_N) // 2
    g = g[y0:y0 + FFT_N, x0:x0 + FFT_N]

    resid = g - cv2.medianBlur(g, 3)
    F = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(resid))))

    idx, counts, outer = _band_index()
    flat = F.ravel()
    bands = (np.bincount(idx, weights=flat, minlength=5) / counts).tolist()

    med = float(np.median(flat)) or 1e-6
    peak = float(flat[outer].max() / med)
    c = FFT_N // 2
    grid = float(F[c::c // 2, c::c // 2].mean() / med)   # upsampling lattice
    slope = float(np.polyfit(np.arange(5), bands, 1)[0])
    return np.array(bands + [peak, grid, slope], dtype=np.float32)


def extract_variants(emb, face_w, spec_w, img, row, full_grid: bool, k: int = 3):
    """Face embedding + spectral vector for every variant of one image.

    Face rows are written only when a face is present -- fusion left-joins and
    fills the rest, so 'no face here' can never read as 'the face model says real'.
    """
    from quorum.degrade import apply, variant_specs
    from quorum.embed import POOL, image_id, normalise

    img = normalise(img)
    iid = image_id(img)
    specs = variant_specs(iid, None if full_grid else k)

    def one(sp):
        """degrade -> spectral -> face, all CPU and all GIL-releasing (cv2/numpy)."""
        name, v = ("clean", img) if sp is None else (sp[0], apply(img, sp[1], sp[2], sp[3]))
        return name, spectral_features(v), face_crop(v)

    crops, names = [], []
    for name, spec, (c, px) in POOL.map(one, [None] + list(specs)):   # map keeps order
        spec_w.add(spec, {**row, "image_id": iid, "variant": name})
        if c is not None:
            crops.append(c)
            names.append((name, px))
    if crops:
        for (name, px), vec in zip(names, emb.embed_batch(crops)):
            face_w.add(vec, {**row, "image_id": iid, "variant": name,
                             "face_present": 1, "face_px": px})
    return iid


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    noise = Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8))
    flat = Image.new("RGB", (256, 256), (128, 128, 128))

    a, b = spectral_features(noise), spectral_features(flat)
    assert a.shape == (8,) and a.dtype == np.float32, a.shape
    assert np.isfinite(a).all() and np.isfinite(b).all(), "non-finite features"
    assert a[:5].mean() > b[:5].mean(), "noise must carry more residual energy than flat grey"

    c, px = face_crop(noise)
    assert c is None and px == 0.0, "detected a face in pure noise"
    assert face_crop_all(noise) == [], "face_crop_all disagrees with face_crop on noise"

    # face_crop() must stay exactly "the largest face", because face.npz was fit
    # on it and every cached face_* row came from it. These are the assertions
    # that catch a multi-face refactor going wrong -- two of them have already
    # done so.
    real = ROOT / "test-images" / "real" / "real1.png"
    if real.exists():
        img = Image.open(real).convert("RGB")
        fs = face_crop_all(img)
        one = face_crop(img)
        if fs:
            assert one[1] == fs[0][1], f"face_crop {one[1]} != largest of {[f[1] for f in fs]}"
            sides = [f[1] for f in fs]
            assert sides == sorted(sides, reverse=True), "face_crop_all must be largest-first"
            assert all(px >= MIN_FACE for _, px, _ in fs), "a face under the floor got through"
            assert len(fs) <= MAX_FACES, f"{len(fs)} faces, cap is {MAX_FACES}"
            for _, _, b in fs:
                assert set(b) == {"x", "y", "width", "height"}, b
                assert b["width"] > 0 and b["height"] > 0, b
        else:
            assert one[0] is None, "face_crop found a face face_crop_all did not"
        print(f"  face_crop == largest of face_crop_all ({len(fs)} face(s) in real1.png)")
    print(f"features.py ok: spectral {a.round(3).tolist()}, no false face")
