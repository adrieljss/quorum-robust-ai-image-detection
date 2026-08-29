"""Official robustness transform grid from the problem statement.

14 settings + clean = 15 variants. Seeded per image so the eval grid is
reproducible run-to-run -- it is a required deliverable.
"""
import io

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

TRANSFORMS = {
    "jpeg":   [90, 70, 50, 30],    # 4  quality
    "blur":   [0.5, 1.0, 2.0],     # 3  gaussian sigma
    "resize": [0.5, 0.25],         # 2  downscale then back up
    "noise":  [0.02, 0.05, 0.10],  # 3  gaussian sigma on [0,1]
    "jitter": [0.20],              # 1  brightness/contrast/sat +-20%
    "crop":   [0.80],              # 1  center crop fraction
}

FLAT = [(k, p) for k, ps in TRANSFORMS.items() for p in ps]
N_SETTINGS = len(FLAT)                                  # 14
N_VARIANTS = N_SETTINGS + 1                             # 15, incl. clean
assert N_SETTINGS == 14, f"expected 14 settings, got {N_SETTINGS}"


def variant_name(kind: str, param) -> str:
    return f"{kind}{param}".replace(".", "")            # jpeg70, blur05, resize025


def seed_from_id(image_id: str) -> int:
    return int(image_id[:8], 16)


def rng_for(image_id: str, i: int):
    """RNG for setting `i` of one image. Seeded per variant, not per image, so
    variants can be generated in any order or in parallel and still reproduce."""
    return np.random.default_rng([seed_from_id(image_id), i])


def apply(img: Image.Image, kind: str, param, rng=None) -> Image.Image:
    rng = rng if rng is not None else np.random.default_rng(0)

    if kind == "jpeg":
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=int(param))
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    if kind == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=float(param)))

    if kind == "resize":
        w, h = img.size
        small = img.resize((max(1, int(w * param)), max(1, int(h * param))), Image.BICUBIC)
        return small.resize((w, h), Image.BICUBIC)

    if kind == "noise":                                  # SEEDED
        a = np.asarray(img, dtype=np.float32) / 255.0
        a = a + rng.normal(0, float(param), a.shape)
        return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))

    if kind == "jitter":                                 # SEEDED
        p = float(param)
        for Enh in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
            img = Enh(img).enhance(1.0 + rng.uniform(-p, p))
        return img

    if kind == "crop":
        w, h = img.size
        cw, ch = int(w * param), int(h * param)
        left, top = (w - cw) // 2, (h - ch) // 2
        return img.crop((left, top, left + cw, top + ch)).resize((w, h), Image.BICUBIC)

    raise ValueError(f"unknown transform: {kind}")


def all_variants(img: Image.Image, image_id: str):
    """Full grid: clean + all 14 settings = 15. For EVAL splits."""
    out = [("clean", img)] + [
        (variant_name(k, p), apply(img, k, p, rng_for(image_id, i)))
        for i, (k, p) in enumerate(FLAT)
    ]
    assert len(out) == N_VARIANTS
    return out


def variant_specs(image_id: str, k: int | None = None):
    """[(name, kind, param, rng)] without touching pixels -- lets a caller run the
    actual transforms in a thread pool. k=None means the full grid."""
    idx = range(N_SETTINGS) if k is None else sorted(
        np.random.default_rng(seed_from_id(image_id)).choice(
            N_SETTINGS, size=min(k, N_SETTINGS), replace=False))
    return [(variant_name(*FLAT[i]), *FLAT[i], rng_for(image_id, i)) for i in idx]


def sample_variants(img: Image.Image, image_id: str, k: int = 3):
    """Clean + k sampled settings. For the TRAIN split -- laptop GPU budget.

    Deterministic given image_id, so a rerun reproduces the same augmentation.
    """
    return [("clean", img)] + [
        (name, apply(img, kind, param, r)) for name, kind, param, r in variant_specs(image_id, k)
    ]


if __name__ == "__main__":
    rgb = np.random.default_rng(1).integers(0, 255, (128, 160, 3), dtype=np.uint8)
    img, iid = Image.fromarray(rgb), "a1b2c3d4e5f6a7b8"

    v = all_variants(img, iid)
    assert len(v) == N_VARIANTS == 15, len(v)
    assert [n for n, _ in v][:3] == ["clean", "jpeg90", "jpeg70"]
    assert len({n for n, _ in v}) == 15, "duplicate variant names"
    assert all(im.size == img.size for n, im in v if n != "crop08")

    s = sample_variants(img, iid)
    assert len(s) == 4 and s[0][0] == "clean"
    assert [n for n, _ in s] == [n for n, _ in sample_variants(img, iid)], "not deterministic"

    # seeded transforms must reproduce byte-for-byte, and actually change the image
    a = dict(all_variants(img, iid)); b = dict(all_variants(img, iid))
    for n in ("noise005", "jitter02"):
        assert np.array_equal(np.asarray(a[n]), np.asarray(b[n])), f"{n} unseeded"
        assert not np.array_equal(np.asarray(a[n]), rgb), f"{n} is a no-op"
    assert not np.array_equal(np.asarray(a["noise005"]), np.asarray(a["noise01"]))
    print("degrade.py ok:", N_SETTINGS, "settings,", N_VARIANTS, "variants")
