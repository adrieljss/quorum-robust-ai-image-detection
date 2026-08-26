"""Frozen CLIP embedding, fp16, versioned on-disk cache.

Also owns the cache format: normalisation, image ids, shard writing and the
shard-aware reader everyone else uses. One module owns the cache so a
preprocessing change is one bump of CACHE_VERSION.
"""
import glob
import hashlib
import io
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# -quickgelu is REQUIRED with the openai weights: they were trained with
# QuickGELU, and plain "ViT-L-14" silently loads them into a standard-GELU model.
# Measured cosine between the two on identical images: 0.88-0.91. No error, no
# crash, just wrong embeddings and every downstream number inheriting them.
BACKBONE = "ViT-L-14-quickgelu"
PRETRAINED = "openai"
CACHE_VERSION = "vitl14_v1"      # bump if preprocessing changes
DIM = 768
MAX_BATCH = 32                   # ponytail: 8GB VRAM ceiling; raise on a bigger GPU
# The pass is CPU-bound, not GPU-bound: measured 84% of per-image time in PIL
# (degrade + JPEG + resize), 17% in the ViT forward. PIL releases the GIL, so
# threads are enough -- no multiprocessing, no shared-memory dance.
WORKERS = int(os.environ.get("QUORUM_WORKERS", min(8, os.cpu_count() or 4)))
POOL = ThreadPoolExecutor(max_workers=WORKERS)

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache" / "embeddings" / CACHE_VERSION
MANIFESTS = ROOT / "data" / "manifests"


def normalise(img: Image.Image) -> Image.Image:
    """JPEG q95 round-trip. MUST run before hashing, so file format cannot
    leak the label and streamed ids match downloaded ones."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=95)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def image_id(img: Image.Image) -> str:
    return hashlib.sha256(img.tobytes()).hexdigest()[:16]


class Embedder:
    def __init__(self, device=None, fp16=True):
        import open_clip
        import torch
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.fp16 = fp16 and self.device == "cuda"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            BACKBONE, pretrained=PRETRAINED)
        self.model = self.model.to(self.device).eval()
        if self.fp16:
            self.model = self.model.half()               # 1.22GB -> 0.61GB
        for p in self.model.parameters():
            p.requires_grad = False                      # never unfreeze

    def embed_batch(self, pil_images) -> np.ndarray:
        out = []
        with self.torch.inference_mode():
            for i in range(0, len(pil_images), MAX_BATCH):
                chunk = pil_images[i:i + MAX_BATCH]
                x = self.torch.stack(list(POOL.map(self.preprocess, chunk))).to(self.device)
                if self.fp16:
                    x = x.half()
                v = self.model.encode_image(x)
                v = v / v.norm(dim=-1, keepdim=True)     # L2 normalise -- do NOT skip
                out.append(v.float().cpu().numpy().astype(np.float32))
        return np.concatenate(out)


class ShardWriter:
    """Flush every `every` embeddings so peak RAM stays bounded.

    Writes {source}_{NNN}.npy alongside rows_{source}_{NNN}.csv; row i of the
    csv describes row i of the array. load_source() reassembles them.
    """

    def __init__(self, source: str, every: int = 20_000):
        self.source, self.every = source, every
        self.vecs, self.rows = [], []
        CACHE.mkdir(parents=True, exist_ok=True)
        MANIFESTS.mkdir(parents=True, exist_ok=True)
        # Continue numbering past whatever this source already wrote, so a rerun
        # appends instead of overwriting _000. build_manifest dedupes on
        # (image_id, variant), so an overlapping rerun is harmless.
        self.shard = len(glob.glob(str(CACHE / f"{source}_*.npy")))

    def add(self, vec, row):
        self.vecs.append(vec)
        self.rows.append(row)
        if len(self.vecs) >= self.every:
            self.flush()

    def flush(self):
        if not self.vecs:
            return
        tag = f"{self.source}_{self.shard:03d}"
        np.save(CACHE / f"{tag}.npy", np.stack(self.vecs))
        pd.DataFrame(self.rows).to_csv(MANIFESTS / f"rows_{tag}.csv", index=False)
        print(f"  flushed {tag}: {len(self.vecs)} embeddings")
        self.vecs, self.rows = [], []
        self.shard += 1


def load_source(source: str):
    """(X, rows) for one source, shards concatenated. X[i] <-> rows.iloc[i]."""
    Xs, Rs = [], []
    for s in sorted(glob.glob(str(CACHE / f"{source}_*.npy"))):
        tag = Path(s).stem                               # NOT split("/") -- Windows
        Xs.append(np.load(s))
        Rs.append(pd.read_csv(MANIFESTS / f"rows_{tag}.csv"))
    if not Xs:
        raise FileNotFoundError(f"no shards for source {source!r} in {CACHE}")
    X, R = np.concatenate(Xs), pd.concat(Rs, ignore_index=True)
    assert len(X) == len(R), f"{source}: {len(X)} vecs vs {len(R)} rows"
    return X, R


def embed_variants(emb, writer, img, row, full_grid: bool, k: int = 3) -> str:
    """Normalise -> id -> embed every variant -> shard. Returns the image_id.

    One pass: this image is in memory once and never again, so all variants it
    will ever need are generated now.
    """
    from quorum.degrade import apply, variant_specs

    img = normalise(img)
    iid = image_id(img)
    specs = variant_specs(iid, None if full_grid else k)
    degraded = POOL.map(lambda sp: apply(img, sp[1], sp[2], sp[3]), specs)
    variants = [("clean", img)] + list(zip((sp[0] for sp in specs), degraded))
    for (name, _), vec in zip(variants, emb.embed_batch([v for _, v in variants])):
        writer.add(vec, {**row, "image_id": iid, "variant": name})
    return iid


if __name__ == "__main__":
    import sys
    if "--clip" in sys.argv:                 # needs torch + a ~1.7GB weight download
        e = Embedder()
        v = e.embed_batch([Image.new("RGB", (512, 512), c) for c in ("red", "blue")])
        assert v.shape == (2, DIM), v.shape
        assert abs(float((v[0] ** 2).sum()) - 1) < 1e-3, "not L2-normalised"
        assert not np.allclose(v[0], v[1]), "identical embeddings for different images"
        print(f"CLIP ok: {v.shape} {v.dtype} device={e.device} fp16={e.fp16}")
        if e.device != "cuda":
            print("WARNING: running on CPU -- the embedding pass will take ~20h, not ~2h")
        raise SystemExit

    rgb = np.random.default_rng(2).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    a, b = Image.fromarray(rgb), Image.fromarray(rgb).convert("RGB")
    assert image_id(normalise(a)) == image_id(normalise(b)), "id not format-stable"
    assert len(image_id(normalise(a))) == 16

    w = ShardWriter("selfcheck", every=3)
    for i in range(7):
        w.add(np.full(DIM, i, np.float32), {"image_id": f"{i:016x}", "variant": "clean"})
    w.flush()
    X, R = load_source("selfcheck")
    assert X.shape == (7, DIM) and len(R) == 7, X.shape
    assert (X[:, 0] == np.arange(7)).all(), "shard order lost"
    for f in glob.glob(str(CACHE / "selfcheck_*.npy")) + glob.glob(str(MANIFESTS / "rows_selfcheck_*.csv")):
        Path(f).unlink()
    print("embed.py cache ok (Embedder needs torch; run the Step 4 sanity check)")
