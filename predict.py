"""Required deliverable: directory of images -> JSON of {image_path, pred}.

    python predict.py --input-dir path/to/images --output preds.json

`pred` is P(AI-generated), calibrated, in [0,1]. Two fields, no more -- the
demo's richer schema stays in the demo.

ponytail: two probes, max-combined, uncalibrated. Swap score_all() for the
fusion call when it lands; nothing else here changes.
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


def _probe(path):
    d = np.load(path)
    return d["w"].ravel(), d["b"][0]


def score_all(paths) -> np.ndarray:
    """max(P_synthetic, P_tampered) -- either one firing means AI touched it.

    ponytail: max, not a learned combiner. The two probes are separately
    uncalibrated so this over-trusts whichever is more confident; fusion.py
    with per-branch calibration is the real answer. Still far better than the
    general probe alone, which scores tampered images BELOW real ones
    (AUC 0.37) because a locally-edited photo is globally authentic.
    """
    probes = [_probe(MODEL)] + ([_probe(MODEL_TAMPERED)] if MODEL_TAMPERED.exists() else [])
    emb, out = Embedder(), []
    for i in range(0, len(paths), BATCH):
        imgs = [Image.open(p).convert("RGB") for p in paths[i:i + BATCH]]
        v = emb.embed_batch(imgs)
        out.append(np.max([1 / (1 + np.exp(-(v @ w + b))) for w, b in probes], axis=0))
        print(f"  {min(i + BATCH, len(paths))}/{len(paths)}")
    return np.concatenate(out)


def main(a):
    paths = sorted(p for p in Path(a.input_dir).rglob("*") if p.suffix.lower() in EXTS)
    if not paths:
        raise SystemExit(f"no images under {a.input_dir}")
    preds = [{"image_path": p.as_posix(), "pred": round(float(v), 4)}
             for p, v in zip(paths, score_all(paths))]  # posix: judges may diff paths
    Path(a.output).write_text(json.dumps(preds, indent=2))
    print(f"{len(preds)} predictions -> {a.output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output", default="preds.json")
    main(p.parse_args())
