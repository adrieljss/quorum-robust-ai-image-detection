"""Required deliverable: directory of images -> JSON of {image_path, pred}.

    python predict.py --input-dir path/to/images --output preds.json

`pred` is P(AI-generated) in [0,1], raw and uncalibrated. Two fields, no more --
the demo's richer schema stays in the demo.

ponytail: fusion.py exists and this still does not call it, deliberately.
Measured on so_fake_ood, clean / worst: raw max 0.9042 / 0.8634, fusion
0.8587 / 0.8340. Calibrating first makes it worse again (ECE 0.048 -> 0.103),
because sid_calib shares generators with train and every branch is saturated
there, so Platt fits an extreme slope that manufactures over-confidence on new
generators. Revisit once a generator-disjoint calibration source exists --
docs/HANDOVER-MODELS.md has the numbers and the fix.
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

    ponytail: max, not a learned combiner, and measurement says keep it that
    way for now -- a fusion LR over the calibrated branch vector scored lower on
    both so_fake_ood and tampered. Far better than the general probe alone,
    which scores tampered images BELOW real ones (AUC 0.37) because a
    locally-edited photo is globally authentic.
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
    # a typo'd path and a genuinely empty one are different problems; without
    # this they produce the same "no images under" line and look like the same one
    root = Path(a.input_dir)
    if not root.is_dir():
        raise SystemExit(f"not a directory: {a.input_dir}")
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in EXTS)
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
