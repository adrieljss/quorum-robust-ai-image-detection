"""Score individual images through the face branch. For eyeballing, not evaluation.

    python scripts/try_face.py photo.jpg other.png
    python scripts/try_face.py photo.jpg --save-crops out/     # check the alignment

Prints the face probe alongside the general probe, because the interesting cases
are the ones where they disagree. Both are P(AI-generated).

Face alignment has silently broken before in this project (a landmark swap that
mirrored every crop), and a mirrored face still produces a confident-looking
number. `--save-crops` is how you check: the eyes should land in the same place
in every crop, level and upright.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import numpy as np
from PIL import Image

from quorum.detectors.general import MODELS
from quorum.embed import Embedder
from quorum.features import face_crop

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _probe(path):
    z = np.load(path)
    return z["w"].ravel(), float(z["b"].ravel()[0])


def _platt(path, key):
    """(a, b) for the branch's calibrator, or None if fusion has not been fitted."""
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=True)
    return tuple(z[key]) if key in z.files else None


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main(a):
    paths = [Path(p) for p in a.images]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise SystemExit(f"not a file: {missing[0]}")

    zf = np.load(MODELS / "face.npz")
    wf, bf = zf["w"].ravel(), float(zf["b"].ravel()[0])
    mu, sd = float(zf["px_mu"]), float(zf["px_sd"])
    wg, bg = _probe(MODELS / "general.npz")
    cal = _platt(MODELS / "fusion.npz", "cal_face")

    emb = Embedder()
    if a.save_crops:
        Path(a.save_crops).mkdir(parents=True, exist_ok=True)

    # Two batched passes, not two per image. The model load is ~13s either way,
    # but batch-of-one wastes the GPU: 0.02s/image inside a batch of 32 against
    # ~0.05s alone. Detect first so the crops travel in one batch with the rest.
    imgs, crops, found = [], [], []
    for p in paths:
        img = Image.open(p)
        crop, px = face_crop(img)
        imgs.append(img.convert("RGB"))
        if crop is not None:
            found.append((len(crops), px))
            crops.append(crop)
            if a.save_crops:
                crop.save(Path(a.save_crops) / f"{p.stem}_face.png")
        else:
            found.append((None, 0.0))

    Vg = emb.embed_batch(imgs)
    Vf = emb.embed_batch(crops) if crops else np.empty((0, 768), np.float32)

    print(f"{'image':28}{'face':>8}{'px':>7}{'P(AI) face':>12}{'P(AI) general':>15}")
    for p, vg, (j, px) in zip(paths, Vg, found):
        pg = sigmoid(wg @ vg + bg)
        if j is None:
            # No face row is written at inference either -- fusion sees
            # face_present=0 and a neutral fill, never "the face model says real".
            print(f"{p.name[:27]:28}{'none':>8}{'-':>7}{'abstain':>12}{pg:>15.4f}")
            continue
        x = np.append(Vf[j], (np.log2(px) - mu) / sd)   # the 769th feature
        s = wf @ x + bf
        pf = sigmoid(cal[0] * s + cal[1]) if cal else sigmoid(s)
        print(f"{p.name[:27]:28}{'yes':>8}{px:>7.0f}{pf:>12.4f}{pg:>15.4f}")

    if cal is None:
        print("\nnote: no cal_face in fusion.npz -- face scores are UNCALIBRATED")
    if a.save_crops:
        print(f"crops -> {a.save_crops}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--save-crops", metavar="DIR")
    main(ap.parse_args())
