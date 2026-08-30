"""Run ONE image through the robustness grid. For eyeballing, not evaluation.

    python scripts/try_grid.py test-images/image.png
    python scripts/try_grid.py photo.jpg --save-dir out/    # write the variants out
    python scripts/try_grid.py photo.jpg --chain            # all 14x14 COMPOSED pairs

The SHIPPED column comes from predict.score_embeddings, so it is by construction
the number predict.py emits. The general/tampered/face columns are
fusion-Platt-calibrated diagnostics on a DIFFERENT scale -- read them for shape,
never compare them to the shipped column or to each other. This is the robustness claim on a single image instead of
averaged over 4,198 of them, which is the version you can actually look at.

--chain exists because the official grid, our cache, and therefore every number
in docs/robustness.md are SINGLE transforms. Real images arrive composed: upload
resizes, the platform recompresses, someone screenshots the result. Nothing in
the eval set measures that, so this is the only place it is measured at all.

Every column is P(AI-generated). The image is
normalised (JPEG q95) and seeded off its own image_id first, exactly as embed.py
does, so these numbers are comparable to docs/robustness.md.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import numpy as np
from PIL import Image

import predict
from quorum.degrade import FLAT, all_variants, apply, rng_for, variant_name
from quorum.detectors.general import MODELS
from quorum.embed import Embedder, image_id, normalise
from quorum.features import face_crop


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def load_models():
    cal = np.load(MODELS / "fusion.npz", allow_pickle=True)
    zf = np.load(MODELS / "face.npz")
    probes = {}
    for n in ("general", "tampered"):
        z = np.load(MODELS / f"{n}.npz")
        probes[n] = (z["w"].ravel(), float(z["b"].ravel()[0]))
    return {"cal": cal, "probes": probes,
            "face": (zf["w"].ravel(), float(zf["b"].ravel()[0]),
                     float(zf["px_mu"]), float(zf["px_sd"]))}


def score(imgs, emb, M):
    """-> list of dicts, one per image. Face crops ride in ONE batch, not one
    batch per image: the model load dominates, but batch-of-one wastes the GPU."""
    crops, found = [], []
    for im in imgs:
        crop, px = face_crop(im)
        if crop is None:
            found.append((None, 0.0))
        else:
            found.append((len(crops), px))
            crops.append(crop)

    V = emb.embed_batch(imgs)
    # THE shipped score, from the one function that defines it. This file used to
    # re-derive it as max() over fusion-Platt-calibrated branches, which is a
    # DIFFERENT model: max() is not invariant to rescaling one argument, so it
    # disagreed with predict.py on the verdict, not just the scale. On
    # test-images/image.png it read 0.7321 AI-GENERATED against predict.py's
    # 0.4916 real. Third time a script in here has grown a private copy that
    # drifted; score_embeddings' own docstring warns about exactly this.
    shipped = predict.score_embeddings(V)
    Vf = emb.embed_batch(crops) if crops else np.empty((0, 768), np.float32)
    wf, bf, mu, sd = M["face"]
    cal = M["cal"]

    out = []
    for v, sh, (j, px) in zip(V, shipped, found):
        r = {"shipped": float(sh)}
        for n, (w, b) in M["probes"].items():
            ca, cb = cal[f"cal_{n}"]
            r[n] = float(sigmoid(ca * (w @ v + b) + cb))
        if j is None:
            r["face"], r["px"] = None, 0.0
        else:
            s = wf @ np.append(Vf[j], (np.log2(px) - mu) / sd) + bf
            ca, cb = cal["cal_face"]
            r["face"], r["px"] = float(sigmoid(ca * s + cb)), px
        out.append(r)
    return out


def single(img, iid, emb, M, save_dir):
    variants = all_variants(img, iid)
    if save_dir:
        d = Path(save_dir)
        d.mkdir(parents=True, exist_ok=True)
        for name, v in variants:
            v.save(d / f"{iid}_{name}.png")

    rows = score([v for _, v in variants], emb, M)
    print(f"{'variant':12}{'general':>9}{'tampered':>10}{'SHIPPED':>9}{'face':>9}{'px':>7}")
    print("  " + "-" * 54)
    for (name, _), r in zip(variants, rows):
        fs = "abstain" if r["face"] is None else f"{r['face']:.4f}"
        pxs = "-" if r["face"] is None else f"{r['px']:.0f}"
        tail = "  <- clean" if name == "clean" else ""
        print(f"{name:12}{r['general']:>9.4f}{r['tampered']:>10.4f}"
              f"{r['shipped']:>9.4f}{fs:>9}{pxs:>7}{tail}")

    ship = [r["shipped"] for r in rows]
    w = int(np.argmin(ship))
    print(f"\nshipped max(general,tampered):  clean {ship[0]:.4f}   "
          f"worst {ship[w]:.4f} ({variants[w][0]})   spread {max(ship) - min(ship):.4f}")
    faces = sum(r["face"] is not None for r in rows)
    print(f"face coverage: {faces}/{len(rows)} variants")
    print(f"\nverdict on clean: "
          f"{'AI-GENERATED' if ship[0] >= 0.5 else 'REAL'} (P(AI) = {ship[0]:.4f})")
    if save_dir:
        print(f"variants -> {save_dir}")


def chained(img, iid, emb, M):
    """Every ordered pair second(first(img)), 14x14 including the diagonal.

    Ordered and diagonal-inclusive on purpose: jpeg-then-blur is not
    blur-then-jpeg, and jpeg-twice is the single most common thing that happens
    to an image on the internet. Stage 2 draws from a different seed stream so
    noise-on-noise does not re-add the identical field.

    Built and embedded 14 at a time -- holding 196 full-res PIL images is ~900MB.
    """
    names = [variant_name(k, p) for k, p in FLAT]
    clean_r = score([img], emb, M)[0]
    grid, cover = {}, {}

    for i, (k1, p1) in enumerate(FLAT):
        stage1 = apply(img, k1, p1, rng_for(iid, i))
        batch = [apply(stage1, k2, p2, rng_for(iid, 100 + 14 * i + j))
                 for j, (k2, p2) in enumerate(FLAT)]
        for nm2, r in zip(names, score(batch, emb, M)):
            grid[(names[i], nm2)] = r["shipped"]
            cover[(names[i], nm2)] = r["face"] is not None
        print(f"  {names[i]:10} done", file=sys.stderr)

    print("\nSHIPPED score x100, row = first transform, col = second\n")
    print(f"{'':10}" + "".join(f"{n[:4]:>5}" for n in names))
    for n1 in names:
        cells = "".join(f"{100 * grid[(n1, n2)]:5.0f}" for n2 in names)
        print(f"{n1:10}{cells}")

    vals = np.array(list(grid.values()))
    singles = score([v for _, v in all_variants(img, iid)][1:], emb, M)
    s_ship = [r["shipped"] for r in singles]

    print(f"\nclean                {clean_r['shipped']:.4f}")
    print(f"worst single (14)    {min(s_ship):.4f}   "
          f"[{names[int(np.argmin(s_ship))]}]")
    print(f"worst chained (196)  {vals.min():.4f}   "
          f"[{' -> '.join(min(grid, key=grid.get))}]")
    print(f"mean chained         {vals.mean():.4f}")
    print(f"chains below 0.5     {int((vals < 0.5).sum())}/196   "
          f"(a flipped verdict)")

    order = sorted(grid, key=grid.get)[:8]
    print("\nworst 8 chains:")
    for key in order:
        print(f"  {key[0]:10} -> {key[1]:10} {grid[key]:.4f}"
              f"{'' if cover[key] else '   (face lost)'}")
    lost = sum(not v for v in cover.values())
    print(f"\nface coverage: {196 - lost}/196 chains"
          f"{'' if not lost else '  -- the face branch abstains on the rest'}")


def main(a):
    src = Path(a.image)
    if not src.is_file():
        raise SystemExit(f"not a file: {src}")
    img = normalise(Image.open(src))       # q95 first -- id and scores both depend on it
    iid = image_id(img)
    print(f"{src.name}  {img.size[0]}x{img.size[1]}  image_id={iid}\n")
    M, emb = load_models(), Embedder()
    (chained(img, iid, emb, M) if a.chain
     else single(img, iid, emb, M, a.save_dir))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--save-dir", metavar="DIR")
    ap.add_argument("--chain", action="store_true",
                    help="all 14x14 composed pairs instead of the 15 singles")
    main(ap.parse_args())
