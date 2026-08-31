"""Quorum vs published detectors -- CNNDetection (CVPR'20) and FatFormer (CVPR'24).

Same pixels into all three: every image is normalised (JPEG q95) and degraded by
quorum.degrade exactly as docs/robustness.md does, then handed to each detector
with that detector's own published preprocessing.

    python scripts/compare_baselines.py --out docs/BASELINES.md

Weights are not in the repo (1.2GB). Point --weights-dir at a directory holding
`blur_jpg_prob0.5.pth` (CNNDetection) and `fatformer_4class.pth`, and
--fatformer-dir at a clone of github.com/Michel-liu/FatFormer with OpenAI's
`pretrained/ViT-L-14.pt` inside it. A detector whose weights are missing is
skipped, not faked.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from predict import EXTS, face_score, load_face, load_probes, score_embeddings  # noqa: E402
from quorum.degrade import FLAT, apply, rng_for, variant_name  # noqa: E402
from quorum.embed import Embedder, image_id, normalise  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

# One per transform family, the harshest setting of each. The full 14 are in
# docs/robustness.md; five keeps a three-model sweep inside a laptop GPU hour.
SETTINGS = ["clean", "jpeg30", "blur20", "resize025", "noise005"]

# (name, real dirs, {fake label: dirs}). Reals are shared across the fake
# columns of a population, so a per-generator AUROC is against the same reals.
POPULATIONS = {
    "organizer_val": (
        [RAW / "organizer_val" / "coco_val2017"],
        {"wildfake_dalle_adv": [RAW / "organizer_val" / "wildfake_dalle_adv"]},
    ),
    "unseen_generators": (
        [RAW / "real_holdout_laion"],
        {g: [RAW / f"probe_{g}"] for g in
         ("adm", "dfgan", "gigagan", "vqvae_ffhq", "gptimage2", "nanobanana")},
    ),
}


def sample(dirs, n):
    """Deterministic stride over the sorted listing -- no seed to drift."""
    files = sorted(p for d in dirs for p in Path(d).rglob("*") if p.suffix.lower() in EXTS)
    if n and len(files) > n:
        files = files[:: max(1, len(files) // n)][:n]
    return files


def variants(img, wanted):
    """{setting: image}, seeded off the image id exactly as the eval grid is."""
    out = {"clean": img} if "clean" in wanted else {}
    for i, (kind, param) in enumerate(FLAT):
        nm = variant_name(kind, param)
        if nm in wanted:
            out[nm] = apply(img, kind, param, rng_for(image_id(img), i))
    return out


# --------------------------------------------------------------------------
# detectors: PIL list -> P(AI-generated), one float per image
# --------------------------------------------------------------------------

def quorum_scorer():
    emb, probes, face = Embedder(), load_probes(), load_face()

    def score(imgs):
        V = emb.embed_batch(imgs)
        return score_embeddings(V, probes, face_score(imgs, emb, face))
    return score


def _torch_batch(model, tf, imgs, device, softmax_idx=None):
    import torch
    with torch.inference_mode():
        x = torch.stack([tf(im) for im in imgs]).to(device)
        out = model(x)
        if softmax_idx is not None:
            return out.softmax(dim=1)[:, softmax_idx].float().cpu().numpy()
        return out.sigmoid().flatten().float().cpu().numpy()


def cnndetection_scorer(weights, resize):
    """Wang et al. 2020, ResNet-50 on ProGAN, blur+JPEG augmented (prob0.5).

    Two preprocessings because the paper's own protocol (crop at native
    resolution) and the obvious one (resize then crop) disagree by a lot on
    images that are not 256px: this detector reads high-frequency statistics
    that resampling destroys. Both are reported; neither is cherry-picked.
    """
    import torch
    from torchvision import transforms
    from torchvision.models import resnet50

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = resnet50(num_classes=1)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=False)["model"])
    model.eval().to(device)

    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    steps = [transforms.Resize((256, 256))] if resize else []
    # pad first: CenterCrop on a native-resolution image smaller than 224 would
    # otherwise zero-pad implicitly and the batch would still stack, silently
    steps += [transforms.CenterCrop(224), transforms.ToTensor(), norm]
    tf = transforms.Compose(steps)

    def score(imgs):
        return _torch_batch(model, tf, [im.convert("RGB") for im in imgs], device)
    return score


def fatformer_scorer(weights, repo):
    """Liu et al. 2024, CLIP ViT-L/14 + forgery-aware adapters, 4-class ProGAN.

    Their clip.py resolves the backbone by the RELATIVE path pretrained/ViT-L-14.pt,
    so the build has to happen with the repo as cwd.
    """
    import torch
    from torchvision import transforms

    sys.path.insert(0, str(repo))
    cwd = Path.cwd()
    try:
        os.chdir(repo)
        from main import get_args_parser          # their defaults, not ours
        from models import build_model
        # --num_vit_adapter 3 is the README's eval command, NOT the parser
        # default of 8: the released checkpoint puts adapters on resblocks
        # 7/15/23, and 8 builds them on 2/5/8/... and refuses to load.
        args = get_args_parser().parse_args(
            ["--test_selected_subsets", "none", "--num_vit_adapter", "3",
             "--num_context_embedding", "8"])
        model = build_model(args)
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=False)["model"])
    finally:
        os.chdir(cwd)
        sys.path.pop(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)
    tf = transforms.Compose([                     # utils/dataset.py, test split
        transforms.Resize((args.img_resolution, args.img_resolution)),
        transforms.CenterCrop(args.crop_resolution),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def score(imgs):
        return _torch_batch(model, tf, [im.convert("RGB") for im in imgs], device, softmax_idx=1)
    return score


def build_detectors(a):
    d = {"quorum": quorum_scorer()}
    w = Path(a.weights_dir)
    cnn = w / "blur_jpg_prob0.5.pth"
    if cnn.exists():
        d["cnndetection_crop"] = cnndetection_scorer(cnn, resize=False)
        d["cnndetection_resize"] = cnndetection_scorer(cnn, resize=True)
    fat, repo = w / "fatformer_4class.pth", Path(a.fatformer_dir or "")
    if fat.exists() and repo.is_dir():
        d["fatformer"] = fatformer_scorer(fat, repo.resolve())
    return d


# --------------------------------------------------------------------------

def run(files, detectors, batch):
    """({detector: {setting: scores}}, n scored). One decode per image.

    A corrupt file in a downloaded probe set is not a reason to lose a 20-minute
    sweep, so it is skipped and counted out of the label vector.
    """
    out, kept = {k: {s: [] for s in SETTINGS} for k in detectors}, 0
    for i in range(0, len(files), batch):
        imgs = []
        for p in files[i:i + batch]:
            try:
                imgs.append(normalise(Image.open(p)))
            except Exception as e:
                print(f"    skip {p.name}: {e}", flush=True)
        if not imgs:
            continue
        kept += len(imgs)
        vs = [variants(im, SETTINGS) for im in imgs]
        for s in SETTINGS:
            batch_imgs = [v[s] for v in vs]
            for name, fn in detectors.items():
                out[name][s].append(fn(batch_imgs))
        print(f"    {min(i + batch, len(files))}/{len(files)}", flush=True)
    return {k: {s: np.concatenate(v) for s, v in d.items()} for k, d in out.items()}, kept


def render(results):
    """AUROC tables, one per population, plus the mean-over-settings column."""
    out, dets = [], sorted({d for r in results.values() for d in r} - {"n"})
    pops = sorted({k.split("/")[0] for k in results})
    for pop in pops:
        rows = {k.split("/")[1]: v for k, v in results.items() if k.startswith(pop + "/")}
        out.append(f"\n### {pop}\n")
        out.append("| generator | detector | " + " | ".join(SETTINGS) + " | mean | drop |")
        out.append("|---|---|" + "---|" * (len(SETTINGS) + 2))
        for gen, per_det in rows.items():
            for det in dets:
                a = [per_det[det][s]["auroc"] for s in SETTINGS]
                out.append(f"| {gen} | {det} | " + " | ".join(f"{x:.3f}" for x in a)
                           + f" | {np.mean(a):.3f} | {a[0] - min(a):.3f} |")
    return "\n".join(out)


def main(a):
    if a.render:
        print(render(json.loads(Path(a.json).read_text())))
        return
    results = {}
    for pop, (real_dirs, fake_map) in POPULATIONS.items():
        if a.only and a.only != pop:
            continue
        print(f"== {pop}: reals")
        r, n_real = run(sample(real_dirs, a.n_real), DETECTORS, a.batch)
        for gen, dirs in fake_map.items():
            print(f"== {pop}: {gen}")
            f, n_fake = run(sample(dirs, a.n_fake), DETECTORS, a.batch)
            for det in DETECTORS:
                for s in SETTINGS:
                    y = np.r_[np.zeros(n_real), np.ones(n_fake)]
                    p = np.r_[r[det][s], f[det][s]]
                    row = results.setdefault(f"{pop}/{gen}", {}).setdefault(det, {})
                    row[s] = {"auroc": round(float(roc_auc_score(y, p)), 4),
                              "acc": round(float(((p >= 0.5) == y).mean()), 4)}
            results[f"{pop}/{gen}"]["n"] = {"real": n_real, "fake": n_fake}
            print(json.dumps({d: results[f"{pop}/{gen}"][d]["clean"] for d in DETECTORS}, indent=1))
            Path(a.json).write_text(json.dumps(results, indent=1))   # crash-proof
    Path(a.json).write_text(json.dumps(results, indent=1))
    print(f"wrote {a.json}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights-dir", default=os.environ.get("QUORUM_BASELINE_DIR", "data/baselines"))
    p.add_argument("--fatformer-dir", default=os.environ.get("QUORUM_FATFORMER_DIR"))
    p.add_argument("--n-real", type=int, default=600)
    p.add_argument("--n-fake", type=int, default=600)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--only", help="one population name")
    p.add_argument("--json", default="docs/baselines.json")
    p.add_argument("--render", action="store_true", help="markdown from an existing --json")
    a = p.parse_args()
    if a.render:
        main(a)
        raise SystemExit
    DETECTORS = build_detectors(a)
    print("detectors:", list(DETECTORS))
    main(a)
