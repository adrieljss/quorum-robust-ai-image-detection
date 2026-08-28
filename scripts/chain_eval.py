"""Chained-degradation evaluation: does the detector survive COMPOSED transforms?

    python scripts/chain_eval.py --self-check           # 2s, no download, no GPU
    python scripts/chain_eval.py --n 100 --out c100.npz # 200 images, ~70 min

211 embeds per image (15 singles + 14x14 chains), measured at ~20s/image on an
8GB card. The first ~5 min of any run is the parquet shard download, not compute.

**Use --n 100 or higher.** One shard holds ~1,986 rows, of which ~2/3 survive
(tampered rows are dropped, then ~1/3 more as calib_ood), so a single shard
covers --n 100 easily and --max-shards rarely needs raising. At --n 25 the AUROC
standard error is ~0.04, which is wider than the entire spread of the worst-8
chain table: that ranking is then pure noise. Always pass --out; a re-run costs
another 2.9GB download.

Every number in docs/robustness.md is a SINGLE transform, because that is all
the official grid and our cache contain. Real images arrive composed: the upload
resizes, the platform recompresses, someone screenshots the result. This script
is the only place that gap is measured.

It reports accuracy and precision alongside AUROC. AUROC is threshold-free and
therefore says nothing about whether the shipped 0.5 cut still works after
degradation -- which is the question a deployment actually asks.

Pixels are not on disk (the streaming pass discarded them), so this re-downloads
one So-Fake-OOD shard, scores it, and deletes it. Rows are kept ONLY if the
manifest says split == test_ood: calib_ood images were fitted on by the
calibrators, and scoring them here would flatter every number below.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from quorum.degrade import FLAT, apply, rng_for, variant_name
from quorum.embed import Embedder, image_id, normalise

NAMES = [variant_name(k, p) for k, p in FLAT]
MANIFEST = ROOT / "data" / "manifests" / "main.csv"
THR = 0.5


def metrics(y, p, thr=THR):
    """AUROC + the threshold-dependent three. Returns None for AUROC when the
    population is single-class -- that is a real state here (a shard can hand
    back all-real rows) and must not read as a score of 0."""
    y, p = np.asarray(y), np.asarray(p)
    hat = (p >= thr).astype(int)
    tp = int(((hat == 1) & (y == 1)).sum())
    fp = int(((hat == 1) & (y == 0)).sum())
    fn = int(((hat == 0) & (y == 1)).sum())
    return {
        "auroc": roc_auc_score(y, p) if len(set(y.tolist())) > 1 else None,
        "acc": float((hat == y).mean()),
        "prec": tp / (tp + fp) if tp + fp else float("nan"),
        "rec": tp / (tp + fn) if tp + fn else float("nan"),
        "n": len(y),
    }


def fmt(tag, m):
    a = "  n/a  " if m["auroc"] is None else f"{m['auroc']:7.4f}"
    return (f"{tag:24}{m['n']:>7,}{a}{m['acc']:>8.3f}"
            f"{m['prec']:>8.3f}{m['rec']:>8.3f}")


HEAD = f"{'setting':24}{'n':>7}{'AUROC':>7}{'acc':>8}{'prec':>8}{'recall':>8}"


def load_probes():
    from predict import load_probes as _lp
    return _lp()


def shipped(V, probes):
    """predict.py's score, imported not reimplemented. (k, 768) -> (k,).

    This function used to Platt-calibrate the branches first, which is a
    DIFFERENT model from the one predict.py ships, and reported it under this
    name. Import the definition; never re-derive it.
    """
    from predict import score_embeddings
    return score_embeddings(V, probes)


def eval_one(img, iid, emb, probes):
    """-> (clean, singles[14], chains[196]) for one image.

    Generated and embedded one stage-1 row at a time: holding 211 full-res PIL
    images is ~900MB, and this loops over hundreds of images.
    """
    singles = [apply(img, k, p, rng_for(iid, i)) for i, (k, p) in enumerate(FLAT)]
    head = shipped(emb.embed_batch([img] + singles), probes)

    chains = np.empty((len(FLAT), len(FLAT)), np.float32)
    for i, s1 in enumerate(singles):
        batch = [apply(s1, k2, p2, rng_for(iid, 100 + 14 * i + j))
                 for j, (k2, p2) in enumerate(FLAT)]
        chains[i] = shipped(emb.embed_batch(batch), probes)
    return float(head[0]), head[1:], chains.ravel()


def collect(a, emb, probes):
    """Stream one shard, keep held-out rows, score each. -> (y, clean, S, C)."""
    from stream_embed import iter_shards, to_label

    m = pd.read_csv(MANIFEST, usecols=["image_id", "split"]).drop_duplicates("image_id")
    split_of = dict(zip(m.image_id, m.split))

    y, clean, S, C, ids = [], [], [], [], []
    counts = {0: 0, 1: 0}
    seen = skipped = 0
    for ex in iter_shards(argparse.Namespace(dataset=a.dataset, split=a.split,
                                             shuffle=True, max_shards=a.max_shards)):
        lab = to_label(ex.get("label", ""))
        if lab is None or counts[lab] >= a.n:
            if all(c >= a.n for c in counts.values()):
                break
            continue
        seen += 1
        img = normalise(ex["image"])
        iid = image_id(img)
        # The guard that makes this an EVAL: calib_ood was fitted on.
        if split_of.get(iid) != "test_ood":
            skipped += 1
            continue
        counts[lab] += 1
        c, s, ch = eval_one(img, iid, emb, probes)
        y.append(lab); clean.append(c); S.append(s); C.append(ch); ids.append(iid)
        print(f"  [{sum(counts.values()):4d}] label={lab} clean={c:.4f} "
              f"chain_worst={ch.min():.4f}", file=sys.stderr)

    if skipped:
        print(f"\nskipped {skipped}/{seen} streamed rows: not split==test_ood "
              f"(calib_ood, or never embedded)", file=sys.stderr)
    # image_id travels with the scores: the shard is deleted after the pass, so
    # without it a false positive found here can never be looked at again. Stage 5
    # wants those cases by name.
    return (np.array(y), np.array(clean), np.array(S, np.float32),
            np.array(C, np.float32), np.array(ids))


def report(y, clean, S, C):
    n_s, n_c = S.shape[1], C.shape[1]
    print(f"\n{HEAD}\n" + "-" * len(HEAD))
    print(fmt("clean", metrics(y, clean)))
    print(fmt(f"singles pooled ({n_s})", metrics(np.repeat(y, n_s), S.ravel())))
    print(fmt(f"chains pooled ({n_c})", metrics(np.repeat(y, n_c), C.ravel())))

    per_s = [metrics(y, S[:, j]) for j in range(n_s)]
    per_c = [metrics(y, C[:, j]) for j in range(n_c)]
    ws = int(np.argmin([m["auroc"] for m in per_s]))
    wc = int(np.argmin([m["auroc"] for m in per_c]))
    print("-" * len(HEAD))
    print(fmt(f"worst single: {NAMES[ws]}", per_s[ws]))
    print(fmt(f"worst chain: {NAMES[wc // n_s][:6]}>{NAMES[wc % n_s][:6]}", per_c[wc]))

    print("\nworst 8 chains by AUROC:")
    for j in sorted(range(n_c), key=lambda j: per_c[j]["auroc"])[:8]:
        m = per_c[j]
        print(f"  {NAMES[j // n_s]:10} -> {NAMES[j % n_s]:10} "
              f"auroc {m['auroc']:.4f}  acc {m['acc']:.3f}  prec {m['prec']:.3f}")

    cl, ch = metrics(y, clean), metrics(np.repeat(y, n_c), C.ravel())
    print(f"\nchaining vs clean:  AUROC {ch['auroc'] - cl['auroc']:+.4f}   "
          f"acc {ch['acc'] - cl['acc']:+.3f}   prec {ch['prec'] - cl['prec']:+.3f}")
    sp = metrics(np.repeat(y, n_s), S.ravel())
    print(f"chaining vs singles: AUROC {ch['auroc'] - sp['auroc']:+.4f}   "
          f"acc {ch['acc'] - sp['acc']:+.3f}   prec {ch['prec'] - sp['prec']:+.3f}")


def self_check():
    y = np.array([0, 0, 1, 1])
    m = metrics(y, np.array([0.1, 0.4, 0.6, 0.9]))
    assert m["auroc"] == 1.0 and m["acc"] == 1.0 and m["prec"] == 1.0, m
    m = metrics(y, np.array([0.9, 0.6, 0.4, 0.1]))          # perfectly inverted
    assert m["auroc"] == 0.0 and m["acc"] == 0.0 and m["prec"] == 0.0, m
    m = metrics(y, np.array([0.9, 0.4, 0.6, 0.4]))          # 1 fp, 1 fn
    assert m["prec"] == 0.5 and m["rec"] == 0.5 and m["acc"] == 0.5, m
    assert metrics(np.array([0, 0]), np.array([0.1, 0.2]))["auroc"] is None
    assert len(NAMES) == 14 and NAMES[0] == "jpeg90", NAMES[:2]
    print("chain_eval self-check ok (metrics + grid names)")


def main(a):
    if a.self_check:
        return self_check()
    probes = load_probes()
    emb = Embedder()
    y, clean, S, C, ids = collect(a, emb, probes)
    if len(y) < 2:
        raise SystemExit(f"only {len(y)} usable rows -- raise --max-shards")
    report(y, clean, S, C)
    # Distance from the correct answer, on ONE scale so the classes are
    # comparable: a real scored 0.94 and a fake scored 0.06 are equally wrong.
    # Ranking reals by -score instead puts a correctly-classified real above a
    # badly-missed fake, which is not what "worst" means.
    wrong = np.where(y == 0, clean, 1 - clean)
    print("\nmost confident errors on CLEAN (for Stage 5):")
    for i in np.argsort(-wrong)[:5]:
        # flagged a real, or missed a fake -- both are "predicted AI == is real"
        err = (clean[i] >= THR) == (y[i] == 0)
        kind = "false positive" if y[i] == 0 else "false negative"
        print(f"  {ids[i]}  label={y[i]}  clean={clean[i]:.4f}  "
              f"{kind if err else 'correct (near miss)'}")
    if a.out:
        np.savez(a.out, y=y, clean=clean, singles=S, chains=C,
                 names=NAMES, image_id=ids)
        print(f"\nraw scores -> {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=25, help="images PER CLASS (default 25)")
    p.add_argument("--max-shards", type=int, default=1)
    p.add_argument("--dataset", default="saberzl/So-Fake-OOD")
    p.add_argument("--split", default="test_image")
    p.add_argument("--out", metavar="NPZ", help="save raw scores for re-analysis")
    p.add_argument("--self-check", action="store_true")
    main(p.parse_args())
