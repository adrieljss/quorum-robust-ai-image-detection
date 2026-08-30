"""docs/figures/error-cases.png -- the representative errors for the Error
Analysis Note, drawn from the cache instead of pasted in.

Kept out of make_figures.py because it is the one figure that needs PIXELS, not
embeddings: it reads data/raw/organizer_val/, which is gitignored. The other six
regenerate from the cache alone and must keep doing so.

Cases are ranked by how STABLY wrong they are -- mean shipped score across all
15 degradation variants -- not by their clean score. A borderline clean miss
teaches nothing; a photograph called fake under every transformation teaches a
lot. All ten below are wrong in 15/15.

    python scripts/error_cases.py

Needs the organizer_val download (RUNBOOK Step 2) and, on first run, ~2 min to
recover image_id -> path by re-hashing it the way the embedding pass did.
"""
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from predict import score_embeddings
from quorum.detectors.general import MODEL, MODEL_TAMPERED, load

FIGURES = ROOT / "docs" / "figures"
RAW = ROOT / "data" / "raw" / "organizer_val"
PATHS = ROOT / "data" / "cache" / "orgval_paths.pkl"
RED, BLUE = "#B4462F", "#2B4A9B"
N = 5
# Two WildFake images differ in bytes (their filenames are content md5s) but are
# the same picture at 45KB and 1.4MB. Both are false negatives, and showing the
# pair twice would waste a panel, so near-duplicates are skipped by cosine.
# Measured at 0.9771 for that pair -- 0.98 lets it straight through.
DUP = 0.95


def paths():
    """image_id -> file. Cached: recomputing means re-hashing 8,719 images."""
    if PATHS.exists():
        return pickle.load(open(PATHS, "rb"))
    from quorum.embed import image_id, normalise
    ps = sorted(p for d in ("coco_val2017", "wildfake_dalle_adv")
                for p in (RAW / d).rglob("*")
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    m = {image_id(normalise(Image.open(p))): p for p in ps}
    PATHS.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(m, open(PATHS, "wb"))
    return m


def probe(p):
    d = np.load(p)
    return d["w"].ravel().astype(np.float32), float(d["b"].ravel()[0])


def cases():
    X, R = load("organizer_val")
    (wg, bg), (wt, bt) = probe(MODEL), probe(MODEL_TAMPERED)
    R = R.assign(p=score_embeddings(X), gen=X @ wg + bg, tam=X @ wt + bt)
    per = R.groupby("image_id").agg(p=("p", "mean"), label=("label", "first"),
                                    gen=("gen", "mean"), tam=("tam", "mean"))
    # One clean embedding per image, for the near-duplicate check.
    c = R.variant == "clean"
    emb = dict(zip(R.image_id[c], X[c.values]))

    def pick(label, ascending):
        out = []
        for iid, r in per[per.label == label].sort_values("p", ascending=ascending).iterrows():
            v = emb[iid]
            if any(float(v @ emb[j]) > DUP for j, _ in out):
                continue
            out.append((iid, r))
            if len(out) == N:
                return out
        return out

    return pick(0, False), pick(1, True)


def panel(ax, path, title, sub, colour):
    """Letterboxed onto one square canvas so every panel has the SAME aspect.

    Without it each row's captions sit at a different height -- COCO is 4:3,
    WildFake is 1:1 -- and the first row's filenames land on top of the second
    row's titles.
    """
    im = Image.open(path).convert("RGB")
    im.thumbnail((420, 420))
    sq = Image.new("RGB", (420, 420), "white")
    sq.paste(im, ((420 - im.width) // 2, (420 - im.height) // 2))
    ax.imshow(sq)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(colour); s.set_linewidth(2.5)
    ax.set_title(title, fontsize=9, fontweight="bold", color=colour, pad=4)
    ax.set_xlabel(sub, fontsize=7.5, color="#444", labelpad=3)


def main():
    P = paths()
    fp, fn = cases()
    fig, axes = plt.subplots(2, N, figsize=(3.1 * N, 8.0))

    for ax, (iid, r) in zip(axes[0], fp):
        who = "tampered" if r.tam > r.gen else "general"
        panel(ax, P[iid], f"pred {r.p:.2f}  -- real photo",
              f"{P[iid].name}\n{who} branch fires  (gen {r.gen:+.1f} / tam {r.tam:+.1f})", RED)
    for ax, (iid, r) in zip(axes[1], fn):
        panel(ax, P[iid], f"pred {r.p:.2f}  -- AI, missed",
              f"{P[iid].name[:20]}...\ngen {r.gen:+.1f} / tam {r.tam:+.1f}", BLUE)

    fig.suptitle("Quorum's most stable errors -- wrong in 15 of 15 degradation variants",
                 fontsize=13, fontweight="bold", y=0.985)
    fig.text(0.5, 0.945, "top: real COCO photographs called AI    "
                         "bottom: WildFake DALL\u00b7E 3 images called real",
             ha="center", fontsize=9.5, color="#555")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.subplots_adjust(hspace=0.28)
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "error-cases.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
