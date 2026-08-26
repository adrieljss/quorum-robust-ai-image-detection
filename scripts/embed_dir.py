"""Same pass as stream_embed.py, for images already on disk.

The downloaded sets go through here -- organizer_val (quarantined), Open Images,
social reals -- so build_manifest.py's assertion A has rows to check.

    python scripts/embed_dir.py --dir data/raw/organizer_val/coco_val2017 \
      --source organizer_val --assign-split test_organizer --label 0 --full-grid
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from anywhere

import argparse
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from quorum.embed import Embedder, ShardWriter, embed_variants

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main(a):
    paths = sorted(p for p in Path(a.dir).rglob("*") if p.suffix.lower() in EXTS)
    if a.limit:
        paths = paths[:a.limit]
    if not paths:
        raise SystemExit(f"no images under {a.dir}")

    emb, writer = Embedder(), ShardWriter(a.source)
    for p in tqdm(paths, desc=a.source):
        try:
            img = Image.open(p)
        except Exception as e:
            print(f"  skip {p}: {e}")
            continue
        embed_variants(emb, writer, img, {
            "label": a.label, "source": a.source,
            "subclass": "real" if a.label == 0 else "full_synthetic",
            "generator": a.generator, "split": a.assign_split,
        }, a.full_grid, a.n_sampled)
    writer.flush()
    print(f"done: {len(paths)} images")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--assign-split", required=True)
    p.add_argument("--label", type=int, required=True, choices=[0, 1])
    p.add_argument("--generator", default="unknown")
    p.add_argument("--limit", type=int)
    p.add_argument("--full-grid", action="store_true")
    p.add_argument("--n-sampled", type=int, default=3)
    main(p.parse_args())
