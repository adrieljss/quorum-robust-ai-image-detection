"""Stream a HF dataset, embed degradation variants, never touch disk.

Train splits get clean + 3 sampled variants; eval splits get the full 15-way
grid (--full-grid), because the robustness table is a required deliverable.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from anywhere

import argparse

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from quorum.embed import Embedder, ShardWriter, embed_variants

# CONFIRMED 2026-08-26 against SID_Set img_id prefixes (label 0 has a bare hash
# id, 1 -> full_synthetic_*, 2 -> tampered_*). So-Fake-OOD spells the same three
# out as strings (REAL / FULL_SYNTHETIC / TAMPERED), handled by the str path.
INT_MAP = {0: 0, 1: 1, 2: None}          # real / full_synthetic / tampered
SUBCLASS = {0: "real", 1: "full_synthetic", 2: "tampered"}
LABEL_CHECK_AT = 300                     # images before demanding both classes


def to_label(v):
    """-> 0 real, 1 AI, None skip (tampered: a different task)."""
    if isinstance(v, (int, np.integer)) or (isinstance(v, str) and v.strip().isdigit()):
        return INT_MAP.get(int(v))
    s = str(v).lower()
    if "tamper" in s:
        return None
    return 0 if "real" in s else 1


def main(a):
    emb = Embedder()
    writer = ShardWriter(a.source)

    ds = load_dataset(a.dataset, split=a.split, streaming=True)
    if a.skip:
        ds = ds.skip(a.skip)          # resume: same seed + same skip = same position
    if a.shuffle:
        # Small buffer on purpose: SID_Set is already class-interleaved (verified),
        # so the value here is shard-order shuffling, not the buffer. A 10k buffer
        # downloads ~5.7GB before the progress bar moves once -- looks like a hang.
        ds = ds.shuffle(seed=42, buffer_size=1_000)

    counts, seen, rows_seen = {0: 0, 1: 0}, 0, a.skip
    pbar = tqdm(total=a.n_per_class * 2, desc=a.source)

    try:
      for ex in ds:
          rows_seen += 1
          raw = ex.get(a.label_field, "")
          y = to_label(raw)
          if y is None:
              continue

          seen += 1
          # A wrong label mapping silently produces a single-class training set
          # and streams the whole dataset to do it. Fail in seconds instead.
          if seen == LABEL_CHECK_AT and min(counts.values()) == 0:
              raise SystemExit(
                  f"label mapping looks wrong: {seen} images, counts={counts}, "
                  f"last {a.label_field}={raw!r}. Run scripts/inspect_dataset.py "
                  f"and fix to_label()/INT_MAP."
              )

          if counts[y] >= a.n_per_class:
              if all(c >= a.n_per_class for c in counts.values()):
                  break
              continue
          counts[y] += 1

          sub = SUBCLASS[int(raw)] if str(raw).strip().isdigit() else str(raw).lower()
          embed_variants(emb, writer, ex[a.image_field], {
              "label": y, "source": a.source, "subclass": sub,
              "generator": str(ex.get("generator", "unknown")),
              "split": a.assign_split,
          }, a.full_grid, a.n_sampled)
          pbar.update(1)

    except KeyboardInterrupt:
        print("\ninterrupted by user")
    except Exception as e:
        print(f"\nSTREAM FAILED: {type(e).__name__}: {e}")
    finally:
        # Always keep what was already embedded -- a multi-hour job must not lose
        # an hour of GPU time to a dropped connection.
        pbar.close()
        writer.flush()
        print(f"done: {sum(counts.values())} images {counts}  rows_seen={rows_seen}")
        if sum(counts.values()) < 2 * a.n_per_class:
            print(f"INCOMPLETE -- resume with: --skip {rows_seen} "
                  f"--n-per-class {a.n_per_class - min(counts.values())} "
                  f"--source {a.source}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--source", required=True)
    p.add_argument("--assign-split", required=True)
    p.add_argument("--n-per-class", type=int, default=20_000)
    p.add_argument("--image-field", default="image")
    p.add_argument("--label-field", default="label")
    p.add_argument("--full-grid", action="store_true", help="all 15 variants -- EVAL splits")
    p.add_argument("--n-sampled", type=int, default=3, help="variants/image when not --full-grid")
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--skip", type=int, default=0,
                   help="skip N rows first -- resume a crashed run (see its final line)")
    main(p.parse_args())
