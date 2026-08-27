"""Stream a HF dataset, embed degradation variants, never touch disk.

Train splits get clean + 3 sampled variants; eval splits get the full 15-way
grid (--full-grid), because the robustness table is a required deliverable.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from anywhere
# Default is 10s. So-Fake-OOD reads 103MB parquet row groups, which takes far
# longer than that whenever anything else is using the link -- e.g. a second
# stream_embed running in parallel. Must be set before huggingface_hub imports.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")

import argparse

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from quorum.embed import ROOT, Embedder, ShardWriter, embed_variants
from quorum.features import extract_variants

SHARD_DIR = ROOT / "data" / "raw" / "_shards"

# CONFIRMED 2026-08-26 against SID_Set img_id prefixes (label 0 has a bare hash
# id, 1 -> full_synthetic_*, 2 -> tampered_*). So-Fake-OOD spells the same three
# out as strings (REAL / FULL_SYNTHETIC / TAMPERED), handled by the str path.
INT_MAP = {0: 0, 1: 1, 2: None}          # real / full_synthetic / tampered
SUBCLASS = {0: "real", 1: "full_synthetic", 2: "tampered"}
LABEL_CHECK_AT = 300                     # images before demanding both classes
TAMPERED_ONLY = False                    # --tampered flips this


def to_label(v):
    """-> 0 real, 1 AI, None skip.

    --tampered inverts the selection: keep ONLY class 2 (a real photo with an
    AI-edited region) and label it AI. Localised editing is a different task
    from fully-synthetic detection, so it gets its own source and its own rows;
    whoever trains decides later whether to mix them.
    """
    if isinstance(v, (int, np.integer)) or (isinstance(v, str) and v.strip().isdigit()):
        return INT_MAP.get(int(v))
    s = str(v).lower()
    if "tamper" in s:
        return 1 if TAMPERED_ONLY else None
    return None if TAMPERED_ONLY else (0 if "real" in s else 1)


def iter_shards(a):
    """Download one parquet shard, yield its rows, delete it, repeat.

    Streaming So-Fake-OOD reads 103MB row groups over a live socket and times
    out on an unreliable link. hf_hub_download is a plain resumable file
    transfer, so a blip costs seconds instead of the whole run. Disk stays at
    one shard (~3GB) because each is deleted once consumed.
    """
    import random

    from huggingface_hub import HfApi, hf_hub_download

    files = sorted(f for f in HfApi().list_repo_files(a.dataset, repo_type="dataset")
                   if f.endswith(".parquet") and a.split in f)
    if a.shuffle:
        random.Random(42).shuffle(files)      # shard order = generator diversity
    print(f"{len(files)} shards; downloading up to {a.max_shards or len(files)}")

    for i, fn in enumerate(files[:a.max_shards or None]):
        print(f"[shard {i+1}] downloading {fn}")
        path = hf_hub_download(a.dataset, fn, repo_type="dataset",
                               local_dir=str(SHARD_DIR))
        try:
            yield from load_dataset("parquet", data_files=path, split="train",
                                    streaming=True)
        finally:
            Path(path).unlink(missing_ok=True)   # keep disk at one shard


def iter_stream(a):
    ds = load_dataset(a.dataset, split=a.split, streaming=True)
    if a.skip:
        ds = ds.skip(a.skip)          # resume: same seed + same skip = same position
    if a.shuffle:
        # The point of shuffling here is SHARD ORDER, not the row buffer: without
        # it So-Fake-OOD hands you its all-real head. Keep the buffer small -- it
        # is prefilled before the first row appears, and at 1.4MB/image a buffer
        # of 1000 means 1.4GB of dead air that reads as a hang.
        ds = ds.shuffle(seed=42, buffer_size=a.shuffle_buffer)
    return ds


def main(a):
    global TAMPERED_ONLY
    if a.tampered:
        INT_MAP.update({0: None, 1: None, 2: 1})
        TAMPERED_ONLY = True
    emb = None if a.no_embed else Embedder()
    img_dir = None
    if a.save_images:
        img_dir = ROOT / 'data' / 'raw' / 'images' / a.source
        img_dir.mkdir(parents=True, exist_ok=True)
    # --features reuses this whole loop -- same stream, same quotas, same resume.
    # Face crops go through the SAME CLIP instance as the general embeddings,
    # which is what holds the parameter budget at ~317M (PIPELINE 2.2).
    # Face rows arrive ~4x slower than spectral (only ~25% of images yield a
    # detectable face), so it needs a proportionally smaller flush or a kill
    # throws away far more of it -- which is exactly what happened once.
    writers = ([ShardWriter("face_" + a.source, every=1_000),
                ShardWriter("spec_" + a.source)]
               if a.features else [ShardWriter(a.source)])
    ds = iter_shards(a) if a.via_download else iter_stream(a)

    # Only the classes this run can actually produce, so --tampered (one class)
    # does not wait forever for a quota that can never fill.
    counts = {y: 0 for y in sorted({v for v in INT_MAP.values() if v is not None})}
    seen, rows_seen = 0, a.skip
    pbar = tqdm(total=a.n_per_class * len(counts), desc=a.source)

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
          meta = {"label": y, "source": a.source, "subclass": sub,
                  "generator": str(ex.get("generator", "unknown")),
                  "split": a.assign_split}
          if a.features:
              extract_variants(emb, writers[0], writers[1], ex[a.image_field],
                               meta, a.full_grid, a.n_sampled)
          else:
              embed_variants(emb, writers[0], ex[a.image_field], meta,
                             a.full_grid, a.n_sampled, img_dir)
          pbar.update(1)

    except KeyboardInterrupt:
        print("\ninterrupted by user")
    except Exception as e:
        print(f"\nSTREAM FAILED: {type(e).__name__}: {e}")
    finally:
        # Always keep what was already embedded -- a multi-hour job must not lose
        # an hour of GPU time to a dropped connection.
        pbar.close()
        for w in writers:
            w.flush()
        print(f"done: {sum(counts.values())} images {counts}  rows_seen={rows_seen}")
        if sum(counts.values()) < len(counts) * a.n_per_class:
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
    p.add_argument("--via-download", action="store_true",
                   help="download parquet shards instead of streaming -- slower to "
                        "start, but resumable and immune to read timeouts")
    p.add_argument("--max-shards", type=int, default=0,
                   help="with --via-download, stop after N shards (0 = all)")
    p.add_argument("--shuffle-buffer", type=int, default=200,
                   help="rows prefilled before the first output; keep small on "
                        "datasets with large images")
    p.add_argument("--tampered", action="store_true",
                   help="embed ONLY SID_Set class 2 (locally edited) as label 1")
    p.add_argument("--features", action="store_true",
                   help="face-crop embeddings + spectral vectors instead of general embeddings")
    p.add_argument("--save-images", action="store_true",
                   help="cache normalised clean JPEGs for the pixel branches")
    p.add_argument("--no-embed", action="store_true",
                   help="pixels only, no GPU -- pair with --save-images")
    p.add_argument("--skip", type=int, default=0,
                   help="skip N rows first -- resume a crashed run (see its final line)")
    main(p.parse_args())
