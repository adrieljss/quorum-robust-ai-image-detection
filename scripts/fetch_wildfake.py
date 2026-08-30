"""Pull ONLY the DALL-E Advanced subset out of WildFake, without downloading it.

ModelScope ships every WildFake DALL-E image in one 25.6GB zip, and the
organizer's benchmark is 8,843 of them. A zip's central directory lives at the
tail and the CDN honours Range, so fsspec + stdlib zipfile can list all 64,495
members and inflate just the ~1.5GB we want. No modelscope SDK, no 25.6GB.

Those 8,843 entries are only 3,719 DISTINCT images: basenames are content
hashes and the same image is filed under several prompt-run folders. Verified
from the central directory -- every one of the 1,808 colliding basenames has an
identical CRC32 and size, and unique (CRC, size) pairs total 3,719 exactly. So
dedupe by basename and fetch each image once; embedding all 8,843 would spend
an hour on 5,124 duplicates that image_id would drop anyway.

    python scripts/fetch_wildfake.py

Then the two passes in docs/HANDOVER.md 'WildFake commands'. Re-running skips
what already landed, so a dropped connection costs only the current file.
"""
import os
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fsspec

BASE = ("https://modelscope.cn/api/v1/datasets/hy2628982280/WildFake/repo"
        "?Revision=master&FilePath=")
ZIP = "Images/Diffusion_based/DALLE.zip"
URL = BASE + ZIP
PREFIX = "DALLE/Advanced/DALLE3"        # 8,843 imgs. Typical/DALLE2 is NOT this.
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/raw/organizer_val/wildfake_dalle_adv"
WORKERS = 8

# The one subset the brief forbids training on, alongside COCO val2017. Every
# other prefix in WildFake is fair game -- but this script's DEFAULT prefix is
# the forbidden one, so any custom fetch is checked against it rather than
# trusting whoever typed the command.
QUARANTINE = "DALLE/Advanced/DALLE3"


def check_prefix(prefix, out, url=""):
    """Refuse to pull anything the brief forbids training on.

    Two forbidden sets, and this is the only place that knows about either:
    WildFake's DALL-E Advanced subset, and COCO val2017. The val2017 check earns
    its keep now that --url makes this a general zip-subset fetcher -- one typo
    of "val2017" for "train2017" would silently poison the only unseen-photography
    number we have.
    """
    if "val2017" in prefix or "val2017" in url:
        raise SystemExit(
            "REFUSED: COCO val2017 is the organizer validation set. The brief "
            "says 'Do not use the following data during training'. Use "
            "train2017, which is a different 118k images and is permitted.")
    if prefix.startswith(QUARANTINE) and Path(out).resolve() != OUT.resolve():
        raise SystemExit(
            f"REFUSED: {prefix!r} is the organizer validation subset. The brief "
            f"says 'Do not use the following data during training'. It belongs "
            f"only in {OUT}, never in a training source.")


def members(prefix=PREFIX, url=URL):
    """One member per DISTINCT image, keyed by its content-hash basename."""
    z = zipfile.ZipFile(fsspec.open(url, "rb").open())
    out = {}
    for n in z.namelist():
        if n.startswith(prefix) and not n.endswith("/"):
            out.setdefault(Path(n).name, n)      # first wins; copies are identical
    return list(out.values())


def main(prefix=PREFIX, url=URL, out=OUT, limit=0, seed=0):
    check_prefix(prefix, out, url)
    names = members(prefix, url)
    if limit and len(names) > limit:
        # Deterministic subsample: a family contributes DIVERSITY, and past a
        # few hundred images per generator the marginal one is nearly free of it.
        import random
        names = sorted(random.Random(seed).sample(names, limit))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("*.part"):             # truncated by an earlier kill
        stale.unlink()
    todo = [n for n in names if not (out / Path(n).name).exists()]
    print(f"{len(names):,} in {prefix}, {len(names)-len(todo):,} already local, "
          f"{len(todo):,} to fetch -> {out}")
    OUT_DIR = out

    # One ZipFile per worker: a shared handle seeks under itself across threads.
    # Re-parsing the central directory 8x costs ~50MB against a 3.5GB transfer.
    local = __import__("threading").local()
    done = [0]

    def grab(name):
        if not hasattr(local, "z"):
            local.z = zipfile.ZipFile(fsspec.open(url, "rb").open())
        # .part then rename: a kill mid-write must not leave a truncated JPEG
        # that the resume check above would then happily skip.
        dst = OUT_DIR / Path(name).name
        tmp = dst.with_suffix(dst.suffix + ".part")
        tmp.write_bytes(local.z.read(name))
        os.replace(tmp, dst)     # Path.rename raises on Windows if dst exists
        done[0] += 1
        if done[0] % 250 == 0:
            print(f"  {done[0]:,}/{len(todo):,}", flush=True)

    with ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(grab, todo))
    n = len([p for p in out.iterdir() if p.suffix != ".part"])
    print(f"{n:,} images in {out}")
    assert n >= len(names), f"expected {len(names):,}, got {n:,}"


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zip", default=ZIP, help="path inside the ModelScope repo")
    ap.add_argument("--url", default=None,
                    help="full URL of ANY range-serving zip, overriding --zip. "
                         "The central directory is read from the tail, so this "
                         "pulls a subset without downloading the archive "
                         "(COCO train2017 is 18GB; 5k images is ~700MB).")
    ap.add_argument("--prefix", default=PREFIX, help="prefix INSIDE the zip")
    ap.add_argument("--out", default=None, help="destination dir")
    ap.add_argument("--limit", type=int, default=0,
                    help="deterministic subsample; 0 = all")
    ap.add_argument("--list", action="store_true",
                    help="read the central directory only -- no download")
    ap.add_argument("--dirs", action="store_true",
                    help="list the top prefixes inside the zip and exit")
    a = ap.parse_args()
    url = a.url or (BASE + a.zip)
    out = Path(a.out) if a.out else OUT

    if a.dirs:                        # what families does this zip actually hold?
        import collections
        z = zipfile.ZipFile(fsspec.open(url, "rb").open())
        # group by CONTAINING directory: zip layouts here vary in depth, and a
        # fixed slice prints one line per file on the flat ones.
        c = collections.Counter(n.rsplit("/", 1)[0]
                                for n in z.namelist() if not n.endswith("/"))
        for k, v in sorted(c.items()):
            print(f"{v:8,d}  {k}")
        raise SystemExit

    if a.list:
        m = members(a.prefix, url)
        print(f"{len(m):,} distinct images under {a.prefix}")
        if a.prefix == PREFIX and a.zip == ZIP:
            assert len(m) == 3719, f"subset changed upstream: {len(m):,} != 3,719"
            print("ok")
        raise SystemExit

    main(a.prefix, url, out, a.limit)
