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

URL = ("https://modelscope.cn/api/v1/datasets/hy2628982280/WildFake/repo"
       "?Revision=master&FilePath=Images/Diffusion_based/DALLE.zip")
PREFIX = "DALLE/Advanced/DALLE3"        # 8,843 imgs. Typical/DALLE2 is NOT this.
OUT = Path(__file__).resolve().parents[1] / "data/raw/organizer_val/wildfake_dalle_adv"
WORKERS = 8


def members(prefix=PREFIX):
    """One member per DISTINCT image, keyed by its content-hash basename."""
    z = zipfile.ZipFile(fsspec.open(URL, "rb").open())
    out = {}
    for n in z.namelist():
        if n.startswith(prefix) and not n.endswith("/"):
            out.setdefault(Path(n).name, n)      # first wins; copies are identical
    return list(out.values())


def main():
    names = members()
    OUT.mkdir(parents=True, exist_ok=True)
    for stale in OUT.glob("*.part"):             # truncated by an earlier kill
        stale.unlink()
    todo = [n for n in names if not (OUT / Path(n).name).exists()]
    print(f"{len(names):,} in {PREFIX}, {len(names)-len(todo):,} already local, "
          f"{len(todo):,} to fetch -> {OUT}")

    # One ZipFile per worker: a shared handle seeks under itself across threads.
    # Re-parsing the central directory 8x costs ~50MB against a 3.5GB transfer.
    local = __import__("threading").local()
    done = [0]

    def grab(name):
        if not hasattr(local, "z"):
            local.z = zipfile.ZipFile(fsspec.open(URL, "rb").open())
        # .part then rename: a kill mid-write must not leave a truncated JPEG
        # that the resume check above would then happily skip.
        dst = OUT / Path(name).name
        tmp = dst.with_suffix(dst.suffix + ".part")
        tmp.write_bytes(local.z.read(name))
        os.replace(tmp, dst)     # Path.rename raises on Windows if dst exists
        done[0] += 1
        if done[0] % 250 == 0:
            print(f"  {done[0]:,}/{len(todo):,}", flush=True)

    with ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(grab, todo))
    n = len([p for p in OUT.iterdir() if p.suffix != ".part"])
    print(f"{n:,} images in {OUT}")
    assert n >= len(names), f"expected {len(names):,}, got {n:,}"


if __name__ == "__main__":
    if "--list" in sys.argv:          # the check: central dir reads without a download
        m = members()
        print(f"{len(m):,} distinct images under {PREFIX}")
        assert len(m) == 3719, f"subset changed upstream: {len(m):,} != 3,719"
        print("ok")
    else:
        main()
