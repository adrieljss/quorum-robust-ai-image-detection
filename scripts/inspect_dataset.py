"""Print one record's fields. Run before every real streaming pass --
guessing field names or label encodings costs two hours."""
import sys

from datasets import load_dataset

name = sys.argv[1]
split = sys.argv[2] if len(sys.argv) > 2 else "train"
ex = next(iter(load_dataset(name, split=split, streaming=True)))
for k, v in ex.items():
    shown = getattr(v, "size", None) or str(v)[:70]
    print(f"{k:20s} {type(v).__name__:12s} {shown}")
