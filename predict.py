"""Required deliverable: directory of images -> JSON of {image_path, pred}.

    python predict.py --input-dir path/to/images --output preds.json

`pred` is P(AI-generated), calibrated, in [0,1]. Two fields, no more -- the
demo's richer schema stays in the demo.

ponytail: scores are random until fusion lands. Wire quorum.pipeline into
score() and nothing else here changes.
"""
import argparse
import json
import random
from pathlib import Path

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def score(path: Path) -> float:
    return round(random.random(), 4)          # STUB


def main(a):
    paths = sorted(p for p in Path(a.input_dir).rglob("*") if p.suffix.lower() in EXTS)
    if not paths:
        raise SystemExit(f"no images under {a.input_dir}")
    preds = [{"image_path": p.as_posix(), "pred": score(p)} for p in paths]  # posix: judges may diff paths
    Path(a.output).write_text(json.dumps(preds, indent=2))
    print(f"{len(preds)} predictions -> {a.output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output", default="preds.json")
    main(p.parse_args())
