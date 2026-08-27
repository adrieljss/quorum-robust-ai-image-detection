#!/bin/sh
# Everything left that does not need a human. WildFake DALL-E Advanced is not
# here: it lives on ModelScope with no HF mirror, so it stays manual.
set -e
echo "=== [1/4] organizer_val: COCO val2017, full grid (quarantined) ==="
python scripts/embed_dir.py --dir data/raw/organizer_val/coco_val2017 \
  --source organizer_val --assign-split test_organizer --label 0 --full-grid

echo "=== [2/4] features: so_fake_ood ==="
python scripts/stream_embed.py --features --shuffle --dataset saberzl/So-Fake-OOD \
  --split test_image --source so_fake_ood --assign-split test_ood \
  --n-per-class 3000 --full-grid --via-download

echo "=== [3/4] features: sid_tampered_eval ==="
python scripts/stream_embed.py --features --shuffle --dataset saberzl/SID_Set \
  --split validation --source sid_tampered_eval --assign-split test_ood \
  --n-per-class 1500 --tampered --full-grid

echo "=== [4/4] done -- build_manifest still needs WildFake ==="
