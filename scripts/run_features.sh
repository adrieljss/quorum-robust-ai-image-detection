#!/bin/sh
# Feature pass for the sources that still need one. sid_train (98.5%) and
# sid_calib (100%) are already covered -- re-running them costs ~90 min to
# recover 1.5%, and fusion left-joins missing rows to 0.5 anyway.
#
# Sequential: concurrent streams starve each other's sockets. Resumable:
# shards append and build_manifest dedupes on (image_id, variant).
set -e
S="python scripts/stream_embed.py --features --shuffle"
$S --dataset saberzl/So-Fake-OOD --split test_image --source so_fake_ood \
   --assign-split test_ood --n-per-class 3000 --full-grid --via-download
$S --dataset saberzl/SID_Set --split validation --source sid_tampered_eval \
   --assign-split test_ood --n-per-class 1500 --tampered --full-grid
$S --dataset saberzl/SID_Set --split train --source sid_tampered \
   --assign-split train --n-per-class 4000 --tampered
