#!/bin/sh
# Wait out the so_fake_ood feature pass (PID survived its wrapper being killed),
# then run the last source. Sequential: two streams starve each other.
while [ "$(powershell -NoProfile -Command '@(Get-Process -Id 49092 -ErrorAction SilentlyContinue).Count')" = "1" ]; do
  sleep 30
done
echo "=== so_fake_ood finished, starting sid_tampered_eval ==="
python scripts/stream_embed.py --features --shuffle --dataset saberzl/SID_Set \
  --split validation --source sid_tampered_eval --assign-split test_ood \
  --n-per-class 1500 --tampered --full-grid
echo "=== all feature passes done ==="
