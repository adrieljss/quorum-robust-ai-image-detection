"""Every self-check in the repo, one command, one exit code.

    python scripts/selfcheck.py          # offline: no embedding cache, ~30s
    python scripts/selfcheck.py --all    # + the checks that read data/cache

The offline set runs on a fresh clone. The probes it needs (`data/models/*.npz`,
3.5KB) are tracked; the 1.2GB embedding cache is not, so anything that would
need `pull_cache.py` is held back for `--all`.

**A green offline run is not a licence to trust a number.** The checks that
catch a contaminated split -- disjointness, the manifest assertions -- are all in
the `--all` set, because they are the only ones that can see the data.

`ruff check` runs here; `ruff format` does not. The rule set is narrow on
purpose (see pyproject.toml) and the formatter is opt-in, because this codebase
hand-aligns comment columns that a Black-style pass would collapse.

Deliberately NOT run here, because they retrain and overwrite shipped weights
rather than check anything:

    python -m quorum.detectors.face     writes data/models/face.npz
    python -m quorum.fusion             writes data/models/fusion.npz, minutes
    python -m quorum.detectors.general  writes general.npz + tampered.npz
                                        (its `--check` half IS run below)
"""
import argparse
import importlib.util
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
CACHE = ROOT / "data" / "cache" / "embeddings"

# (label, argv, module that must be importable or the check is skipped)
OFFLINE = [
    ("ruff check         ", [PY, "-m", "ruff", "check", "."], "ruff"),
    ("predict            ", [PY, "predict.py", "--self-check"], None),
    ("quorum.degrade     ", [PY, "-m", "quorum.degrade"], None),
    ("quorum.embed       ", [PY, "-m", "quorum.embed"], None),
    ("quorum.calibrate   ", [PY, "-m", "quorum.calibrate"], "sklearn"),
    ("quorum.features    ", [PY, "-m", "quorum.features"], "cv2"),
    ("quorum...text      ", [PY, "-m", "quorum.detectors.text"], "rapidocr_onnxruntime"),
    ("chain_eval         ", [PY, "scripts/chain_eval.py", "--self-check"], None),
    ("app.app            ", [PY, "app/app.py", "--self-check"], "flask"),
]

# build_manifest rewrites data/manifests/main.csv -- that is its job, it is
# deterministic, and its assertions gate the write, so a partial cache fails
# before it can truncate anything.
CACHED = [
    ("build_manifest     ", [PY, "scripts/build_manifest.py"], None),
    ("splits disjoint    ", [PY, "-m", "quorum.detectors.general", "--check"], None),
    ("quorum...spectral  ", [PY, "-m", "quorum.detectors.spectral"], None),
]


def run(label, argv, needs):
    if needs and importlib.util.find_spec(needs) is None:
        print(f"SKIP  {label}  (no {needs})")
        return None
    t = time.perf_counter()
    r = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
    dt = time.perf_counter() - t
    ok = r.returncode == 0
    print(f"{'ok  ' if ok else 'FAIL'}  {label}  {dt:5.1f}s")
    if not ok:
        # The assertion line is the whole message; dumping 200 lines of pandas
        # output above it is how a red run gets ignored.
        tail = (r.stderr or r.stdout).strip().splitlines()[-12:]
        print("".join(f"        | {ln}\n" for ln in tail), end="")
    return ok


def main(do_all):
    checks = list(OFFLINE)
    if do_all:
        if not CACHE.exists():
            raise SystemExit(f"--all needs {CACHE.relative_to(ROOT)} "
                             f"-- run scripts/pull_cache.py first")
        checks += CACHED
    else:
        print("offline only -- pass --all to add the checks that read data/cache\n")

    res = [run(*c) for c in checks]
    ran = [r for r in res if r is not None]
    print(f"\n{sum(ran)}/{len(ran)} passed, {len(res) - len(ran)} skipped")
    return 0 if all(ran) else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--all", action="store_true",
                   help="also run the checks that need data/cache/embeddings")
    raise SystemExit(main(p.parse_args().all))
