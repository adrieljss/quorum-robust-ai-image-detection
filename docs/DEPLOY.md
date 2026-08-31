# Deploying the demo to a Hugging Face Space

The demo runs as a **Docker Space** on the free CPU tier. Flask serves the
frontend, the backend and `/api/analyze` from one URL, so there is no split
deployment and no CORS.

## Why this is cheap

The entire trained model is **305 KB** of `.npz` — seven files. Everything heavy
is CLIP, which is public and gets baked into the image at build time. Crucially
**the demo needs none of the datasets**: no embedding cache, no raw images. So
none of the SID_Set / So-Fake-OOD / CC BY-NC licence constraints follow it into
a public deployment.

`.dockerignore` enforces that boundary. It excludes `data/cache/` (1.9 GB) and
`data/raw/`, and it is a **licence boundary, not an optimisation** — a Space is
public. If a build fails on a missing file, fix the code path rather than
loosening that file.

## What the image carries

| | |
|---|---|
| `data/models/*.npz` | 305 KB — general, tampered, face, spectral, text_crop, fusion, content_prompts |
| `data/models/yunet.onnx` | 233 KB — face detection |
| CLIP ViT-L-14-quickgelu | 1.71 GB, baked at build time |
| RapidOCR PP-OCRv4 | ~10 MB, baked; `signals.text` is null without it |

## Deploy

1. **Create the Space** at <https://huggingface.co/new-space> under the
   `techjam2026blueberryjam` org. SDK **Docker**, blank template, **public**.

2. **Push this repo to it.** The Space needs the repo *root*, not just `app/` —
   `app/analyzer.py` imports `quorum/` and `predict.py` from the parent.

   ```bash
   git remote add space https://huggingface.co/spaces/techjam2026blueberryjam/quorum
   git push space main
   ```

3. **Add the Space frontmatter.** HF reads config from the YAML block at the top
   of `README.md`. Adding it to this repo's README would put a YAML table on the
   GitHub landing page, so instead set it on the Space only — edit `README.md`
   in the HF web UI after the first push and prepend:

   ```yaml
   ---
   title: Quorum
   emoji: 🔍
   colorFrom: indigo
   colorTo: gray
   sdk: docker
   app_port: 7860
   pinned: false
   ---
   ```

   Do not pull that commit back into GitHub.

## Three things this Dockerfile got wrong, so they are not reintroduced

All three were found by building it, and all three passed a green build.

1. **torch and torchvision from different indexes.** `open_clip_torch` depends
   on torchvision, so installing `torch` alone from the CPU index lets pip
   satisfy that with a PyPI torchvision built against a *different* torch. It
   imports far enough to look fine, then dies on
   `RuntimeError: operator torchvision::nms does not exist` — so `open_clip`
   never loads and the demo is dead. Both now come from the same index in the
   same command, with an import check appended.
2. **`(clip_bake && ocr_bake) || echo` swallowed a CLIP failure.** The build
   exited 0 and shipped an image with no CLIP weights in it, which is precisely
   what the bake exists to prevent. They are separate `RUN` steps now; the CLIP
   one has no fallback.
3. **`.dockerignore` excluded `data/cache/` and missed `data/.cache/`** —
   dotted, a different path — which put HuggingFace download metadata *listing
   the private cache repo's contents* into a public image. It is an allowlist
   now (`data/**`, then `!data/models/**`). A blocklist has to anticipate every
   path; an allowlist only has to name what we want.

The last two are now **build assertions**, not things anyone checks by eye: the
final `RUN` fails the build unless the safetensors file is in the layer and
`data/` is under 2 MB.

## Verify before you call it done

```bash
docker build -t quorum-space .
docker run --rm -p 7860:7860 quorum-space
# then open http://localhost:7860 and upload test-images/real/real3.jpg
```

Two things to check, both of which have a wrong answer that looks fine:

- **The verdict matches `predict.py`.** Run
  `python predict.py --input-dir test-images --output preds.json` and compare.
  They currently **disagree** on face images — `app/analyzer.py:267` calls
  `score_embeddings` without `face=`, so the demo scores two branches where the
  deliverable scores three. On `fake1.png` that is 0.4588 vs 0.5115, which
  crosses the verdict boundary. Fix that before judging.
- **No dataset files in the image.**
  `docker run --rm quorum-space du -sh /home/user/app/data` should be under
  1 MB. If it is not, `.dockerignore` did not apply and you may be about to
  publish licensed data.

## Runtime expectations, measured

On 2 CPU threads, locally:

```
model load from local disk     9.5 s      once per container, at startup
CLIP forward, one image        3.45 s
resident memory, CPU-only      2.43 GB    peak 2.53 GB
```

A free Space has 16 GB, so memory is not the constraint. Latency is: a request
runs up to **three** CLIP passes — the image, the face crop, and the text tiles —
so budget roughly 10 s per upload, plus an OCR sweep. `--preload` in the
Dockerfile pays the 9.5 s model load at container start rather than on the first
request, so the first visitor does not eat it.

**Verified in the container** (`docker run -p 7860:7860`), not just locally:

```
GET /                          HTTP 200
POST /api/analyze  fake_quorum_poster.png
   verdict uncertain, confidence 0.4165
   general 0.7517  tampered 0.035  face null  regularity 0.477  text 0.7629
POST /api/analyze  fake1.png
   provenance declares_ai=True, C2PA manifest present
```

Image is **5.59 GB** — inside the Space limit, but it is a real cold-pull cost.
Mostly torch + torchvision + the 1.6 GB baked cache.

**Not measured:** cold-start time on Spaces itself — how long HF takes to pull
and start a ~4 GB image. Time it once after the first deploy and record it here.

Two changes would roughly halve the request latency if it matters:

- Batch the three CLIP passes into one `embed_batch` call.
- Make the text branch opt-in per request. It is display-only and costs an OCR
  sweep plus a third CLIP pass.

## Gunicorn settings, and why

One worker, on purpose: each worker loads its own ~2.4 GB copy of CLIP, so a
second doubles memory for no throughput on a 2-vCPU box. Four threads handle
concurrent uploads — torch releases the GIL. `--timeout 300` because gunicorn's
30 s default would kill exactly the requests we care about.
