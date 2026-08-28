# Quorum frontend handover

This document is for the person wiring the real analyzer into the Flask app in `./app`. The UI is already built against a stable HTTP contract. Do not change request or response shapes unless you also update `static/js/app.js`.

The web app is self-contained in this directory. The rest of the repository (models, `predict.py`, `quorum/`) is out of scope for the frontend.

---

## 1. Setup

All commands assume the current working directory is **`app/`** (this folder). Do not run the server from the repository root.

### Python environment

```powershell
cd app
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

On macOS / Linux:

```bash
cd app
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000). There is no landing page; the upload screen is `/`.

### Layout

```
app/
  app.py                 Flask server + demo stub for POST /api/analyze
  requirements.txt       Flask only
  templates/index.html   Single-page UI
  static/css/styles.css
  static/js/app.js       Client: upload, fetch, slideshow, rendering
  static/favicon.svg
  docs/                  This handover + agent notes
```

### Demo mode

`DEMO_MODE = True` in `app.py`. While that flag is true, `/api/analyze` returns **deterministic fake results** (seeded from filename + byte length). Nothing is written to disk. A “Demo preview” pill appears in the header.

---

## 2. API contract

The frontend talks to **one** endpoint.

| | |
|---|---|
| Method | `POST` |
| URL | `/api/analyze` |
| Encoding | `multipart/form-data` |
| File field | **`images`** (repeat the same field name for every file) |
| Order | Result `i` must correspond to uploaded file `i` |

### Request (FormData)

The browser builds:

```
POST /api/analyze
Content-Type: multipart/form-data; boundary=…

images: <file 0>
images: <file 1>
…
```

Notes:

- Field name **must** be `images`. `request.files.getlist("images")` is what Flask uses.
- Do not require extra text fields. `filename` is taken from the uploaded `FileStorage`.
- Accepted extensions today: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.gif`.
- Frontend also rejects files over 12 MB each, max 24 files. Flask `MAX_CONTENT_LENGTH` is 50 MB for the whole request.
- The client **must not** set `Content-Type` manually; the browser supplies the multipart boundary.

### Success response (`200`)

```json
{
  "results": [
    {
      "filename": "portrait.jpg",
      "verdict": "likely_ai",
      "confidence": 0.87,
      "provenance": {
        "c2pa": null,
        "exif_software": "Stable Diffusion"
      },
      "signals": {
        "general": 0.91,
        "face": 0.83,
        "text": 0.62,
        "regularity": 0.40
      },
      "content_type": "portrait",
      "explanation": "Catchlights in the eyes are inconsistent; background signage is unresolved.",
      "degradation_estimate": "heavy_jpeg",
      "reliability": "medium"
    }
  ]
}
```

`results` **must** be an array. Length **must** equal the number of uploaded files. A single image is still `{ "results": [ {…} ] }`, never a bare object.

### Per-image fields

| Field | Type | Required | Shown in UI? |
|---|---|---|---|
| `filename` | string | recommended | yes (overlay) |
| `verdict` | `"likely_ai"` \| `"likely_real"` \| `"uncertain"` | **yes** | yes (headline) |
| `confidence` | number `0–1` | **yes** | yes (ring) |
| `signals.general` | number `0–1` or `null` | **yes** | yes (bar) |
| `signals.face` | number `0–1` or `null` | **yes** | yes (bar) |
| `signals.text` | number `0–1` or `null` | **yes** | yes (bar) |
| `signals.regularity` | number `0–1` or `null` | **yes** | yes (bar) |
| `explanation` | string | recommended | yes (paragraph) |
| `reliability` | `"high"` \| `"medium"` \| `"low"` | recommended | chip |
| `content_type` | string | optional | chip |
| `degradation_estimate` | string | optional | chip |
| `provenance.c2pa` | string or `null` | optional | chip **only if non-null** |
| `provenance.exif_software` | string or `null` | optional | chip **only if non-null** |

**Missing-branch rule (do not break this):** if a model did not run (no face, no text, etc.), send `null`, **not** `0.0`. The UI labels that bar `n/a` / “not measured”. Zero would look like “the model says real.”

Signal values are treated as **P(AI-generated)** in `[0, 1]`.

Known `degradation_estimate` values the UI pretty-prints: `clean`, `light_jpeg`, `heavy_jpeg`, `blur`, `resize`, `noise`. Unknown strings are shown as-is.

### Error response

```json
{ "error": "human-readable message" }
```

Use a 4xx/5xx status. The frontend displays `error` on the error stage and offers retry (same files, same POST).

---

## 3. How to connect the real backend

You do **not** need a second server. Replace the stub inside this Flask app.

### Step A — keep serving the UI

Leave these routes alone:

- `GET /` → `templates/index.html`
- static files under `static/`

### Step B — turn off the demo

In `app.py`:

1. Set `DEMO_MODE = False` (search for `DEMO_MODE`).
2. Stop calling `_demo_result()`. You can delete `_demo_result` and `_clamp` once unused.
3. In `analyze()`, replace the `if DEMO_MODE: … else: 501` block with a call into the real pipeline.

Suggested shape (do not copy blindly — this is the seam):

```python
# Inside analyze(), after you have `payload` bytes and `filename`:
from wherever import analyze_image  # your team's function

result = analyze_image(payload, filename=filename)
# result must be the per-image dict in section 2
result["filename"] = filename
results.append(result)
```

The frontend already sends `FormData`; you do not edit `app.js` if the JSON matches section 2.

### Step C — what to import from the ML repo

The models live **above** this folder (`predict.py`, `quorum/`). `predict.py` currently emits only `{image_path, pred}`. The **demo schema is richer**. Map fusion output onto the dict in section 2.

Typical mapping:

| UI field | Likely source |
|---|---|
| `signals.general` | calibrated general probe |
| `signals.face` | calibrated face probe, or `null` if `face_present` is false |
| `signals.text` | calibrated text probe, or `null` if no text |
| `signals.regularity` | spectral / regularity scorer |
| `confidence` | fused P(AI) used for the headline |
| `verdict` | threshold on `confidence` (or an explicit fusion class) |
| `reliability` | your reliability head / agreement heuristic |
| `content_type` | CLIP zero-shot label |
| `degradation_estimate` | degradation head (may be a probability today; map to a label if you can) |
| `explanation` | optional; a short string is enough. Empty string is fine until you have one |
| `provenance` | `c2pa` / EXIF if you implement them; otherwise both `null` |

Load models **once at process start**, not per request (see `scripts/try_face.py` in the parent repo).

### Step D — files you actually edit

| File | Why |
|---|---|
| `app.py` | Replace stub. Keep `GET /` and the `/api/analyze` URL. |
| `requirements.txt` | Add ML / image deps the analyzer needs. |
| `static/js/app.js` | **Only if** the URL, field name `images`, or JSON keys change. Search `BACKEND CONTRACT`. |
| `templates/index.html` | No change required for a real model. The demo pill is driven by `demo_mode`. |

You should **not** need to rewrite CSS for the connection.

### Step E — optional JS knobs

In `static/js/app.js` at the top:

- `ANALYZE_URL` — change if you nest the app under a prefix.
- `VERDICT_LABELS` / `DEGRADATION_LABELS` — add enums rather than inventing new JSON keys.
- `SIGNAL_META` — add a fifth bar only if the API adds a fifth signal key.

### Step F — checklist before calling it done

- [ ] `DEMO_MODE` is `False` and `_demo_result` is unused.
- [ ] One file and many files both return `{ "results": [ ... ] }` with matching length.
- [ ] Null branches render as `n/a`, not 0%.
- [ ] A 400/500 with `{ "error": "..." }` shows the error stage.
- [ ] Header no longer shows “Demo preview”.
- [ ] No TikTok branding (already none in this UI).

---

## 4. Frontend behavior the backend can rely on

- Upload stage is the home screen. After results, “New batch” returns there and revokes object URLs.
- Multiple images → one card, slideshow (arrows, dots, left/right keys, swipe).
- Single image → same card, nav hidden.
- Retry on error re-POSTs the same `selectedFiles`.
- Previews are local `blob:` URLs; the server never has to return image bytes.

If something looks wrong in the UI after you connect models, inspect the Network tab for `/api/analyze` first. Nine times out of ten it is a missing `results` array, a length mismatch, or `signals.face: 0` instead of `null`.
