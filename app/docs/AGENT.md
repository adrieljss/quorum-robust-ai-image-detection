# Agent notes — Quorum web app (`./app`)

Read this before changing the Flask UI. Full API contract: `FRONTEND-HANDOVER.md` in this folder.

## Mission

`./app` is a **single Flask process** that serves the Quorum detector UI and (later) inference. There is no separate Node server. Do not add a landing page. Do not edit files outside `./app` unless the user explicitly asks.

## Current state

- UI: upload → (stub analyze) → result card / slideshow.
- Demo stub: `DEMO_MODE = True` in `app.py`. Fake JSON is deterministic per filename+size.
- Real models are **not** called. Parent-repo `predict.py` is a different, thinner schema (`image_path` + `pred` only).

## Do / don't

- **Do** keep `POST /api/analyze` with form field `images` and body `{ "results": [ ... ] }`.
- **Do** send `null` for skipped branches (face/text), never `0`.
- **Do** comment new JavaScript in the same style as `static/js/app.js`.
- **Don't** invent a second API or put inference in the browser.
- **Don't** dump every JSON key into the UI. Provenance chips only if non-null.
- **Don't** add TikTok logos or trademarks.

## Where to change what

| Intent | File |
|---|---|
| Wire real ML | `app.py` (`analyze()`, `DEMO_MODE`) |
| Request URL / FormData / rendering | `static/js/app.js` (see `BACKEND CONTRACT`) |
| Markup / stages | `templates/index.html` |
| Visual design | `static/css/styles.css` (`:root` tokens first) |
| Contract for humans | `docs/FRONTEND-HANDOVER.md` |

## UI stages (ids)

`upload-stage` → `loading-stage` → `results-stage` or `error-stage`. `showStage()` in `app.js` toggles `.is-active`.

## Slideshow

Reuse one `#result-card`. `results[]` holds `{ file, objectUrl, data }`. `goTo(i)` wraps. Dots only if `length > 1`.

## Design tokens

Warm palette: cream `#f6efe6`, paper `#fffaf4`, ink `#2c1e14`, orange `#d56a2b`, brown `#3d2b1f`. Type: Fraunces (headings) + Source Sans 3 (UI).

## Verification

From `./app`: `python app.py`, open `http://127.0.0.1:5000`. Test 1 image, several images, reject a `.txt`, then error path (stop server and Analyze).
