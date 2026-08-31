# Quorum backend handover — response additions

This document describes additions to the API contract defined in `FRONTEND-HANDOVER.md`. It does not replace that document; read them together.

Sections 2–5 describe the `regions` field. The `face`-only version of this was already folded into `FRONTEND-HANDOVER.md`'s own example response — but these sections have since been updated again to add a `text` region type (section 4), which has NOT been folded in yet. Do not skip 2–5 as "already handled"; re-check section 4 specifically. Section 6 describes `signals.tampered`, section 7 describes new `provenance` fields, and section 8 describes `signals.regularity` now carrying real values — none of the three has been folded into `FRONTEND-HANDOVER.md` yet.

---

## 1. Background

The response already reports whether the face branch fired (`signals.face`), but not where in the image the detected face is. `regions` adds that position, plus a self-contained score and verdict per region, so the UI can draw a labeled box over the relevant part of the uploaded image.

---

## 2. What's new

Every response now includes a `regions` key: an array, always present, of zero or more objects.

### Example — face and text both detected

```json
{
  "verdict": "likely_ai",
  "confidence": 0.82,
  "...": "... (all other fields per FRONTEND-HANDOVER.md section 2, unchanged) ...",
  "regions": [
    {
      "type": "face",
      "bbox": { "x": 532, "y": 212, "width": 341, "height": 414 },
      "score": 0.91,
      "verdict": "likely_ai"
    },
    {
      "type": "text",
      "bbox": { "x": 98, "y": 1123, "width": 98, "height": 17 },
      "score": null,
      "verdict": null
    }
  ]
}
```

### Example — nothing detected

```json
{
  "...": "...",
  "regions": []
}
```

### Region fields

| Field | Type | Meaning |
|---|---|---|
| `type` | string | Detection branch this region came from: `"face"` or `"text"`. |
| `bbox.x` | number | Left edge, in pixels from the image's left edge. |
| `bbox.y` | number | Top edge, in pixels from the image's top edge. |
| `bbox.width` | number | Box width in pixels. |
| `bbox.height` | number | Box height in pixels. |
| `score` | number `0–1`, or `null` | Same calibrated P(AI-generated) value as the matching `signals.*` field. `null` for a `"text"` region -- see section 4. |
| `verdict` | `"likely_ai"` \| `"likely_real"` \| `"uncertain"`, or `null` | Same bucketing as the top-level `verdict`, applied to this region's own score. `null` for a `"text"` region -- see section 4. |

---

## 3. Coordinate system

`bbox` coordinates are in pixels, relative to the image **as uploaded** — its `naturalWidth` / `naturalHeight` — not the size it happens to be displayed at in the browser.

If a box is drawn with absolutely-positioned CSS over an `<img>` that has been scaled to fit the page, the box must be scaled by the same ratio:

```js
const img = document.querySelector("#result-image"); // the <img> showing the upload
const scaleX = img.clientWidth / img.naturalWidth;
const scaleY = img.clientHeight / img.naturalHeight;

const boxStyle = {
  left:   region.bbox.x * scaleX,
  top:    region.bbox.y * scaleY,
  width:  region.bbox.width * scaleX,
  height: region.bbox.height * scaleY,
};
```

`naturalWidth` / `naturalHeight` are available on any loaded `<img>` element and do not need to be requested from the backend separately.

---

## 4. Current scope

- **`face`: at most one region.** The underlying model (`data/models/face.npz`) was trained on a single face per image rather than an ensemble, so only the largest detected face is scored. A photo with multiple people still produces at most one `face` region. `score`/`verdict` are always real values here.
- **`text`: up to 10 regions, `score` and `verdict` always `null`.** These come from RapidOCR locating where text sits in the image -- there is no trained model that judges whether detected text is AI-generated, so nothing honest can go in those two fields. This is a deliberate exception to the rule above: the box is real (OCR ran and found something), but it does not carry, or affect, a verdict. Render it as a plain outline / label with no AI-or-real coloring, distinct from how `face` is shown. When more than 10 text regions are found, the 10 with the highest OCR confidence are kept.
- Runs on every image, regardless of `content_type` -- a caption or sign in an otherwise non-text photo is still found, at the cost of OCR being the slowest step in a request.
- **No entry is included for branches without a trained model at all** (there currently are none left in this category, now that `face` and `text` both produce entries). This still follows the same rule as `signals.*` in `FRONTEND-HANDOVER.md`: an empty `regions` array means nothing was found to report, not that nothing was checked.
- This scope can change without a change to the response shape: additional `type` values, or a trained probe giving `text` a real `score`/`verdict`, can arrive as changes to existing entries or new ones in the same array.

---

## 5. Rendering guidance

To keep any future region types from requiring frontend changes, render `regions` as a generic loop over its contents rather than branching on `type` (`text` regions already need `score`/`verdict` handled as possibly-null, per section 4 — do not assume every region has a number to show):

```js
regions.forEach(region => {
  // draw a box using region.bbox (scaled per section 3)
  // label it with region.type, region.score, region.verdict
});
```

A suggested per-region label: `region.score == null ? region.type : \`${region.type} · ${Math.round(region.score * 100)}% · ${region.verdict}\`` — for example "face · 91% · likely_ai" when scored, or just "text" when not (as it always is for `text` regions today). Styling is left to whatever fits the rest of the UI.

---

## 6. New signal: `signals.tampered`

`signals` now includes a fifth key, `tampered`, alongside `general`, `face`, `text`, and `regularity`.

```json
"signals": {
  "general": 0.91,
  "face": 0.83,
  "tampered": 0.22,
  "text": null,
  "regularity": null
}
```

`tampered` follows the same conventions as every other key in `signals`: a number `0`–`1` (this branch's own P(AI-generated), independent of the headline `confidence`), or `null` if the branch did not run. It does not appear in `regions`, since it scores the image as a whole rather than a specific cropped area, so there is no bounding box to draw for it.

`FRONTEND-HANDOVER.md`'s note on `SIGNAL_META` — add a bar only if the API adds a signal key — covers this case directly.

---

## 7. New object: `provenance` metadata fields

`provenance` now carries three keys beyond `c2pa` and `exif_software`.

```json
"provenance": {
  "c2pa": null,
  "exif_software": "Adobe Photoshop 25.0",
  "declares_ai": false,
  "summary": "1 field found; none name an AI tool",
  "fields": [
    { "where": "exif", "key": "Software", "value": "Adobe Photoshop 25.0", "kind": "editor" }
  ]
}
```

| Field | Type | Meaning |
|---|---|---|
| `declares_ai` | boolean | `true` if any metadata field matched a known AI-tool marker (e.g. an EXIF or C2PA field naming a generator). Positive-only, like `exif_software`: absence proves nothing, since re-encoding and most platforms strip this. |
| `summary` | string | One human-readable sentence describing what was found, suitable for direct display. |
| `fields` | array | Every individual metadata field that was read, regardless of `kind`. Each entry: `where` (`"exif"`, `"xmp"`, `"png:<key>"`, or `"c2pa"`), `key` (the field name), `value` (the raw value, display-safe), `kind` (`"ai"`, `"camera"`, `"editor"`, or `null` if unclassified). |

This never affects `confidence` or `verdict` — metadata is evidence about a file's declared origin, not a signal any branch was trained to score, so it cannot be measured the way `signals.*` can (normalisation strips it before any branch sees the image). Treat it as informational only, the same way `exif_software` already was.

---

## 8. `signals.regularity` now has real values

`FRONTEND-HANDOVER.md` already documented `regularity` as `number 0–1, or null` — the shape hasn't changed. What's new is the value: it was always `null` before (`"n/a"` on every image, per that earlier screenshot in the group chat), because no trained model backed it. A spectral probe now does, so this key will show a real percentage on images where the model that produces it is available.

**This value carries no weight in `verdict` or `confidence`, on purpose.** The branch behind it (`quorum/detectors/spectral.py`) is measured, by its own author, to be worse than the shipped scorer in every configuration tested, and its docstring says explicitly that it must not enter the combiner. It is shown for the same reason `general`/`face`/`tampered` are shown — as one more signal a judge can look at — not because it informs the headline number. Do not build any UI logic that treats a high `regularity` value as agreeing or disagreeing with `verdict`; it isn't part of how `verdict` was computed.