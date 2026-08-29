# Quorum backend handover — `regions` field

This document describes one addition to the API contract defined in `FRONTEND-HANDOVER.md`. It does not replace that document; read them together. Everything in `FRONTEND-HANDOVER.md` section 2 is unchanged. This adds a new field to the same response object.

The backend (`app/analyzer.py`) already returns `regions` on every response. `static/js/app.js` does not read it yet, since `FRONTEND-HANDOVER.md` (top of file) states that response-shape changes require a corresponding update there. Sections 2–4 below describe what that update needs to do.

---

## 1. Background

The response already reports whether the face branch fired (`signals.face`), but not where in the image the detected face is. `regions` adds that position, plus a self-contained score and verdict per region, so the UI can draw a labeled box over the relevant part of the uploaded image.

---

## 2. What's new

Every response now includes a `regions` key: an array, always present, of zero or more objects.

### Example — face detected

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
| `type` | string | Detection branch this region came from. Currently only `"face"` occurs. |
| `bbox.x` | number | Left edge, in pixels from the image's left edge. |
| `bbox.y` | number | Top edge, in pixels from the image's top edge. |
| `bbox.width` | number | Box width in pixels. |
| `bbox.height` | number | Box height in pixels. |
| `score` | number `0–1` | Same calibrated P(AI-generated) value as the matching `signals.*` field. |
| `verdict` | `"likely_ai"` \| `"likely_real"` \| `"uncertain"` | Same bucketing as the top-level `verdict`, applied to this region's own score. |

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

- **At most one region is returned, and only `type: "face"`.** The underlying model (`data/models/face.npz`) was trained on a single face per image rather than an ensemble, so only the largest detected face is scored. A photo with multiple people still produces at most one region.
- **No entry is included for branches without a trained model** (for example text/OCR). This follows the same rule as `signals.*` in `FRONTEND-HANDOVER.md`: a branch that did not run is represented by its absence, not by a placeholder or a null-valued entry. An empty `regions` array means no branch with a trained model found anything to report, not that nothing was checked.
- This scope can change without a change to the response shape: additional `type` values can be added as new entries in the same array as detection models become available.

---

## 5. Rendering guidance

To keep future region types (for example a future `"text"` type) from requiring frontend changes, render `regions` as a generic loop over its contents rather than branching on `type`:

```js
regions.forEach(region => {
  // draw a box using region.bbox (scaled per section 3)
  // label it with region.type, region.score, region.verdict
});
```

A suggested per-region label: `` `${region.type} · ${Math.round(region.score * 100)}% · ${region.verdict}` `` (for example, "face · 91% · likely_ai"). Styling is left to whatever fits the rest of the UI.
