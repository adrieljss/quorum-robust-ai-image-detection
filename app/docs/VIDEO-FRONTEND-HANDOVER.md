# Quorum video frontend handover

This document defines the backend contract for the single-video UI at
`GET /video`. The UI is served by the same Flask process as the image app.

## Endpoint

| | |
|---|---|
| Method | `POST` |
| URL | `/api/analyze-video` |
| Encoding | `multipart/form-data` |
| File field | `video` |
| Files per request | Exactly one |

The browser sends the original filename with the file part. It does not send
any other form fields. Do not require the client to set `Content-Type`; the
browser adds the multipart boundary.

The frontend accepts one MP4, WebM, or MOV file, at most 45 MB. The Flask app
currently has a 50 MB total request cap, so the endpoint must remain within it.

## Success response

Return HTTP `200` with exactly one result wrapped in a `result` object:

```json
{
  "result": {
    "filename": "clip.mp4",
    "verdict": "likely_ai",
    "confidence": 0.87,
    "explanation": "The sampled frames show inconsistent motion around the subject's hands.",
    "signals": {
      "visual": 0.91,
      "temporal": 0.84,
      "audio": null
    }
  }
}
```

| Field | Type | Required | UI behavior |
|---|---|---|---|
| `result.filename` | string | recommended | available for future display; the UI uses the local filename now |
| `result.verdict` | `likely_ai`, `likely_real`, or `uncertain` | yes | result headline |
| `result.confidence` | number from `0` to `1` | yes | confidence ring |
| `result.explanation` | string | recommended | short explanation below the verdict |
| `result.signals` | object | yes | one probability bar per key |
| `result.signals.*` | number from `0` to `1` or `null` | yes per emitted branch | model probability or `n/a` |

The `signals` object is intentionally extensible. The frontend renders every
key it receives, so the backend can use names such as `visual`, `temporal`,
`audio`, `face_consistency`, or `tampered`. Values represent that branch's
P(AI-generated). Send `null`, never `0`, when a branch did not run.

## Errors

For any client or analyzer error, return a 4xx or 5xx response with:

```json
{ "error": "human-readable message" }
```

Examples include no video supplied, more than one video supplied, an
unsupported container, a corrupt file, a request larger than the cap, or a
model failure. The UI displays the message and lets the user retry the same
video.

## Backend implementation notes

- Replace the temporary `POST /api/analyze-video` placeholder in `app.py`; do
  not change the existing image endpoint, `/api/analyze`.
- Read the upload with `request.files.get("video")` and reject a missing or
  empty file before invoking the inference pipeline.
- Validate MP4, WebM, and MOV by extension and preferably by container
  inspection. Do not trust the browser MIME type alone.
- Preserve the response wrapper: a video result is `{ "result": { ... } }`,
  not an array and not a bare result object.
- The browser keeps the uploaded video locally for playback. The backend does
  not need to return a video URL, frames, or thumbnails.
