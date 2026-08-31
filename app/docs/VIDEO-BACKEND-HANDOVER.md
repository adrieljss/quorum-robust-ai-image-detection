# Quorum video backend handover

`POST /api/analyze-video` accepts the video selected by the `/video` page,
samples it with OpenCV, and sends every sampled frame to the shipped model in
the repository-root `predict.py`.

## Request

Send one `multipart/form-data` part named `video`.

| Constraint | Value |
|---|---|
| Field name | `video` |
| Number of files | exactly one |
| Containers | MP4, WebM, MOV |
| Maximum size | 45 MB in the frontend; 50 MB Flask request cap |

## Sampling policy

| Video duration | Sampling rate |
|---|---|
| up to 1 minute | 1 frame per second |
| over 1 through 2 minutes | 2 frames per second |
| over 2 through 5 minutes | 3 frames per second |
| over 5 minutes | 4 frames per second |

The 3 fps band fills the gap between the explicitly requested 2 fps and 4 fps
rules. Sampled frames are processed in batches of 32, so every selected frame
is sent to `predict.py` without retaining all decoded video frames in memory.

## Success response

The endpoint returns HTTP `200`:

```json
{
  "result": {
    "filename": "clip.mp4",
    "verdict": "likely_ai",
    "confidence": 0.82,
    "explanation": "Sampled 60 frames at 1 fps. 44 sampled frames were classified as likely AI.",
    "signals": {
      "general": 0.79,
      "tampered": 0.82,
      "face": 0.74,
      "text": null,
      "regularity": null
    },
    "duration_seconds": 59.98,
    "sampling_fps": 1,
    "sampled_frame_count": 60
  },
  "frames": [
    {
      "timestamp_seconds": 12.0,
      "verdict": "likely_ai",
      "confidence": 0.88,
      "provenance": { "c2pa": null, "exif_software": null },
      "signals": {
        "general": 0.83,
        "tampered": 0.88,
        "face": null,
        "text": null,
        "regularity": null
      },
      "content_type": null,
      "explanation": "Frame-level prediction from the shipped image model.",
      "degradation_estimate": null,
      "reliability": null,
      "regions": []
    }
  ]
}
```

### Aggregate `result`

The UI uses `result` for its headline. `confidence` is the median confidence
of every sampled frame. `signals.general`, `signals.tampered`, and
`signals.face` are medians over frames where that branch ran. `text` and
`regularity` are `null` because no trained video-frame branch has been wired
for them. The result bucket thresholds are:

- `likely_ai`: confidence ≥ 0.60
- `likely_real`: confidence ≤ 0.40
- `uncertain`: otherwise

### Frame `frames`

Every sampled frame is analyzed. The response includes the detailed `frames`
array only when the aggregate verdict is `likely_ai`; it is an empty array for
`likely_real` and `uncertain`. Each frame contains its video timeline position
in `timestamp_seconds` plus the image-result-compatible fields shown above.

Face regions are emitted when the sampled frame's face branch runs. Each uses
the standard `{type, bbox, score, verdict}` shape, with pixel coordinates in
the decoded frame. Text regions are currently omitted because the video path
does not run OCR on every sampled frame.

## Errors

Errors use a 4xx/5xx status and this object:

```json
{ "error": "human-readable message" }
```

The endpoint returns a 400 for missing/multiple uploads, unsupported
extensions, empty files, and unreadable videos. Model or unexpected server
failures return 500.
