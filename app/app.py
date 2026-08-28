"""
Quorum web app — Flask entry point.

Run from this directory (./app):
    python app.py

This file serves the frontend and currently includes a DEMO STUB for
POST /api/analyze. Real inference must replace the stub only — see
docs/FRONTEND-HANDOVER.md.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

from flask import Flask, jsonify, render_template, request

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
# Templates live in ./templates, static files in ./static.
# Do not change these folder names unless you also update Flask config.
app = Flask(
    __name__,
    static_folder="static",
    template_folder="templates",
)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB total upload cap

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

# Flip this to False once a real analyzer is wired in (see handover doc).
DEMO_MODE = True


@app.route("/")
def index():
    """Serve the single-page upload / results interface. No landing page."""
    return render_template("index.html", demo_mode=DEMO_MODE)


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Analyze one or more uploaded images.

    Contract (keep this stable when replacing the stub):
      Request: multipart/form-data
        field name: "images"  (repeat for each file)
      Response: JSON object
        {
          "results": [ { ...per-image payload... }, ... ]
        }
      Per-image payload matches the schema in docs/FRONTEND-HANDOVER.md.

    The frontend also echoes each file's original filename so results can
    be matched in upload order (index 0 of `results` is the first file).
    """
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No images uploaded. Use form field name 'images'."}), 400

    results = []
    for stored_file in files:
        filename = stored_file.filename or "untitled"
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            return jsonify({"error": f"Unsupported file type: {filename}"}), 400

        # Read bytes so a future real backend can pass them to the model.
        # The demo stub does not persist uploads to disk.
        payload = stored_file.read()
        if not payload:
            return jsonify({"error": f"Empty file: {filename}"}), 400

        if DEMO_MODE:
            result = _demo_result(filename, payload)
        else:
            # BACKEND HOOK: replace this branch with the real pipeline call.
            # Keep the same dict shape the frontend already renders.
            return jsonify({"error": "Real analyzer is not connected yet."}), 501

        result["filename"] = filename
        results.append(result)

    return jsonify({"results": results})


def _demo_result(filename: str, payload: bytes) -> dict:
    """
    Deterministic fake result so the same file looks consistent across reloads.

    Seeded from filename + size, not pixel content — this is UI demo data only.
    Delete or stop calling this function when the real backend lands.
    """
    seed = int(hashlib.sha256(f"{filename}:{len(payload)}".encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)

    verdicts = ("likely_ai", "likely_real", "uncertain")
    verdict = rng.choices(verdicts, weights=(0.42, 0.42, 0.16), k=1)[0]

    if verdict == "likely_ai":
        confidence = rng.uniform(0.72, 0.96)
        reliability = rng.choice(["high", "medium", "medium"])
    elif verdict == "likely_real":
        confidence = rng.uniform(0.68, 0.94)
        reliability = rng.choice(["high", "medium"])
    else:
        confidence = rng.uniform(0.48, 0.64)
        reliability = rng.choice(["medium", "low"])

    general = _clamp(confidence + rng.uniform(-0.08, 0.08))
    face = _clamp(rng.uniform(0.25, 0.95) if rng.random() > 0.22 else None)
    text = _clamp(rng.uniform(0.20, 0.88) if rng.random() > 0.35 else None)
    regularity = _clamp(rng.uniform(0.18, 0.78))

    content_type = rng.choice(
        ["portrait", "scene", "object", "animal", "text-heavy"]
    )
    degradation = rng.choice(
        ["clean", "light_jpeg", "heavy_jpeg", "blur", "resize", "noise"]
    )

    explanations = {
        "likely_ai": [
            "Catchlights in the eyes are inconsistent; background signage is unresolved.",
            "Repeating texture in hair and fabric, with over-smooth skin gradients.",
            "Lettering on distant signs collapses into plausible-looking noise.",
        ],
        "likely_real": [
            "Sensor noise and optical falloff match a camera capture pipeline.",
            "Irregular specular highlights and natural depth-of-field variation.",
            "No periodic upsampling grid; edges follow optical, not generative, blur.",
        ],
        "uncertain": [
            "Signals disagree after compression; treat this as a low-confidence call.",
            "Heavy recompression washed out the cues the branches rely on.",
            "Face and general probes pulled in opposite directions.",
        ],
    }

    exif_options = [None, "Stable Diffusion", "Midjourney", "Adobe Photoshop", None]
    c2pa_options = [None, None, None, "c2pa-claim-present"]

    return {
        "verdict": verdict,
        "confidence": round(confidence, 2),
        "provenance": {
            "c2pa": rng.choice(c2pa_options),
            "exif_software": rng.choice(exif_options),
        },
        "signals": {
            "general": round(general, 2),
            "face": None if face is None else round(face, 2),
            "text": None if text is None else round(text, 2),
            "regularity": round(regularity, 2),
        },
        "content_type": content_type,
        "explanation": rng.choice(explanations[verdict]),
        "degradation_estimate": degradation,
        "reliability": reliability,
    }


def _clamp(value: float | None) -> float | None:
    if value is None:
        return None
    return max(0.0, min(1.0, value))


if __name__ == "__main__":
    # 0.0.0.0 is useful if you test from another device on the LAN.
    app.run(host="127.0.0.1", port=5000, debug=True)
