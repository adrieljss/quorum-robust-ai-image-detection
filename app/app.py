"""
Quorum web app — Flask entry point.

Run from this directory (./app):
    python app.py

Real inference is wired in via analyzer.py -- demo mode is off, and
analyze() calls straight into the real pipeline.
"""

from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, render_template, request

import analyzer
import video_analyzer

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
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov"}

# Real analyzer is always used -- no demo/fake mode.
DEMO_MODE = False


@app.route("/")
def index():
    """Serve the single-page upload / results interface. No landing page."""
    return render_template("index.html", demo_mode=DEMO_MODE)


@app.route("/video")
def video():
    """Serve the single-video analysis interface."""
    return render_template("video.html", demo_mode=DEMO_MODE)


@app.route("/api/analyze-video", methods=["POST"])
def analyze_video():
    """Analyze one uploaded video through the sampled-frame pipeline."""
    files = request.files.getlist("video")
    if len(files) != 1:
        return jsonify({"error": "Upload exactly one video using form field 'video'."}), 400

    stored_file = files[0]
    filename = stored_file.filename or "untitled"
    if Path(filename).suffix.lower() not in ALLOWED_VIDEO_EXTENSIONS:
        return jsonify({"error": f"Unsupported video type: {filename}"}), 400
    payload = stored_file.read()
    if not payload:
        return jsonify({"error": f"Empty video: {filename}"}), 400

    try:
        response_payload = video_analyzer.analyze_video(payload, filename)
        app.logger.info("Video analysis response for %s: %s", filename, response_payload)
        return jsonify(response_payload)
    except ValueError as exc:
        return jsonify({"error": f"Could not analyze {filename}: {exc}"}), 400
    except Exception as exc:
        app.logger.exception("video analysis failed for %s", filename)
        return jsonify({"error": f"Could not analyze {filename}: {exc}"}), 500


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Analyze one or more uploaded images.

    Contract (keep this stable):
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

        payload = stored_file.read()
        if not payload:
            return jsonify({"error": f"Empty file: {filename}"}), 400

        # Wrapped in try/except so a bad upload or a model failure returns a
        # clean {"error": ...} instead of a 500 with no body.
        try:
            result = analyzer.analyze_image(payload, filename=filename)
        except Exception as exc:
            app.logger.exception("analyze_image failed for %s", filename)
            return jsonify({"error": f"Could not analyze {filename}: {exc}"}), 500

        result["filename"] = filename
        results.append(result)

    return jsonify({"results": results})


if __name__ == "__main__":
    # debug=True's reloader spawns a second process that re-imports this file,
    # so analyzer.py loads its models lazily (on first request) rather than at
    # import time -- otherwise the ~1.7GB CLIP weights would load twice. If you
    # turn debug off, you can optionally call analyzer.warm_up() here to pay
    # the load cost at startup instead of on the first request.
    # 0.0.0.0 is useful if you test from another device on the LAN.
    app.run(host="127.0.0.1", port=5000, debug=True)
