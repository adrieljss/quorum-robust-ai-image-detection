"""Video inference pipeline for the Flask UI.

This module deliberately calls the shipped ``predict.py`` primitives directly,
rather than routing frames through analyzer.py. Frames are decoded with OpenCV,
sampled at a duration-dependent rate, and scored in bounded batches so a long
video does not require every decoded frame to remain in memory at once.
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

AI_THRESHOLD = 0.60
REAL_THRESHOLD = 0.40
BATCH_SIZE = 32

_lock = threading.Lock()
_state = None


class _State:
    """The predictor dependencies, loaded once on the first video request."""

    def __init__(self):
        from quorum.embed import Embedder
        import predict as shipped

        self.shipped = shipped
        self.embedder = Embedder()
        self.probes = shipped.load_probes()
        self.face_model = shipped.load_face()


def _get_state() -> _State:
    global _state
    if _state is None:
        with _lock:
            if _state is None:
                _state = _State()
    return _state


def sampling_rate(duration_seconds: float) -> int:
    """Return frames per second for the requested duration policy.

    The requested 1 / 2 / 4 fps rules leave videos between two and five
    minutes unspecified. Those use 3 fps, preserving the one-fps-per-minute
    progression until the explicit four-fps rule for videos over five minutes.
    """
    if duration_seconds <= 60:
        return 1
    if duration_seconds <= 120:
        return 2
    if duration_seconds <= 300:
        return 3
    return 4


def _verdict(score: float) -> str:
    if score >= AI_THRESHOLD:
        return "likely_ai"
    if score <= REAL_THRESHOLD:
        return "likely_real"
    return "uncertain"


def _frame_indices(frame_count: int, fps: float, sample_fps: int) -> list[int]:
    """Evenly spaced source-frame indexes, including the first frame once."""
    duration = frame_count / fps
    times = np.arange(0, duration, 1.0 / sample_fps)
    indexes = [min(int(round(time * fps)), frame_count - 1) for time in times]
    return list(dict.fromkeys(indexes))


def _signal_median(records: Iterable[dict], key: str):
    values = [record["signals"][key] for record in records if record["signals"].get(key) is not None]
    return round(float(np.median(values)), 4) if values else None


def _score_batch(images: list[Image.Image], timestamps: list[float], state: _State) -> list[dict]:
    """Score one decoded frame batch with the model API in predict.py."""
    vectors = np.asarray(state.embedder.embed_batch(images), dtype=np.float64)
    face_scores = state.shipped.face_score(images, state.embedder, state.face_model)
    confidences = state.shipped.score_embeddings(vectors, state.probes, face=face_scores)
    records = []

    for timestamp, vector, face_score, confidence in zip(timestamps, vectors, face_scores, confidences):
        branch_scores = state.shipped.branch_scores(vector, state.probes)
        face_value = float(face_score) if float(face_score) > 0 else None
        confidence = float(confidence)
        records.append({
            "timestamp_seconds": round(float(timestamp), 3),
            "verdict": _verdict(confidence),
            "confidence": round(confidence, 4),
            "provenance": {"c2pa": None, "exif_software": None},
            "signals": {
                "general": branch_scores.get("general"),
                "tampered": branch_scores.get("tampered"),
                "face": round(face_value, 4) if face_value is not None else None,
                "text": None,
                "regularity": None,
            },
            "content_type": None,
            "explanation": "Frame-level prediction from the shipped image model.",
            "degradation_estimate": None,
            "reliability": None,
            "regions": [],
        })
    return records


def _aggregate_result(frames: list[dict], filename: str, duration_seconds: float, sample_fps: int) -> dict:
    confidences = [frame["confidence"] for frame in frames]
    confidence = round(float(np.median(confidences)), 4)
    verdict = _verdict(confidence)
    likely_ai_frames = sum(frame["verdict"] == "likely_ai" for frame in frames)
    return {
        "filename": filename,
        "verdict": verdict,
        "confidence": confidence,
        "explanation": (
            f"Sampled {len(frames)} frames at {sample_fps} fps. "
            f"{likely_ai_frames} sampled frames were classified as likely AI."
        ),
        "signals": {
            "general": _signal_median(frames, "general"),
            "tampered": _signal_median(frames, "tampered"),
            "face": _signal_median(frames, "face"),
            "text": None,
            "regularity": None,
        },
        "duration_seconds": round(duration_seconds, 3),
        "sampling_fps": sample_fps,
        "sampled_frame_count": len(frames),
    }


def analyze_video(payload: bytes, filename: str) -> dict:
    """Analyze an uploaded video and return the API response payload.

    ``frames`` contains per-frame results only when the aggregate verdict is
    ``likely_ai``. Every frame is still sent through predict.py before the
    aggregate is computed, as required for a reliable video-level decision.
    """
    suffix = Path(filename).suffix.lower() or ".mp4"
    temp_path = None
    capture = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_file.write(payload)
            temp_path = temp_file.name

        capture = cv2.VideoCapture(temp_path)
        if not capture.isOpened():
            raise ValueError("not a readable video file")
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if source_fps <= 0 or frame_count <= 0:
            raise ValueError("video has no readable frame timing information")

        duration_seconds = frame_count / source_fps
        sample_fps = sampling_rate(duration_seconds)
        indexes = _frame_indices(frame_count, source_fps, sample_fps)
        if not indexes:
            raise ValueError("video contains no frames to analyze")

        state = _get_state()
        frame_results = []
        batch_images = []
        batch_timestamps = []
        for index in indexes:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, bgr = capture.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            batch_images.append(Image.fromarray(rgb))
            batch_timestamps.append(index / source_fps)
            if len(batch_images) == BATCH_SIZE:
                frame_results.extend(_score_batch(batch_images, batch_timestamps, state))
                batch_images, batch_timestamps = [], []

        if batch_images:
            frame_results.extend(_score_batch(batch_images, batch_timestamps, state))
        if not frame_results:
            raise ValueError("video frames could not be decoded")

        result = _aggregate_result(frame_results, filename, duration_seconds, sample_fps)
        return {"result": result, "frames": frame_results if result["verdict"] == "likely_ai" else []}
    finally:
        if capture is not None:
            capture.release()
        if temp_path:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass
