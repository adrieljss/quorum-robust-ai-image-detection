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
        from quorum.detectors.face import MODEL as FACE_MODEL
        from quorum.detectors.spectral import MODEL as SPECTRAL_MODEL
        from quorum.fusion import MODEL as FUSION_MODEL
        import predict as shipped

        self.shipped = shipped
        self.embedder = Embedder()
        self.probes = shipped.load_probes()
        self.general_w = np.asarray(self.probes[0][0], dtype=np.float64)
        self.general_b = float(self.probes[0][1])
        if len(self.probes) > 1:
            self.tampered_w = np.asarray(self.probes[1][0], dtype=np.float64)
            self.tampered_b = float(self.probes[1][1])
        else:
            self.tampered_w = self.tampered_b = None

        with np.load(FACE_MODEL) as z:
            self.face_w = np.asarray(z["w"], dtype=np.float64).ravel()
            self.face_b = float(z["b"][0])
            self.face_px_mu = float(z["px_mu"])
            self.face_px_sd = float(z["px_sd"])
        if SPECTRAL_MODEL.exists():
            with np.load(SPECTRAL_MODEL) as z:
                self.spectral_w = np.asarray(z["w"], dtype=np.float64).ravel()
                self.spectral_b = float(z["b"][0])
        else:
            self.spectral_w = self.spectral_b = None
        with np.load(FUSION_MODEL) as z:
            self.cal_face = (float(z["cal_face"][0]), float(z["cal_face"][1]))
            self.cal_tampered = (
                float(z["cal_tampered"][0]),
                float(z["cal_tampered"][1]),
            )
            self.cal_spectral = (
                float(z["cal_spectral"][0]),
                float(z["cal_spectral"][1]),
            )


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


def _sigmoid(raw: float) -> float:
    return float(1.0 / (1.0 + np.exp(-raw)))


def _platt(raw: float, calibration: tuple[float, float]) -> float:
    a, b = calibration
    return _sigmoid(a * raw + b)


def _face_bbox(image: Image.Image) -> dict | None:
    """Return the largest detected face box in the normalized frame."""
    from quorum.features import DET_MAX, MIN_FACE, _detector

    bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    height, width = bgr.shape[:2]
    scale = min(1.0, DET_MAX / max(height, width))
    small = cv2.resize(bgr, (int(width * scale), int(height * scale))) if scale < 1.0 else bgr
    _, faces = _detector(small.shape[1], small.shape[0]).detect(small)
    if faces is None or not len(faces):
        return None
    x, y, box_width, box_height = (max(faces, key=lambda face: face[2] * face[3]) / scale)[:4]
    if min(box_width, box_height) < MIN_FACE:
        return None
    return {
        "x": int(round(float(x))),
        "y": int(round(float(y))),
        "width": int(round(float(box_width))),
        "height": int(round(float(box_height))),
    }


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
    """Score one batch with analyzer.py-equivalent preprocessing and scales."""
    from quorum.embed import normalise
    from quorum.features import MIN_FACE, face_crop, spectral_features

    # The shipped probes were trained on normalise()'s JPEG round-trip. Sending
    # raw OpenCV pixels directly into CLIP changes their input distribution.
    normalized = [normalise(image) for image in images]
    face_crops = []
    face_meta = []
    for index, image in enumerate(normalized):
        crop, pixels = face_crop(image)
        if crop is not None and pixels >= MIN_FACE:
            face_crops.append(crop)
            face_meta.append((index, pixels))

    vectors = np.asarray(state.embedder.embed_batch(normalized), dtype=np.float64)
    face_vectors = (
        np.asarray(state.embedder.embed_batch(face_crops), dtype=np.float64)
        if face_crops
        else np.empty((0, vectors.shape[1]), dtype=np.float64)
    )
    face_calibrated = [None] * len(normalized)
    face_shipped = np.zeros(len(normalized), dtype=np.float64)
    for face_vector, (index, pixels) in zip(face_vectors, face_meta):
        size_feature = (np.log2(pixels) - state.face_px_mu) / state.face_px_sd
        face_raw = float(np.concatenate([face_vector, [size_feature]]) @ state.face_w + state.face_b)
        face_calibrated[index] = _platt(face_raw, state.cal_face)
        face_shipped[index] = _sigmoid(face_raw - state.shipped.SHIFT)

    confidences = state.shipped.score_embeddings(vectors, state.probes, face=face_shipped)
    records = []

    for index, (timestamp, vector, confidence) in enumerate(zip(timestamps, vectors, confidences)):
        general_calibrated = _sigmoid(float(vector @ state.general_w + state.general_b))
        tampered_calibrated = None
        if state.tampered_w is not None:
            tampered_raw = float(vector @ state.tampered_w + state.tampered_b)
            tampered_calibrated = _platt(
                tampered_raw / state.shipped.TAMPERED_SCALE,
                state.cal_tampered,
            )
        regularity_calibrated = None
        if state.spectral_w is not None:
            spectral_raw = float(spectral_features(normalized[index]).astype(np.float64) @ state.spectral_w + state.spectral_b)
            regularity_calibrated = _platt(spectral_raw, state.cal_spectral)
        confidence = float(confidence)
        regions = []
        if face_calibrated[index] is not None:
            bbox = _face_bbox(normalized[index])
            if bbox is not None:
                regions.append({
                    "type": "face",
                    "bbox": bbox,
                    "score": round(face_calibrated[index], 4),
                    "verdict": _verdict(face_calibrated[index]),
                })
        records.append({
            "timestamp_seconds": round(float(timestamp), 3),
            "verdict": _verdict(confidence),
            "confidence": round(confidence, 4),
            "provenance": {"c2pa": None, "exif_software": None},
            "signals": {
                "general": round(general_calibrated, 4),
                "tampered": round(tampered_calibrated, 4) if tampered_calibrated is not None else None,
                "face": round(face_calibrated[index], 4) if face_calibrated[index] is not None else None,
                "text": None,
                "regularity": round(regularity_calibrated, 4) if regularity_calibrated is not None else None,
            },
            "content_type": None,
            "explanation": "Frame-level prediction from the shipped image model.",
            "degradation_estimate": None,
            "reliability": None,
            "regions": regions,
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
        # Keep every scored sample in frame_results. The API exposes that
        # complete list when the aggregate verdict is AI, as required by the
        # frontend's timestamp marker and frame-inspection interaction.
        return {
            "result": result,
            "frames": frame_results if result["verdict"] == "likely_ai" else [],
        }
    finally:
        if capture is not None:
            capture.release()
        if temp_path:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass
