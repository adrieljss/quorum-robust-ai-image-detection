"""
Quorum real inference backend for the Flask demo.

Wires the trained probes (../data/models/*.npz) and the CLIP embedder
(../quorum/embed.py) into the schema in docs/FRONTEND-HANDOVER.md section 2,
via the seam that doc describes: analyze_image(payload, filename=filename).

Notes worth keeping in mind:

  - confidence/verdict reuse predict.py's score_embeddings() (max(general,
    tampered), shifted) instead of quorum/fusion.py's combiner. Both
    predict.py's own docstring and HANDOVER.md measure fusion losing to
    max() on So-Fake-OOD -- don't swap this back without re-checking those
    numbers. fusion.npz is still used here, just for its Platt calibration,
    not its verdict.
  - signals.text and degradation_estimate are always None -- no trained,
    saved model behind either (text branch has two experimental attempts in
    quorum/detectors/text.py, neither with a saved .npz; degradation head
    never persisted). A number here would be fabricated, not measured.
  - provenance IS built now (quorum/provenance.py) and reads C2PA, EXIF, XMP
    and PNG text chunks off the ORIGINAL upload bytes. It never touches
    confidence: it cannot be measured on any eval set we have, because
    normalise() strips exactly what it reads. Read that module's docstring
    before wiring declares_ai into a score.
  - signals.regularity IS reported, from data/models/spectral.npz, and is
    DISPLAY ONLY. The branch is near-chance -- 0.6736 clean AUC, collapsing
    to 0.5471 under noise -- and loses in every combination tested, under
    max() and under a learned combiner both (see that module's docstring).
    It is shown because a weak signal labelled as weak is context for a
    verdict; it is not in `confidence` and must not be. It is deliberately
    absent from st.probes, which is what score_embeddings() maxes over.
  - regions: at most one entry, type "face". Same rule as above -- no
    trained probe, no entry, so an empty list means "nothing scorable
    found," not "nothing was checked." Only the single largest face is
    scored, matching how face.npz was trained (one face per image, not an
    ensemble). tampered does not get a region entry: it scores the whole
    image (same CLIP embedding as general, just a different linear probe on
    top), not a specific crop, so it has nothing to draw a box around.
  - _face_bbox() re-runs face detection instead of editing
    quorum/features.py to have face_crop() return the box it already
    computes internally. That file is shared with the training pipeline, so
    re-detecting locally (a few extra ms) keeps it untouched. bbox is in
    pixel coords of the image as uploaded -- normalise() only re-encodes
    the JPEG, it does not resize.
  - Models load lazily, on first request, not at import. Flask's debug
    reloader re-executes this module in a supervisor process that never
    serves a request; eager loading would load the ~1.7GB CLIP weights
    twice.
"""

from __future__ import annotations

import io
import sys
import threading
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

# app.py runs from ./app; make the parent repo (quorum/, predict.py, data/)
# importable regardless of cwd.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Demo UX layer on top of the shipped score: how `confidence` gets bucketed
# into likely_ai / likely_real / uncertain. The score itself is untouched.
AI_THRESHOLD = 0.60
REAL_THRESHOLD = 0.40
MIN_FACE_PX = 64  # kept in sync with quorum/features.py:MIN_FACE

_lock = threading.Lock()
_state: Optional["_State"] = None


class _State:
    """Everything loaded once: the CLIP embedder plus every trained probe."""

    def __init__(self):
        from quorum.embed import Embedder
        from quorum.detectors.face import MODEL as FACE_MODEL
        from quorum.detectors.spectral import MODEL as SPECTRAL_MODEL
        from quorum.fusion import MODEL as FUSION_MODEL
        import predict as shipped

        self.shipped = shipped
        self.embedder = Embedder()
        self.probes = shipped.load_probes()  # [(w_general, b), (w_tampered, b)]
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

        # DISPLAY ONLY, and it is not in self.probes for that reason -- probes
        # is what score_embeddings() maxes over. 8 FFT features, not the 768-d
        # embedding, so it could not ride that matmul even if it were wanted.
        try:
            with np.load(SPECTRAL_MODEL) as z:
                self.spectral_w = np.asarray(z["w"], dtype=np.float64).ravel()
                self.spectral_b = float(z["b"][0])
        except FileNotFoundError:
            # `python -m quorum.detectors.spectral --save` was never run. The
            # signal goes back to null rather than the demo failing to start.
            self.spectral_w = self.spectral_b = None

        with np.load(FUSION_MODEL) as z:  # calibration only, not fusion's verdict
            self.cal_face = (float(z["cal_face"][0]), float(z["cal_face"][1]))
            self.cal_tampered = (float(z["cal_tampered"][0]), float(z["cal_tampered"][1]))
            self.cal_spectral = (float(z["cal_spectral"][0]), float(z["cal_spectral"][1]))
            # cal_general is intentionally NOT loaded. The current general.npz
            # update, that probe's Platt scaling is folded directly into its
            # saved weights (quorum/detectors/general.py:calibrate(), confirmed
            # in predict.py's own docstring), so general_raw is already a
            # calibrated logit. fusion.npz's cal_general is a leftover pair
            # fitted against the OLD (pre-fold) general probe; applying it here
            # would double-calibrate and produce a wrong signals.general value.


def _get_state() -> _State:
    """Thread-safe lazy singleton. First request pays the load cost."""
    global _state
    if _state is None:
        with _lock:
            if _state is None:
                _state = _State()
    return _state


def warm_up() -> None:
    """Optional: load models at process start instead of on first request."""
    _get_state()


def _platt(raw: float, ab: Tuple[float, float]) -> float:
    a, b = ab
    return float(1.0 / (1.0 + np.exp(-(a * raw + b))))


def _sigmoid(raw: float) -> float:
    """Plain sigmoid, for a raw score that is already a calibrated logit
    (currently: general_raw, since general.npz's Platt fit is folded into its
    weights). Use _platt() instead for a branch whose raw decision value still
    needs its own (a, b) applied."""
    return float(1.0 / (1.0 + np.exp(-raw)))


def _verdict_from_score(score: float) -> str:
    """Same bucketing as the headline verdict, for one branch's score."""
    if score >= AI_THRESHOLD:
        return "likely_ai"
    if score <= REAL_THRESHOLD:
        return "likely_real"
    return "uncertain"


def _face_bbox(img: Image.Image) -> Optional[dict]:
    """{"x", "y", "width", "height"} in native pixel coords, or None.

    Re-runs face_crop()'s detection step (same model/rules) -- see module
    docstring for why this isn't just added to quorum/features.py instead.
    """
    import cv2
    from quorum.features import DET_MAX, MIN_FACE, _detector

    bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    k = min(1.0, DET_MAX / max(h, w))
    small = cv2.resize(bgr, (int(w * k), int(h * k))) if k < 1.0 else bgr
    _, faces = _detector(small.shape[1], small.shape[0]).detect(small)
    if faces is None or not len(faces):
        return None
    f = max(faces, key=lambda r: r[2] * r[3]) / k  # back to native coords
    x, y, bw, bh = f[:4]
    if min(bw, bh) < MIN_FACE:
        return None
    return {
        "x": int(round(float(x))),
        "y": int(round(float(y))),
        "width": int(round(float(bw))),
        "height": int(round(float(bh))),
    }


def _content_type(general_vec: np.ndarray) -> str:
    """CLIP zero-shot label, reusing the embedding already computed."""
    from quorum.fusion import CONTENT, content_onehot
    onehot = content_onehot(general_vec[None, :].astype(np.float32))
    return CONTENT[int(np.argmax(onehot[0]))]


def _explanation(verdict: str, general_cal: float, tampered_cal: Optional[float],
                  face_cal: Optional[float], content_type: str) -> str:
    """Short templated rationale, not a learned explainability model --
    names whichever branch drove the number."""
    branches = [("general synthesis probe", general_cal)]
    if tampered_cal is not None:
        branches.append(("localized-edit probe", tampered_cal))
    if face_cal is not None:
        branches.append(("face probe", face_cal))

    if verdict == "likely_ai":
        label, score = max(branches, key=lambda kv: kv[1])
        return (f"The {label} carried the strongest signal "
                f"({score:.0%}) on this {content_type} image.")
    if verdict == "likely_real":
        label, score = min(branches, key=lambda kv: kv[1])
        return (f"No branch found strong artifacts; the {label} was most "
                f"confident this is authentic ({1 - score:.0%}).")
    return (f"Branches landed close to the decision boundary on this "
            f"{content_type} image -- treat this as low-confidence.")


def _reliability(confidence: float, face_present: bool,
                  general_cal: float, face_cal: Optional[float]) -> str:
    """Heuristic proxy, not a trained reliability head (none exists yet):
    decisiveness (distance from 0.5) plus cross-branch agreement."""
    decisiveness = abs(confidence - 0.5) * 2.0
    agreement = (1.0 - abs(general_cal - face_cal)) if (face_present and face_cal is not None) else 1.0
    score = 0.6 * decisiveness + 0.4 * agreement
    if score > 0.66:
        return "high"
    if score > 0.40:
        return "medium"
    return "low"


def analyze_image(payload: bytes, filename: str = "") -> dict:
    """bytes -> the per-image dict documented in FRONTEND-HANDOVER.md section 2."""
    st = _get_state()

    from quorum.embed import normalise
    from quorum.features import face_crop

    try:
        raw_img = Image.open(io.BytesIO(payload)).convert("RGB")
        raw_img.load()
    except Exception:
        raise ValueError(f"not a valid image file: {filename}" if filename else "not a valid image file")

    from quorum import provenance as prov
    # On the ORIGINAL bytes: normalise() below re-encodes and strips
    # every field this reads.
    prov_report = prov.inspect(payload)

    img = normalise(raw_img)  # same JPEG q95 round-trip every probe trained on
    face_img, face_px = face_crop(img)
    face_present = face_img is not None and face_px >= MIN_FACE_PX

    batch = [img] + ([face_img] if face_present else [])
    vecs = st.embedder.embed_batch(batch)  # one CLIP pass for both branches
    general_vec = np.asarray(vecs[0], dtype=np.float64)
    face_vec = np.asarray(vecs[1], dtype=np.float64) if face_present else None

    general_raw = float(general_vec @ st.general_w + st.general_b)
    tampered_raw = (float(general_vec @ st.tampered_w + st.tampered_b)
                     if st.tampered_w is not None else None)

    # THE shipped score -- see module docstring, do not recompute by hand.
    confidence = float(st.shipped.score_embeddings(general_vec[None, :], st.probes)[0])
    verdict = _verdict_from_score(confidence)

    general_cal = _sigmoid(general_raw)
    tampered_cal = _platt(tampered_raw, st.cal_tampered) if tampered_raw is not None else None

    face_cal = None
    regions = []
    if face_present:
        z = (np.log2(face_px) - st.face_px_mu) / st.face_px_sd
        design = np.concatenate([face_vec, [z]])
        face_raw = float(design @ st.face_w + st.face_b)
        face_cal = _platt(face_raw, st.cal_face)

        bbox = _face_bbox(img)
        if bbox is not None:
            regions.append({
                "type": "face",
                "bbox": bbox,
                "score": round(face_cal, 4),
                "verdict": _verdict_from_score(face_cal),
            })
        # else: face_crop() found a face but re-detection above didn't --
        # shouldn't happen, but fail closed (no region) rather than guess a box.

    # Reads PIXELS, not the embedding: a centre crop at NATIVE resolution, since
    # a resize destroys the high frequencies this branch exists to look at.
    # Display only -- `confidence` above is already final and cannot see this.
    spectral_cal = None
    if st.spectral_w is not None:
        from quorum.features import spectral_features
        sf = np.asarray(spectral_features(img), dtype=np.float64)
        if np.abs(sf).sum() > 0:   # 25 rows in spec_so_fake_ood come out all-zero
            spectral_cal = _platt(float(sf @ st.spectral_w + st.spectral_b),
                                  st.cal_spectral)

    content_type = _content_type(general_vec)
    reliability = _reliability(confidence, face_present, general_cal, face_cal)
    explanation = _explanation(verdict, general_cal, tampered_cal, face_cal, content_type)

    return {
        "verdict": verdict,
        "confidence": round(confidence, 4),
        "provenance": {
            # Contract keys (FRONTEND-HANDOVER section 2): string or null, and
            # the UI shows a chip only when non-null.
            "c2pa": prov_report["c2pa"],
            "exif_software": prov_report["exif_software"],
            # Extra, ignored by the current UI and one line from being shown.
            # declares_ai is deliberately NOT folded into confidence -- see the
            # three reasons in quorum/provenance.py's docstring.
            "declares_ai": prov_report["declares_ai"],
            "summary": prov_report["summary"],
            "fields": prov_report["fields"],
        },
        "signals": {
            "general": round(general_cal, 4),
            "face": round(face_cal, 4) if face_cal is not None else None,
            "tampered": round(tampered_cal, 4) if tampered_cal is not None else None,
            "text": None,        # branch cut -- see module docstring
            # Near-chance on its own (0.6736 clean, 0.5471 under noise) and NOT
            # in the score -- label it accordingly in the UI, it is context for
            # a verdict rather than a second opinion on one.
            "regularity": round(spectral_cal, 4) if spectral_cal is not None else None,
        },
        "content_type": content_type,
        "explanation": explanation,
        "reliability": reliability,
        "regions": regions,
    }
 
 