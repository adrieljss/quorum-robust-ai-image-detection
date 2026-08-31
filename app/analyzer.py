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
  - signals.regularity IS populated (quorum/detectors/spectral.py's own
    saved probe + fusion.npz's cal_spectral), but display-only, the same as
    text region boxes -- it never touches confidence, verdict, reliability,
    or explanation below. spectral.py's own docstring is explicit that this
    branch must not enter the combiner: 0.6736 clean AUC, collapsing to
    0.5471 under noise, worse than the shipped probe alone in every
    configuration tested. Wiring it into anything that decides the verdict
    would need that measurement to change first, not just a code change.
    Falls back to None if spectral.npz is absent, same as every other
    optional branch.
  - regions: face gets at most one entry (largest detected face, matching
    how face.npz was trained on one face per image, not an ensemble), always
    with a real score/verdict. text gets up to MAX_TEXT_REGIONS entries
    (RapidOCR locations, most-confident first) with score/verdict always
    null -- OCR finds WHERE text is but nothing scores whether it is
    AI-generated, so there is nothing honest to put in those fields. This is
    a deliberate exception to "every region has a score": the request was to
    surface detected text without it implying or affecting a verdict.
    tampered does not get a region entry at all: it scores the whole image
    (same CLIP embedding as general, a different linear probe on top), not a
    specific crop, so there is no box to draw for it.
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
        self.ocr = None  # lazy: only construct RapidOCR if a request needs it
        # load_probes() pre-scales tampered's (w, b) by shipped.TAMPERED_SCALE
        # (currently 1.25) as of the three-branch predict.py update. That scale
        # is exactly right for feeding st.probes back into score_embeddings()
        # below, but it means self.tampered_w/b are NOT on the same scale
        # fusion.npz's cal_tampered was fitted against -- see the unscaling step
        # in analyze_image() where signals.tampered is computed.
        self.probes = shipped.load_probes()  # [(w_general, b), (w_tampered*1.25, b*1.25)]
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

        # Optional: quorum/detectors/spectral.py's own docstring says this
        # branch must never enter the scorer (measured worse than chance in
        # combination -- clean 0.6736, collapsing to 0.5471 under noise), so
        # it is display-only here too, same as text. Absent -> None, same
        # missing-branch handling as everything else.
        if SPECTRAL_MODEL.exists():
            with np.load(SPECTRAL_MODEL) as z:
                self.spectral_w = np.asarray(z["w"], dtype=np.float64).ravel()
                self.spectral_b = float(z["b"][0])
        else:
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


MAX_TEXT_REGIONS = 10  # UI cap on a screenshot/document with dozens of words


def _text_regions(img: Image.Image, st: "_State") -> list:
    """[{"type": "text", "bbox": {...}, "score": None, "verdict": None}, ...]

    Uses RapidOCR only to LOCATE text, never to judge it -- there is no
    saved, trained probe for whether detected text is AI-generated (see
    module docstring), so score/verdict are always null here, deliberately
    outside the usual "regions always have a score" convention. Runs on
    every image regardless of content_type, so a sign or caption in a
    non-"text" photo is still found -- OCR is the slow part of this
    project, and that cost is accepted here on purpose rather than skipped
    based on a guess.

    RapidOCR returns a 4-point quadrilateral per detection (text can be
    rotated); this collapses each to its axis-aligned bounding box to match
    every other region's {x, y, width, height} shape, at the cost of a
    slightly loose box on tilted text.

    The engine itself (st.ocr) is constructed once and cached on _State,
    same as the CLIP embedder -- quorum.detectors.text._ocr() builds a new
    RapidOCR() on every call, which is fine for a one-off training script
    but wasteful across many requests in a long-lived server process.
    """
    if st.ocr is None:
        from rapidocr_onnxruntime import RapidOCR
        st.ocr = RapidOCR()

    res, _ = st.ocr(np.asarray(img.convert("RGB")))
    if not res:
        return []
    # RapidOCR rows are [box(4x2), text, confidence]; keep the most confident
    # detections when there are more than MAX_TEXT_REGIONS.
    res = sorted(res, key=lambda r: float(r[2]), reverse=True)[:MAX_TEXT_REGIONS]
    out = []
    for box, _text, _conf in res:
        pts = np.asarray(box, dtype=np.float64)
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        out.append({
            "type": "text",
            "bbox": {
                "x": int(round(x0)),
                "y": int(round(y0)),
                "width": int(round(x1 - x0)),
                "height": int(round(y1 - y0)),
            },
            "score": None,
            "verdict": None,
        })
    return out


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

    general_cal = _sigmoid(general_raw)
    # tampered_raw comes from st.probes, which load_probes() now pre-scales by
    # shipped.TAMPERED_SCALE. That scale is correct for score_embeddings() below
    # (it uses st.probes directly), but fusion.npz's cal_tampered was fitted on
    # the UNSCALED decision value -- undo the scale before calibrating this
    # display-only copy, or the number shown here reads on the wrong axis.
    tampered_cal = (_platt(tampered_raw / st.shipped.TAMPERED_SCALE, st.cal_tampered)
                     if tampered_raw is not None else None)

    face_cal = None
    face_shipped = 0.0  # predict.face_score()'s own fill for "no face": 0.0, not 0.5
    regions = []
    if face_present:
        z = (np.log2(face_px) - st.face_px_mu) / st.face_px_sd
        design = np.concatenate([face_vec, [z]])
        face_raw = float(design @ st.face_w + st.face_b)
        face_cal = _platt(face_raw, st.cal_face)
        # Same formula as predict.face_score(), reusing the face_vec/face_px
        # already computed above -- calling predict.face_score() fresh here
        # would re-run detection and a second CLIP pass for no reason.
        face_shipped = 1.0 / (1.0 + np.exp(-(face_raw - st.shipped.SHIFT)))

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

    # Runs regardless of face_present / content_type -- see _text_regions()
    # docstring for why this always scans rather than skipping when it looks
    # unnecessary.
    regions.extend(_text_regions(img, st))

    # Display-only, like text -- see the spectral note in the module
    # docstring and in _State.__init__. Never touches confidence, verdict,
    # reliability, or explanation below; only signals.regularity.
    regularity_cal = None
    if st.spectral_w is not None:
        from quorum.features import spectral_features
        spec = spectral_features(img).astype(np.float64)
        spectral_raw = float(spec @ st.spectral_w + st.spectral_b)
        regularity_cal = _platt(spectral_raw, st.cal_spectral)

    # THE shipped score -- see module docstring, do not recompute by hand.
    # Since the face branch joined predict.py, this is a THREE-branch max
    # (general, tampered, face) -- omitting face= here would silently revert
    # to the old two-branch score and under-report confidence on every image
    # that has a face.
    confidence = float(st.shipped.score_embeddings(
        general_vec[None, :], st.probes, face=np.array([face_shipped]))[0])
    verdict = _verdict_from_score(confidence)

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
            "regularity": round(regularity_cal, 4) if regularity_cal is not None else None,
        },
        "content_type": content_type,
        "explanation": explanation,
        "reliability": reliability,
        "regions": regions,
    }
 