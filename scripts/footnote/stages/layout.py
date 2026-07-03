"""Stage 7: layout planner.

For each researched overlay, pick a screen position that doesn't cover
the speaker's face. We use a deterministic heuristic + LLM only as a
tiebreaker for ambiguous frames.

The deterministic side:
  1. Detect speaker face bbox (uses OpenCV haarcascade — fast, no API)
  2. If face is in upper half → lower-third placement
  3. If face is in lower half → upper-third placement
  4. If face is centered → side-right (most talking-head videos have the
     speaker in the center)

This is intentionally NOT relying on LLM vision for every frame — it's
expensive and inconsistent across frames (the temporal-consistency problem).
The deterministic heuristic produces stable placements; we use LLM vision
only for unusual frames where face detection fails.
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# ─── Face detection (optional, uses OpenCV if available) ────────────────────

def _detect_face_bbox(image_path: str) -> tuple[float, float, float, float] | None:
    """Return (x, y, w, h) in 0-1 normalized coords. None if no face."""
    try:
        import cv2  # type: ignore
    except ImportError:
        return None
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = cascade.detectMultiScale(gray, 1.1, 5)
        if len(faces) == 0:
            return None
        # Pick the largest face
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        return (x / w, y / h, fw / w, fh / h)
    except Exception as e:
        logger.warning("face detection failed: %s", e)
        return None


# ─── Deterministic placement rules ──────────────────────────────────────────

def _placement_for_face(face_bbox: tuple[float, float, float, float] | None,
                         intent: str) -> tuple[float, float, float, float]:
    """Given face bbox and the proposer's placement intent, pick the actual
    bbox for the overlay. Output is (x, y, w, h) in 0-1 normalized.

    Overlay dimensions: 480x140 px on a 1920x1080 frame ≈ 25% wide × 13% tall.
    """
    # Overlay default dimensions
    OV_W, OV_H = 0.42, 0.20

    if face_bbox is None:
        # Fallback: lower-left third
        return (0.05, 0.70, OV_W, OV_H)

    fx, fy, fw, fh = face_bbox
    face_center_y = fy + fh / 2

    # Face in upper half → lower-third overlay
    if face_center_y < 0.5:
        return (0.5 - OV_W / 2, 0.74, OV_W, OV_H)
    # Face in lower half → upper-third
    if face_center_y > 0.6:
        return (0.5 - OV_W / 2, 0.06, OV_W, OV_H)
    # Face center; place on the opposite side from intent
    if intent == "side-left":
        return (0.55, 0.55, OV_W, OV_H)
    return (0.03, 0.55, OV_W, OV_H)


# ─── Animations ─────────────────────────────────────────────────────────────

def _animations_for(placement_intent: str) -> tuple[str, str]:
    """Pick entry + exit animations based on intent. Stable choices that
    look professional and avoid distraction."""
    if "lower" in placement_intent:
        return ("slide-up", "slide-down")
    if "upper" in placement_intent:
        return ("slide-down", "slide-up")
    if "left" in placement_intent:
        return ("slide-left", "slide-left")
    return ("fade", "fade")


# ─── Layout for all overlays ────────────────────────────────────────────────

def layout_all(researched_overlays: list[dict], frames: list[dict]) -> list[dict]:
    """For each researched overlay, compute placement + animations.

    frames is the sampled-frames list from stage 3; we pick the frame
    closest to the overlay's timestamp.
    """
    layouts: list[dict] = []
    for i, ro in enumerate(researched_overlays):
        ts = ro["proposal"].get("timestamp_sec", 0)
        # Find closest frame
        if frames:
            frame = min(frames, key=lambda f: abs(f["timestamp"] - ts))
            face = _detect_face_bbox(frame["path"]) if Path(frame["path"]).exists() else None
        else:
            face = None
        intent = ro["proposal"].get("placement_intent", "lower-third")
        bbox = _placement_for_face(face, intent)
        entry, exit_ = _animations_for(intent)
        layouts.append({
            "overlay_idx": i,
            "bbox": bbox,
            "entry_animation": entry,
            "exit_animation": exit_,
            "detected_face": face,
        })
    return layouts
