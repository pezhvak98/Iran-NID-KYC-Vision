
from __future__ import annotations

import importlib.metadata as _metadata

import cv2
import numpy as np

from card_layout import FIELD_ROIS
from image_processing import crop_roi

_face_cascade = None
_face_cascade_load_error: str | None = None


def _installed_opencv_packages() -> list[str]:
    """
    Read the Python environment directly to avoid guessing which OpenCV
    package is installed. If multiple OpenCV variants are detected 
    simultaneously, that is the root cause.
    """
    found = []
    try:
        for dist in _metadata.distributions():
            name = dist.metadata.get("Name", "")
            if name and "opencv" in name.lower():
                found.append(f"{name}=={dist.version}")
    except Exception:  # noqa: BLE001
        pass
    return sorted(set(found))


def _get_face_cascade():
    global _face_cascade, _face_cascade_load_error

    if _face_cascade is not None or _face_cascade_load_error is not None:
        return _face_cascade

    try:
        if not hasattr(cv2, "CascadeClassifier"):
            raise AttributeError(
                "Your OpenCV version does not support CascadeClassifier."
            )

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)

        if cascade.empty():
            raise ValueError("Cascade file failed to load.")

        _face_cascade = cascade
    except Exception as error:
        # Convert the scary error into a simple warning since the base ROI crop is good enough.
        _face_cascade_load_error = str(error)

    return _face_cascade


def _pad_box(
    x: int,
    y: int,
    w: int,
    h: int,
    frame_w: int,
    frame_h: int,
    top: float,
    bottom: float,
    side: float
) -> tuple[int, int, int, int]:
    """Add padding around the detected face box in a passport-photo style."""
    new_x1 = max(0, int(x - w * side))
    new_x2 = min(frame_w, int(x + w * (1 + side)))
    new_y1 = max(0, int(y - h * top))
    new_y2 = min(frame_h, int(y + h * (1 + bottom)))
    return new_x1, new_y1, new_x2, new_y2


def extract_face_photo(card_image: np.ndarray) -> dict:
    """
    Output:
        {
            "image": np.ndarray or None,
            "face_detected": bool,
            "warning": str or None,
        }
    """
    photo_config = FIELD_ROIS.get("photo")

    if not photo_config:
        return {
            "image": None,
            "face_detected": False,
            "warning": "Photo region not defined in card_layout."
        }

    coarse_crop = crop_roi(card_image, photo_config["box"], pad_ratio=0.15)

    if coarse_crop is None or coarse_crop.size == 0:
        return {
            "image": None,
            "face_detected": False,
            "warning": "Photo region could not be cropped."
        }

    cascade = _get_face_cascade()

    if cascade is None:
        return {
            "image": coarse_crop,
            "face_detected": False,
            "warning": (
                "Smart face detection is disabled due to library limitations; "
                "using approximate photo region."
            ),
        }

    gray = cv2.cvtColor(coarse_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=5,
        minSize=(40, 40),
    )

    if len(faces) == 0:
        return {
            "image": coarse_crop,
            "face_detected": False,
            "warning": (
                "Face not detected automatically; returning the entire "
                "approximate photo region."
            ),
        }

    # If multiple faces are found, select the largest one (smaller ones are
    # usually noise or graphic elements, not the main face).
    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])

    frame_h, frame_w = coarse_crop.shape[:2]
    x1, y1, x2, y2 = _pad_box(
        x, y, w, h, frame_w, frame_h,
        top=0.55,     # Space above the head for hair
        bottom=0.85,  # Space below for shoulders/chin
        side=0.35,
    )

    if x2 <= x1 or y2 <= y1:
        return {
            "image": coarse_crop,
            "face_detected": False,
            "warning": "Final crop was invalid."
        }

    final_crop = coarse_crop[y1:y2, x1:x2]

    return {
        "image": final_crop,
        "face_detected": True,
        "warning": None
    }


def encode_photo_png(image: np.ndarray) -> bytes | None:
    if image is None or image.size == 0:
        return None

    success, buffer = cv2.imencode(".png", image)
    if not success:
        return None

    return buffer.tobytes()