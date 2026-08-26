from __future__ import annotations

import cv2
import numpy as np

# kind can be one of: "number" | "text" | "date"
FIELD_ROIS: dict[str, dict] = {
    "national_id": {"box": [0.367, 0.236, 0.822, 0.337], "kind": "number"},
    "first_name": {"box": [0.369, 0.343, 0.823, 0.442], "kind": "text"},
    "last_name": {"box": [0.370, 0.447, 0.824, 0.546], "kind": "text"},
    "birth_date": {"box": [0.371, 0.552, 0.825, 0.650], "kind": "date"},
    "father_name": {"box": [0.371, 0.654, 0.826, 0.739], "kind": "text"},
    "expiry_date": {"box": [0.371, 0.745, 0.827, 0.829], "kind": "date"},
    "photo": {"box": [0.055, 0.199, 0.357, 0.759], "kind": "photo"},
}

FIELD_LABELS = {
    "national_id": "National ID",
    "first_name": "First Name",
    "last_name": "Last Name",
    "father_name": "Father's Name",
    "birth_date": "Birth Date",
    "expiry_date": "Expiry Date",
    "photo": "Photo",
}

_TEXT_FIELDS = ("national_id", "first_name", "last_name", "birth_date", "father_name", "expiry_date")


def draw_debug_overlay(card_image: np.ndarray) -> np.ndarray:
    """Return the card image with ROI boxes overlaid (for calibration)."""
    overlay = card_image.copy()
    height, width = overlay.shape[:2]

    color_map = {
        "number": (0, 200, 255),
        "text": (60, 220, 60),
        "date": (255, 140, 0),
        "photo": (0, 0, 255),
    }

    for field, config in FIELD_ROIS.items():
        x1, y1, x2, y2 = config["box"]
        pt1 = (int(x1 * width), int(y1 * height))
        pt2 = (int(x2 * width), int(y2 * height))
        color = color_map.get(config["kind"], (255, 255, 255))

        cv2.rectangle(overlay, pt1, pt2, color, 2)
        cv2.putText(
            overlay,
            field,
            (pt1[0], max(12, pt1[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    return overlay