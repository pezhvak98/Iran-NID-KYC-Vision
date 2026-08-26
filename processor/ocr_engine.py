"""
OCR Layer: EasyOCR + shape-based digit recognizer (B Yekan).

Fusion rule: The shape-based engine only overrides when constrained by
(checksum / valid date / agreement). Otherwise, EasyOCR with downstream
fixes is used.
"""

from __future__ import annotations

import threading

from text_utils import (
    DATE_ALLOWLIST,
    DIGIT_ALLOWLIST,
    NAME_ALLOWLIST,
    is_valid_national_code,
    only_digits,
)

_reader = None
_reader_lock = threading.Lock()
_gpu_enabled: bool | None = None


def _detect_gpu() -> bool:
    global _gpu_enabled
    if _gpu_enabled is not None:
        return _gpu_enabled
    try:
        import torch
        _gpu_enabled = bool(torch.cuda.is_available())
    except Exception:
        _gpu_enabled = False
    return _gpu_enabled


def create_reader():
    import easyocr
    return easyocr.Reader(["fa", "en"], gpu=_detect_gpu(), verbose=False)


def get_reader():
    global _reader
    if _reader is None:
        with _reader_lock:
            if _reader is None:
                _reader = create_reader()
    return _reader


def warmup() -> None:
    try:
        import numpy as np
        reader = get_reader()
        dummy = np.full((64, 200, 3), 255, dtype="uint8")
        with _reader_lock:
            reader.readtext(dummy, detail=0)
    except Exception as error:
        print(f"[ocr_engine] warmup failed (will retry on first real request): {error}")


def is_gpu_active() -> bool:
    return _detect_gpu()


def _run_ocr(image, allowlist: str):
    reader = get_reader()
    with _reader_lock:
        return reader.readtext(
            image,
            detail=1,
            allowlist=allowlist or None,
            mag_ratio=2.0,
            contrast_ths=0.05,
            adjust_contrast=0.7,
            text_threshold=0.5,
            low_text=0.3,
            paragraph=False,
        )


def _box_center_x(box) -> float:
    xs = [point[0] for point in box]
    return sum(xs) / len(xs)


# ---------------------------------------------------------------------------
# Shape-Based Digit Engine (B Yekan patterns) — Outputs Latin digits
# ---------------------------------------------------------------------------

def _cv_digits(image) -> tuple[str | None, float]:
    try:
        from digit_recognizer import recognize_digits
        return recognize_digits(image)
    except Exception:
        return None, 0.0


def _easyocr_digits(image) -> tuple[str | None, float]:
    results = _run_ocr(image, DIGIT_ALLOWLIST)
    if not results:
        return None, 0.0
    results.sort(key=lambda item: _box_center_x(item[0]))
    digits = "".join(only_digits(text) for _, text, _ in results)
    confidences = [float(c) for _, _, c in results]
    if not digits:
        return None, 0.0
    return digits, sum(confidences) / len(confidences)


def _is_valid_date8(digits8: str) -> bool:
    if not digits8 or len(digits8) != 8:
        return False
    y, m, d = int(digits8[:4]), int(digits8[4:6]), int(digits8[6:])
    return 1250 <= y <= 1420 and 1 <= m <= 12 and 1 <= d <= 31


def read_digits(image) -> tuple[str | None, float]:
    easy_d, easy_c = _easyocr_digits(image)
    cv_d, cv_c = _cv_digits(image)

    if easy_d and cv_d and cv_d == only_digits(easy_d):
        return cv_d, min(1.0, max(easy_c, cv_c) + 0.2)

    if cv_d and len(cv_d) == 10 and is_valid_national_code(cv_d):
        return cv_d, max(cv_c, 0.9)

    if easy_d:
        return easy_d, easy_c

    if cv_d and len(cv_d) == 10 and cv_c >= 0.85:
        return cv_d, cv_c

    return (cv_d, cv_c) if cv_d else (None, 0.0)


def read_date(image) -> tuple[str | None, float]:
    easy_d, easy_c = _easyocr_digits(image)
    cv_d, cv_c = _cv_digits(image)
    easy8 = only_digits(easy_d) if easy_d else None

    if easy8 and cv_d and cv_d == easy8:
        return cv_d, min(1.0, max(easy_c, cv_c) + 0.2)

    if cv_d and _is_valid_date8(cv_d) and not (easy8 and _is_valid_date8(easy8)):
        return cv_d, cv_c

    if easy8 and _is_valid_date8(easy8) and not (cv_d and _is_valid_date8(cv_d)):
        return easy8, easy_c

    results = _run_ocr(image, DATE_ALLOWLIST)
    if results:
        results.sort(key=lambda item: _box_center_x(item[0]))
        fragments = [text.strip() for _, text, _ in results if text and text.strip()]
        if fragments:
            confidences = [float(c) for _, _, c in results]
            return "".join(fragments), sum(confidences) / len(confidences)

    if easy_d:
        return easy_d, easy_c
    if cv_d:
        return cv_d, cv_c
    return None, 0.0


def read_name(image) -> tuple[str | None, float]:
    results = _run_ocr(image, NAME_ALLOWLIST)
    if not results:
        return None, 0.0
    results.sort(key=lambda item: _box_center_x(item[0]), reverse=True)
    fragments = [text.strip() for _, text, _ in results if text and text.strip()]
    confidences = [float(conf) for _, _, conf in results]
    if not fragments:
        return None, 0.0
    return " ".join(fragments), sum(confidences) / len(confidences)


def read_free_text(image) -> list[dict]:
    results = _run_ocr(image, "")
    lines = []
    for box, text, confidence in results or []:
        normalized_box = []
        for point in box:
            try:
                normalized_box.append([int(point[0]), int(point[1])])
            except (TypeError, ValueError):
                continue
        lines.append(
            {"text": text, "confidence": float(confidence), "box": normalized_box}
        )
    return lines