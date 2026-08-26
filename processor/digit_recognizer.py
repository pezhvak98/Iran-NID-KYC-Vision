"""
Persian digit recognition using font-aware Template Matching (B Yekan) — no training, no torch.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))

FONT_CANDIDATES = [
    os.path.join(_BASE, "BYekan.ttf"),
    os.path.join(_BASE, "fonts", "BYekan.ttf"),
    "C:/Windows/Fonts/BYekan.ttf",
    "C:/Windows/Fonts/tahoma.ttf",
]

PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
REJECT_CHARS = "/-"

_TEMPLATE_SIZE = 28
_templates: dict[str, list[np.ndarray]] | None = None
_last_debug: list = []


def _load_font(size: int):
    from PIL import ImageFont

    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    raise RuntimeError(
        "B Yekan font not found; place BYekan.ttf next to digit_recognizer.py."
    )


def _render_mask(char: str, render_height: int = 64) -> np.ndarray:
    from PIL import Image, ImageDraw

    font = _load_font(render_height)
    img = Image.new("L", (render_height * 2, render_height * 2), 0)
    draw = ImageDraw.Draw(img)
    draw.text((render_height // 2, render_height // 2), char, font=font, fill=255)
    arr = np.array(img)
    ys, xs = np.where(arr > 100)
    if len(xs) == 0:
        return np.zeros((8, 8), np.uint8)
    return arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _normalize(mask: np.ndarray, size: int = _TEMPLATE_SIZE) -> np.ndarray:
    """
    Normalization while preserving aspect ratio: first pad to a square, then resize.
    (The previous version stretched directly to a square, which made '1' look like '/'.)
    Since both templates and queries go through this function, the comparison remains fair.
    """
    h, w = mask.shape[:2]
    side = max(h, w)
    canvas = np.zeros((side, side), np.uint8)
    y0 = (side - h) // 2
    x0 = (side - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = mask

    resized = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(resized, (3, 3), 0)
    return blurred.astype(np.float32) / 255.0


def _get_templates() -> dict[str, list[np.ndarray]]:
    global _templates
    if _templates is None:
        _templates = {}
        kernel = np.ones((2, 2), np.uint8)
        for i, ch in enumerate(PERSIAN_DIGITS):
            base = _render_mask(ch)
            _templates[str(i)] = [
                _normalize(base),
                _normalize(cv2.dilate(base, kernel)),
                _normalize(cv2.erode(base, kernel)),
            ]
        for ch in REJECT_CHARS:
            _templates["reject:" + ch] = [_normalize(_render_mask(ch))]
    return _templates


def _soft_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = float((a * b).sum())
    union = float((a * a).sum() + (b * b).sum() - inter)
    return inter / union if union > 1e-6 else 0.0


def _best_shift_score(query: np.ndarray, template: np.ndarray) -> float:
    best = 0.0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            shifted = np.roll(np.roll(query, dy, axis=0), dx, axis=1)
            score = _soft_iou(shifted, template)
            if score > best:
                best = score
    return best


def classify_blob(blob: np.ndarray) -> tuple[str, float]:
    query = _normalize(blob)
    best_key, best_score = "reject", 0.0
    for key, variants in _get_templates().items():
        for tpl in variants:
            score = _best_shift_score(query, tpl)
            if score > best_score:
                best_key, best_score = key, score
    if best_key.startswith("reject:"):
        return "reject", best_score
    return best_key, best_score


def _binarize(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape[:2]
    if h < 80:
        scale = 80.0 / h
        gray = cv2.resize(gray, (int(w * scale), 80), interpolation=cv2.INTER_CUBIC)
    denoised = cv2.medianBlur(gray, 3)
    bs = 25 if min(denoised.shape) > 30 else 15
    if bs % 2 == 0:
        bs += 1
    binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, bs, 10
    )
    # If adaptive is too extreme on already-normalized near-binary images, use Otsu
    black_ratio = 1.0 - (np.count_nonzero(binary) / binary.size)
    if black_ratio < 0.02 or black_ratio > 0.75:
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def segment_with_boxes(gray: np.ndarray):
    binary = _binarize(gray)
    inv = 255 - binary
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, 8)

    comps = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if h < 8 or area < 12:
            continue
        comps.append([x, y, w, h, (labels[y:y + h, x:x + w] == i).astype(np.uint8) * 255])
    if not comps:
        return []

    # Height-based filtering using median (robust to a few tall label characters)
    heights = sorted(c[3] for c in comps)
    med_h = heights[len(heights) // 2]
    comps = [c for c in comps if 0.5 * med_h <= c[3] <= 2.2 * med_h]
    if not comps:
        return []

    cys = [c[1] + c[3] / 2.0 for c in comps]
    med_cy = sorted(cys)[len(cys) // 2]
    filtered = [c for c, cy in zip(comps, cys) if abs(cy - med_cy) <= 0.7 * med_h]
    if filtered:
        comps = filtered

    comps.sort(key=lambda c: c[0])
    widths = sorted(c[2] for c in comps)
    median_w = widths[len(widths) // 2]

    out = []
    for x, y, w, h, mask in comps:
        for blob in _split_wide(mask, median_w):
            out.append((x, y, w, h, blob))
    return out


def _split_wide(blob: np.ndarray, median_w: float) -> list[np.ndarray]:
    h, w = blob.shape
    if w < 1.6 * median_w or w < 1.4 * h:
        return [blob]
    col_sum = blob.sum(axis=0).astype(float)
    lo, hi = int(w * 0.25), int(w * 0.75)
    if hi <= lo:
        return [blob]
    cut = lo + int(np.argmin(col_sum[lo:hi]))
    parts = []
    for p in (blob[:, :cut], blob[:, cut:]):
        ys, xs = np.where(p > 0)
        if len(xs) == 0:
            continue
        parts.append(p[ys.min():ys.max() + 1, xs.min():xs.max() + 1])
    return parts if parts else [blob]


def recognize_digits(image, min_digits: int = 6) -> tuple[str | None, float]:
    global _last_debug
    _last_debug = []
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        comps = segment_with_boxes(gray)
        if len(comps) < min_digits:
            return None, 0.0

        scored = []
        for (x, y, w, h, blob) in comps:
            label, score = classify_blob(blob)
            _last_debug.append((x, y, w, h, label, round(float(score), 2)))
            if label != "reject":
                scored.append((label, float(score)))
        if not scored:
            return None, 0.0

        max_score = max(s for _, s in scored)
        cutoff = max(0.55, max_score - 0.35)
        digits = [lab for lab, s in scored if s >= cutoff]
        kept = [s for _, s in scored if s >= cutoff]

        if len(digits) < min_digits or len(digits) > 12:
            return None, 0.0
        return "".join(digits), float(sum(kept) / len(kept))
    except Exception:
        return None, 0.0


def save_debug(image, path: str = "debug_digits.jpg") -> None:
    try:
        vis = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for (x, y, w, h, label, score) in _last_debug:
            color = (0, 255, 0) if label != "reject" else (0, 0, 255)
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 1)
            cv2.putText(
                vis,
                f"{label}:{score}",
                (x, max(10, y - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 0, 0),
                1
            )
        cv2.imwrite(path, vis)
    except Exception:
        pass