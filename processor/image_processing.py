from __future__ import annotations

import cv2
import numpy as np

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

# Standard ID card aspect ratio (ISO/IEC 7810 ID-1) ~= 1.586.
CARD_ASPECT_RATIO = 85.60 / 53.98
# How far a candidate quad's ratio may deviate (either orientation) and
# still be considered a card. Wide enough to tolerate perspective/rotation.
CARD_ASPECT_TOLERANCE = 0.22

# Fixed output size for the perspective-corrected card. Keeping this fixed
# (rather than "whatever size the detected quad happens to be") is what
# lets every downstream ROI box use the same normalized coordinates.
CARD_W, CARD_H = 1100, 694

# Width the detection stage downscales to before searching for the card.
# Only affects detection speed/robustness; the final warp always uses the
# full-resolution source image.
DETECTION_TARGET_WIDTH = 700

# Fixed working width for blur/brightness assessment. Keeps calculate_blur_score
# consistent across different input resolutions (variance-of-Laplacian is not
# scale-invariant) and keeps assess_quality fast on very large phone photos.
QUALITY_ASSESSMENT_WIDTH = 1200


# --------------------------------------------------------------------------
# Decoding
# --------------------------------------------------------------------------

def decode_image(file_bytes: bytes) -> np.ndarray:
    """Decode uploaded bytes into an OpenCV BGR image."""
    nparr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Unable to decode the uploaded image.")

    return image


# --------------------------------------------------------------------------
# Quality assessment
# --------------------------------------------------------------------------

def calculate_blur_score(image: np.ndarray) -> float:
    """Variance of the Laplacian. Lower = blurrier.

    Resized to a fixed working width first: variance-of-Laplacian is not
    scale-invariant, so without this a 4000px phone photo and a 1200px one
    would land on different blur scales even at identical actual sharpness,
    making the fixed `is_blurry` threshold below inconsistent across
    devices. Fixing the width also keeps this fast on very large photos.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray, _ = _resize_for_detection(gray, target_width=QUALITY_ASSESSMENT_WIDTH)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def calculate_brightness(image: np.ndarray) -> float:
    """Mean brightness (HSV V channel)."""
    small, _ = _resize_for_detection(image, target_width=QUALITY_ASSESSMENT_WIDTH)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 2]))


def assess_quality(image: np.ndarray) -> dict:
    height, width = image.shape[:2]

    blur_score = calculate_blur_score(image)
    brightness = calculate_brightness(image)

    return {
        "width": width,
        "height": height,
        "blur_score": round(blur_score, 2),
        "brightness": round(brightness, 1),
        "is_blurry": blur_score < 80,
        "is_dark": brightness < 70,
        "is_overexposed": brightness > 210,
        "resolution_ok": min(width, height) >= 480,
    }


# --------------------------------------------------------------------------
# Lighting / sharpness enhancement
# --------------------------------------------------------------------------

def normalize_lighting(image: np.ndarray) -> np.ndarray:
    """Local contrast/lighting correction with CLAHE on the LAB L channel."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def adjust_gamma(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """gamma > 1 brightens; gamma < 1 darkens."""
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
    ).astype("uint8")

    return cv2.LUT(image, table)


def unsharp_mask(image: np.ndarray, amount: float = 1.2, radius: float = 3.0) -> np.ndarray:
    """Mild sharpening."""
    blurred = cv2.GaussianBlur(image, (0, 0), radius)
    return cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)


def auto_enhance(image: np.ndarray, quality: dict) -> np.ndarray:
    """Apply lighting correction and sharpening based on measured quality."""
    enhanced = normalize_lighting(image)

    brightness = quality["brightness"]
    gamma = 1.0

    if brightness < 75:
        gamma = 1.4
    elif brightness > 205:
        gamma = 0.75

    if gamma != 1.0:
        enhanced = adjust_gamma(enhanced, gamma)

    if quality["is_blurry"]:
        enhanced = unsharp_mask(enhanced)

    return enhanced


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------

def order_points(points: np.ndarray) -> np.ndarray:
    """
    Order 4 corner points consistently:
    0: top-left | 1: top-right | 2: bottom-right | 3: bottom-left
    """
    rect = np.zeros((4, 2), dtype="float32")

    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]

    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]

    return rect


def _aspect_score(quad: np.ndarray) -> float:
    """
    0.0  -> quad's aspect ratio is outside the tolerance window for an
            ID-1 card (in either orientation) -- hard reject.
    ~1.0 -> quad's aspect ratio almost exactly matches the real card.

    Used both as a hard filter (score <= 0 => discard) and as part of the
    soft ranking score for surviving candidates.
    """
    pts = order_points(quad)
    tl, tr, br, bl = pts

    width = (np.linalg.norm(br - bl) + np.linalg.norm(tr - tl)) / 2.0
    height = (np.linalg.norm(br - tr) + np.linalg.norm(bl - tl)) / 2.0

    if width <= 1 or height <= 1:
        return 0.0

    ratio = max(width, height) / min(width, height)
    deviation = abs(ratio - CARD_ASPECT_RATIO)

    if deviation > CARD_ASPECT_TOLERANCE:
        return 0.0

    return 1.0 - (deviation / CARD_ASPECT_TOLERANCE)


def four_point_transform(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Generic perspective correction that outputs whatever size the detected
    quad naturally implies. Kept as a public utility; the card pipeline
    itself uses `warp_card`, which outputs a *fixed* size so ROI boxes
    defined in normalized coordinates always line up.
    """
    rect = order_points(points)
    tl, tr, br, bl = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(br - tr)
    height_b = np.linalg.norm(bl - tl)
    max_height = max(int(height_a), int(height_b))

    if max_width < 20 or max_height < 20:
        raise ValueError("Detected card region is too small.")

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))


def warp_card(image: np.ndarray, quad: np.ndarray,
              width: int = CARD_W, height: int = CARD_H) -> np.ndarray:
    """Perspective-correct the card into a FIXED size so ROI boxes defined
    in normalized (0..1) coordinates always land on the right field.

    `order_points` only labels corners by their spatial position in the
    photo (top-left, top-right, ...) -- it has no idea which pair of
    opposite sides is the card's actual LONG (width) side versus its
    SHORT (height) side. If the card was photographed rotated ~90 degrees
    (held in portrait against a landscape frame, for instance), the
    "top edge" in the photo is really the card's short edge. Warping that
    straight onto the fixed WIDTHxHEIGHT canvas would stretch the short
    edge to fill the full width and squeeze the long edge into the short
    output height -- a visibly distorted result, and every ROI box
    defined for the correct orientation would then land on the wrong
    field entirely. This is detected by comparing the two pairs of
    opposite sides' actual measured lengths and relabelling the corners
    (rotating them 90 degrees) whenever the "vertical" pair turns out to
    be the longer one, before the fixed-size transform is built. Any
    resulting 180-degree ambiguity from this relabelling is caught and
    fixed afterwards by `_ensure_orientation`, so only the long/short
    assignment needs to be corrected here.
    """
    pts = order_points(quad).astype(np.float32)
    tl, tr, br, bl = pts

    horizontal = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2.0
    vertical = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0
    if vertical > horizontal:
        tl, tr, br, bl = bl, tl, tr, br
        pts = np.array([tl, tr, br, bl], dtype=np.float32)

    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(image, matrix, (width, height), flags=cv2.INTER_CUBIC)


# --------------------------------------------------------------------------
# Card detection: shared low-level building blocks
# --------------------------------------------------------------------------

def _resize_for_detection(image: np.ndarray, target_width: int = DETECTION_TARGET_WIDTH):
    """Downscale for the (relatively expensive) detection search. Detection
    only needs to find *where* the card is, not read it, so working at a
    fraction of the resolution is a big speed win with no accuracy cost --
    final corners are refined on the full-resolution image afterwards."""
    height, width = image.shape[:2]
    if width <= target_width:
        return image, 1.0
    scale = target_width / float(width)
    resized = cv2.resize(image, (target_width, int(height * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


def _skin_mask(image: np.ndarray) -> np.ndarray:
    """Hand/skin region mask (YCrCb), dilated to safely cover fingers plus a
    small margin. Subtracted from every candidate-generation mask below so a
    hand holding the card can neither merge with the card's contour nor be
    mistaken for it.

    The raw YCrCb range below is intentionally broad (it has to cover a wide
    range of skin tones), which means it can also fire on thin, coincidentally
    warm-toned details that are NOT a hand at all -- a ruled-paper line, a
    printed rule, a card's own beige background. Those false positives are
    only a few pixels wide, whereas a real gripping hand is a broad blob, so
    an opening with a wide kernel is applied first to strip out anything too
    thin to plausibly be a finger/palm before the mask is used to exclude
    pixels from card detection.
    """
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    return cv2.dilate(mask, np.ones((11, 11), np.uint8), iterations=2)


def _gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude, used for the edge-support check below.
    Computed once per detection call and shared by every candidate."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def _edge_support_score(grad_mag: np.ndarray, quad: np.ndarray, samples_per_side: int = 24) -> float:
    """
    Sample the real image gradient magnitude along all 4 sides of a
    candidate quad and return a 0..1 confidence score.

    This is a key defence against several false-positive patterns:
      - Ruled paper lines can form rectangle-ish blobs, but the actual
        photographed *card edge* almost never lines up with them, so a
        candidate built purely from paper-line contours tends to score low
        here even if its shape happens to pass the aspect-ratio test.
      - A hand gripping part of the border locally lowers this score
        (occluded side has weak/no true edge), but with three sides still
        intact the average typically stays high enough for the correct
        quad to still win against unrelated false positives.
      - A natural surface with one strong physical feature (e.g. a
        countertop seam or a stone vein) plus otherwise-unrelated texture
        noise on the other sides can still average out to a deceptively
        high score. The score is therefore weighted toward the WEAKEST of
        the four sides, not just the overall average: a genuine card has
        comparably strong contrast on all four sides, while a coincidental
        match typically has one strong edge carrying the other three.
    """
    h, w = grad_mag.shape[:2]
    pts = order_points(quad)

    side_means = []
    for i in range(4):
        p0, p1 = pts[i], pts[(i + 1) % 4]
        total = 0.0
        count = 0
        # Skip the very ends of each side (near corners) where the sampled
        # gradient is noisier due to interpolation/rounding of the corner.
        for t in np.linspace(0.08, 0.92, samples_per_side):
            x, y = p0 + (p1 - p0) * t
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < w and 0 <= yi < h:
                total += grad_mag[yi, xi]
                count += 1
        side_means.append(total / count if count else 0.0)

    # Empirically, a confidently photographed edge sits well above ~60 on
    # this scale; normalize and clip so each side's score stays in [0, 1].
    side_scores = [min(1.0, m / 60.0) for m in side_means]
    return float(0.55 * min(side_scores) + 0.45 * (sum(side_scores) / 4.0))


def _polygon_from_contour(contour: np.ndarray) -> np.ndarray | None:
    """Reduce a contour to a 4-point convex polygon if possible. Falls back
    to approximating the convex hull with a looser epsilon, which recovers
    a clean quad even when card corners are slightly rounded or a short
    stretch of the border is occluded/noisy."""
    peri = cv2.arcLength(contour, True)
    if peri <= 0:
        return None

    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    if len(approx) != 4:
        hull = cv2.convexHull(contour)
        approx = cv2.approxPolyDP(hull, 0.03 * peri, True)

    if len(approx) != 4 or not cv2.isContourConvex(approx):
        return None

    return approx.reshape(4, 2).astype(np.float32)


def _texture_density(fine_binary: np.ndarray, quad: np.ndarray) -> float:
    """Fraction of the quad's interior that has dense fine print (card body
    is busy; a blank patch of desk/paper mistakenly picked up as the card
    would score low here)."""
    mask = np.zeros(fine_binary.shape, np.uint8)
    cv2.fillConvexPoly(mask, quad.astype(int), 255)
    area = cv2.countNonZero(mask)
    if area <= 0:
        return 0.0
    return cv2.countNonZero(cv2.bitwise_and(fine_binary, mask)) / float(area)


def _print_map_no_lines(gray: np.ndarray) -> np.ndarray:
    """
    Fine print/texture map with long straight lines removed.

    This is what makes the texture-based candidate generator safe to use
    on ruled/lined paper: notebook rulings are long straight segments, so
    opening the raw fine-edge map with long horizontal and vertical
    structuring elements isolates (and lets us subtract) exactly those
    lines, while short, irregular strokes -- printed text, the card's
    security pattern, portrait photo -- survive.
    """
    lap = cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_16S, ksize=3))
    _, fine = cv2.threshold(lap, 8, 255, cv2.THRESH_BINARY)

    horizontal_lines = cv2.morphologyEx(
        fine, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (31, 1))
    )
    vertical_lines = cv2.morphologyEx(
        fine, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 31))
    )
    return cv2.subtract(fine, cv2.add(horizontal_lines, vertical_lines))


# --------------------------------------------------------------------------
# Card detection: the three independent candidate-generation strategies
# --------------------------------------------------------------------------

def _build_edge_maps(gray: np.ndarray) -> list[np.ndarray]:
    """
    Build the four complementary binary edge maps shared by both the direct
    edge strategy and the interior-hole strategy below: a fixed-threshold
    Canny (most photos), an adaptive-threshold Canny (flat/low-contrast
    backgrounds), a median-based "auto" Canny (recovers weak/motion-blurred
    edges a fixed threshold misses), and a Laplacian-based fine-detail map
    (catches very subtle boundaries -- e.g. a pale card on white paper --
    that Canny's hysteresis thresholding can miss entirely on some sides
    even though the intensity step is real, just small). A light
    morphological close bridges the small gaps that blur or sensor noise
    leave in an otherwise continuous border before contours are traced.
    Computed once per detection call and reused, since these are among the
    more expensive steps in the pipeline.
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    close_kernel = np.ones((9, 9), np.uint8)

    canny = cv2.Canny(blurred, 60, 180)
    canny = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, close_kernel)
    canny = cv2.dilate(canny, None, iterations=2)

    adaptive = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 7
    )
    adaptive_edges = cv2.Canny(adaptive, 40, 140)
    adaptive_edges = cv2.morphologyEx(adaptive_edges, cv2.MORPH_CLOSE, close_kernel)
    adaptive_edges = cv2.dilate(adaptive_edges, None, iterations=2)

    med = float(np.median(blurred))
    lo, hi = int(max(0, 0.5 * med)), int(min(255, 1.2 * med))
    auto_edges = cv2.Canny(blurred, lo, hi)
    auto_edges = cv2.morphologyEx(auto_edges, cv2.MORPH_CLOSE, close_kernel)
    auto_edges = cv2.dilate(auto_edges, None, iterations=2)

    fine_edges = _print_map_no_lines(gray)
    fine_edges = cv2.dilate(fine_edges, np.ones((3, 3), np.uint8), iterations=1)
    fine_edges = cv2.morphologyEx(fine_edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    return [canny, adaptive_edges, auto_edges, fine_edges]


def _candidates_from_edges(edge_maps: list[np.ndarray], not_skin: np.ndarray,
                            area_ref: float, grad_mag: np.ndarray) -> list[tuple[float, np.ndarray]]:
    """Strategy A: Canny + adaptive-threshold edges -> polygon contours.
    General-purpose; the main workhorse on plain/uncluttered backgrounds
    and also on lined paper, where `_edge_support_score` filters out the
    paper's own lines (they don't line up with the card's real edge).

    A hand gripping the card's border does not just add noise, it changes
    the *shape* of the enclosed region: fingers sit outside the true edge,
    so the traced outline gets an extra convex bump there, which throws off
    a direct polygon fit. The fix is to fill the enclosed region first and
    only then carve out any skin-coloured pixels -- that turns the hand's
    contribution into a concave bite (missing interior) instead of a bump
    (added exterior). A bite leaves the true corners untouched, so the
    convex-hull fallback inside `_polygon_from_contour` transparently
    repairs it back into a clean rectangle; a bump could not be repaired
    that way.
    """
    candidates: list[tuple[float, np.ndarray]] = []
    for edges in edge_maps:
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:6]:
            area = cv2.contourArea(contour)
            # Card should fill a meaningful part of the frame, but not all
            # of it (that would usually be the frame border itself). The
            # lower bound is intentionally generous: a card photographed
            # from farther away, or one that doesn't fill an on-screen
            # guide exactly, can legitimately occupy a small fraction of
            # the frame.
            if area < area_ref * 0.025 or area > area_ref * 0.98:
                continue

            filled = np.zeros(edges.shape, np.uint8)
            cv2.drawContours(filled, [contour], -1, 255, -1)
            filled = cv2.bitwise_and(filled, not_skin)
            if cv2.countNonZero(filled) < area_ref * 0.02:
                continue

            inner_contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not inner_contours:
                continue
            inner = max(inner_contours, key=cv2.contourArea)

            quad = _polygon_from_contour(inner)
            if quad is None:
                # A bite that eats into an actual corner (not just the
                # middle of a side) leaves a pentagon that no epsilon will
                # cleanly reduce to 4 points -- the corner point is simply
                # gone, not just noisy. minAreaRect still recovers it well
                # in that case: the two sides meeting at that corner are
                # still long/clear enough to fix the rotated bounding box,
                # even with the corner pixels themselves missing.
                quad = cv2.boxPoints(cv2.minAreaRect(inner)).astype(np.float32)

            aspect = _aspect_score(quad)
            if aspect <= 0:
                continue

            edge_support = _edge_support_score(grad_mag, quad)
            score = 0.55 * aspect + 0.45 * edge_support
            candidates.append((score, quad))

    return candidates


def _candidates_from_holes(edge_maps: list[np.ndarray], not_skin: np.ndarray,
                            area_ref: float, grad_mag: np.ndarray) -> list[tuple[float, np.ndarray]]:
    """
    Strategy D: reconstruct the card from its own enclosed interior regions.

    On a background that's visually similar to the card (a pale, busy
    pattern like a Persian rug is the motivating case), the card's OUTER
    edge frequently fuses with nearby background edges once dilated --
    Strategy A then traces one large blob spanning card-plus-background,
    which fails the aspect-ratio gate (or worse, passes it by coincidence
    at the wrong scale). But the card's own printing (a photo box, text
    fields, the emblem watermark) still subdivides its blank interior into
    several small enclosed regions -- topological "holes" in the edge map,
    each still walled in by a clean, high-contrast edge (print-on-card
    contrast is usually much cleaner than the ambiguous card-vs-background
    contrast that broke Strategy A). `cv2.RETR_CCOMP` surfaces these holes
    directly regardless of what happens on the outside of their walls.

    A single hole is normally just one field, far too small and the wrong
    shape to be the card by itself. So starting from each of the largest
    few holes as a seed, this greedily adds whichever remaining hole most
    improves the convex hull's match to the card's aspect ratio, stopping
    once no further addition helps. On a real card the holes this recovers
    are the photo box, the title bar, and the text-field gaps -- together
    their convex hull reconstructs the full card boundary even though no
    single traced contour ever contained it.
    """
    candidates: list[tuple[float, np.ndarray]] = []

    for edges in edge_maps:
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            continue
        hierarchy = hierarchy[0]

        holes = []
        for i, c in enumerate(contours):
            if hierarchy[i][3] == -1:  # not a hole (no parent) -> skip
                continue
            area = cv2.contourArea(c)
            if area < area_ref * 0.003 or area > area_ref * 0.35:
                continue
            holes.append(c.reshape(-1, 2).astype(np.float32))

        if len(holes) < 2:
            continue

        seeds = sorted(holes, key=cv2.contourArea, reverse=True)[:4]
        pool = sorted(holes, key=cv2.contourArea, reverse=True)[:10]

        # How far apart two fragments may be and still plausibly belong to
        # the same card. Real card fields (photo box, title, text lines)
        # sit close together relative to the frame; this is what stops the
        # greedy growth below from "improving" its aspect ratio by jumping
        # to an unrelated hole on the far side of a busy background.
        gap_limit = 0.30 * (area_ref ** 0.5)

        for seed in seeds:
            current = seed
            remaining = [h for h in pool if h is not seed]
            merges = 0

            hull = cv2.convexHull(current.reshape(-1, 1, 2))
            quad = _polygon_from_contour(hull)
            best_score = _aspect_score(quad) if quad is not None else 0.0

            improved = True
            while improved and remaining:
                improved = False
                cur_x0, cur_y0 = current[:, 0].min(), current[:, 1].min()
                cur_x1, cur_y1 = current[:, 0].max(), current[:, 1].max()

                grown = None
                grown_score = best_score
                for cand in remaining:
                    cand_x0, cand_y0 = cand[:, 0].min(), cand[:, 1].min()
                    cand_x1, cand_y1 = cand[:, 0].max(), cand[:, 1].max()
                    gap_x = max(0.0, max(cur_x0 - cand_x1, cand_x0 - cur_x1))
                    gap_y = max(0.0, max(cur_y0 - cand_y1, cand_y0 - cur_y1))
                    if (gap_x * gap_x + gap_y * gap_y) ** 0.5 > gap_limit:
                        continue

                    combo = np.vstack([current, cand])
                    hull = cv2.convexHull(combo.reshape(-1, 1, 2))
                    if cv2.contourArea(hull) > area_ref * 0.5:
                        continue
                    quad = _polygon_from_contour(hull)
                    if quad is None:
                        quad = cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)
                    score = _aspect_score(quad)
                    if score > grown_score:
                        grown_score = score
                        grown = cand
                if grown is not None:
                    current = np.vstack([current, grown])
                    remaining = [h for h in remaining if h is not grown]
                    best_score = grown_score
                    improved = True
                    merges += 1

            hull = cv2.convexHull(current.reshape(-1, 1, 2))
            quad = _polygon_from_contour(hull)
            if quad is None:
                quad = cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)
            aspect = _aspect_score(quad)
            if aspect <= 0:
                continue

            # A single hole (no merges at all) is just one card field on
            # its own -- far too small to plausibly be the whole card, even
            # on the rare occasion its raw aspect ratio happens to land
            # near 1.586 by chance. Requiring at least one successful merge
            # plus a plausible minimum area stops such coincidental small
            # fragments from outscoring a correct, larger reconstruction of
            # the actual card (which necessarily involves combining several
            # fields and so tends to score a bit lower on aspect alone).
            area_frac = cv2.contourArea(hull) / area_ref
            if merges == 0 or area_frac < 0.04:
                continue

            # Individual fragments are no longer skin-filtered (a warm-toned
            # patterned background, e.g. a rug, legitimately produces many
            # skin-coloured holes that still sit right at the true card
            # edge and are needed for a correct reconstruction). As a
            # lighter final safeguard instead, reject only if the finished
            # shape's own interior turns out to be mostly skin-coloured --
            # that's the actual signature of "this reconstructed a hand",
            # which individual-fragment filtering was trying to prevent.
            mask = np.zeros(not_skin.shape, np.uint8)
            cv2.fillConvexPoly(mask, quad.astype(np.int32), 255)
            mask_area = cv2.countNonZero(mask)
            if mask_area > 0:
                skin_frac = 1.0 - (cv2.countNonZero(cv2.bitwise_and(mask, not_skin)) / mask_area)
                if skin_frac > 0.4:
                    continue

            edge_support = _edge_support_score(grad_mag, quad)
            # Weighted toward aspect here more than the other strategies:
            # a reconstructed hull's "edge" is partly its own interior
            # print, not the true card border, so edge_support is a
            # noisier signal for this strategy specifically.
            score = 0.65 * aspect + 0.35 * edge_support
            candidates.append((score, quad))

    return candidates


def _candidates_from_texture(gray: np.ndarray, not_skin: np.ndarray,
                              area_ref: float, grad_mag: np.ndarray) -> list[tuple[float, np.ndarray]]:
    """Strategy B: dense-print texture blob (line-free). Good when the
    card's printed surface is visually busier than a plain background
    (e.g. a wooden desk) where Canny alone gives a weak/broken outline."""
    fine = _print_map_no_lines(gray)

    blob = cv2.dilate(fine, np.ones((7, 7), np.uint8), iterations=2)
    blob = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    blob = cv2.morphologyEx(blob, cv2.MORPH_OPEN, np.ones((21, 21), np.uint8))
    blob = cv2.bitwise_and(blob, not_skin)

    candidates: list[tuple[float, np.ndarray]] = []
    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:4]:
        area = cv2.contourArea(contour)
        if area < area_ref * 0.025 or area > area_ref * 0.97:
            continue

        # NOTE: rectangularity is intentionally checked on the polygon fit
        # below rather than on this raw contour -- `blob` already had the
        # skin mask subtracted, so a hand gripping an edge leaves a concave
        # bite here. A bite lowers contourArea without changing the
        # minAreaRect bound, which would unfairly fail a raw-contour
        # rectangularity gate even though the true card is still a clean
        # rectangle once the convex hull (inside `_polygon_from_contour`)
        # repairs the bite.
        quad = _polygon_from_contour(contour)
        if quad is None:
            quad = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32)

        aspect = _aspect_score(quad)
        if aspect <= 0:
            continue

        density = _texture_density(fine, quad)
        if density < 0.05:
            continue

        edge_support = _edge_support_score(grad_mag, quad)
        score = 0.35 * aspect + 0.35 * min(1.0, density * 2.0) + 0.30 * edge_support
        candidates.append((score, quad))

    return candidates


def _candidates_from_color(bgr: np.ndarray, not_skin: np.ndarray,
                            area_ref: float, grad_mag: np.ndarray) -> list[tuple[float, np.ndarray]]:
    """Strategy C: bright / low-saturation blob. Most ID cards are pale
    (white/light background) compared to patterned surroundings such as
    ruled paper or fabric, so this catches cases where the card's own
    internal edges are too faint for strategies A/B but it still stands
    out by colour."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    color = cv2.bitwise_and(cv2.inRange(v, 80, 255), cv2.inRange(s, 0, 130))
    color = cv2.morphologyEx(color, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    color = cv2.morphologyEx(color, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    color = cv2.bitwise_and(color, not_skin)

    candidates: list[tuple[float, np.ndarray]] = []
    contours, _ = cv2.findContours(color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:4]:
        area = cv2.contourArea(contour)
        if area < area_ref * 0.025 or area > area_ref * 0.97:
            continue

        # See the matching note in `_candidates_from_texture`: rectangularity
        # is deliberately not checked on this raw (possibly hand-bitten)
        # contour, only on the polygon fit below.
        quad = _polygon_from_contour(contour)
        if quad is None:
            quad = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32)

        aspect = _aspect_score(quad)
        if aspect <= 0:
            continue

        edge_support = _edge_support_score(grad_mag, quad)
        score = 0.5 * aspect + 0.5 * edge_support
        candidates.append((score, quad))

    return candidates


def _refine_quad(gray: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Snap the 4 corners onto the true edge with sub-pixel accuracy. Runs
    on the FULL-resolution grayscale image, since this is where sub-pixel
    precision actually matters for a clean perspective warp."""
    try:
        corners = quad.astype(np.float32).reshape(-1, 1, 2)
        refined = cv2.cornerSubPix(
            gray, corners, (7, 7), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01),
        ).reshape(4, 2)

        pts = order_points(refined)
        if _aspect_score(pts) > 0:
            return pts
    except Exception:  # noqa: BLE001 - subpixel refinement is best-effort
        pass

    return order_points(quad)


# --------------------------------------------------------------------------
# Card detection: orchestrator
# --------------------------------------------------------------------------

def _spans_full_frame(quad: np.ndarray, frame_w: int, frame_h: int,
                       threshold: float = 0.88) -> bool:
    """
    True if a candidate's bounding box covers almost the entire detection
    frame in BOTH dimensions at once, or extends outside the frame at all.
    A real card, photographed with any normal composition, essentially
    never does this -- there is always some visible background margin on
    at least one side, and its corners obviously can't lie beyond the
    photo itself. A quad that spans (nearly) the whole frame in both width
    and height, or overshoots it, is a reliable sign of a false positive
    (the photo's own border, a vignette, a busy background that happened
    to pass the aspect gate by coincidence, or an over-eager reconstruction
    that bridged unrelated fragments), not a genuine detection -- no
    matter how well it otherwise scored. Checked centrally, after all four
    strategies report their candidates, so no strategy has to special-case
    it.
    """
    xs, ys = quad[:, 0], quad[:, 1]
    if xs.min() < -1 or ys.min() < -1 or xs.max() > frame_w + 1 or ys.max() > frame_h + 1:
        return True
    width_frac = (xs.max() - xs.min()) / float(frame_w)
    height_frac = (ys.max() - ys.min()) / float(frame_h)
    return width_frac >= threshold and height_frac >= threshold


def _touches_frame_corner(quad: np.ndarray, frame_w: int, frame_h: int,
                           margin: float = 3.0) -> bool:
    """
    True if any corner of the candidate coincides with an actual corner of
    the detection frame (i.e. lands within `margin` px of BOTH a left/right
    edge AND a top/bottom edge at once). A real card practically never has
    a corner exactly at the photo's own corner -- that combination is the
    signature of a detector tracing the photo's own two border edges
    rather than a real object (seen in practice on a natural-stone
    background: a genuine countertop seam supplies one strong, real edge,
    and the image border supplies the other two sides "for free", so the
    candidate can otherwise score deceptively well). This is a narrower,
    stricter check than `_spans_full_frame`: it catches a small
    frame-hugging corner clip that doesn't span most of the frame at all.
    """
    xs, ys = quad[:, 0], quad[:, 1]
    near_x_edge = (xs <= margin) | (xs >= frame_w - 1 - margin)
    near_y_edge = (ys <= margin) | (ys >= frame_h - 1 - margin)
    return bool(np.any(near_x_edge & near_y_edge))



def detect_card(image: np.ndarray) -> np.ndarray | None:
    """
    Find the card's 4 corners in `image`'s own coordinate system.

    Runs four independent candidate generators (edges / texture / colour /
    interior-holes) on a downscaled copy, scores every candidate with a
    shared aspect + edge-support metric, discards any candidate that spans
    almost the entire frame (never a real card, always a false positive),
    and returns the sub-pixel-refined corners of the single best remaining
    candidate -- or None if nothing passed validation.

    Never raises: any internal failure is treated as "no card found" so
    callers can fall back to the guided center-crop without special-casing
    exceptions.
    """
    try:
        detect_img, scale = _resize_for_detection(image)
        h, w = detect_img.shape[:2]
        area_ref = float(h * w)

        gray = cv2.cvtColor(detect_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 7, 50, 50)
        grad_mag = _gradient_magnitude(gray)
        edge_maps = _build_edge_maps(gray)

        # A gripping hand must not be able to form, bridge, or masquerade
        # as the card's border in ANY of the strategies below.
        not_skin = cv2.bitwise_not(_skin_mask(detect_img))

        candidates: list[tuple[float, np.ndarray]] = []
        candidates += _candidates_from_edges(edge_maps, not_skin, area_ref, grad_mag)
        candidates += _candidates_from_texture(gray, not_skin, area_ref, grad_mag)
        candidates += _candidates_from_color(detect_img, not_skin, area_ref, grad_mag)
        candidates += _candidates_from_holes(edge_maps, not_skin, area_ref, grad_mag)

        candidates = [
            (s, q) for s, q in candidates
            if not _spans_full_frame(q, w, h) and not _touches_frame_corner(q, w, h)
        ]

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0], reverse=True)
        best_quad = candidates[0][1]

        if scale != 1.0:
            best_quad = best_quad / scale

        full_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return _refine_quad(full_gray, best_quad)

    except Exception:  # noqa: BLE001 - detection failure -> caller falls back
        return None


# --------------------------------------------------------------------------
# Fallback crop (used only when detect_card() returns None)
# --------------------------------------------------------------------------

def center_crop(image: np.ndarray, crop_ratio: float = 0.96) -> np.ndarray:
    """
    Fallback used when no candidate quad passes validation. The frontend
    already shows a card-shaped guide overlay on the live camera preview,
    so we assume the user has roughly aligned a real ID-1 card inside that
    guide -- and crop the largest CARD_ASPECT_RATIO rectangle centred in
    the frame, shrunk slightly to drop the guide's own border/background
    sliver.

    NOTE: this used to crop by shrinking the *camera frame's own* aspect
    ratio, which silently misaligned every ROI box whenever the frame
    wasn't already shaped like a card (almost always, since phone frames
    are portrait-ish and a card is landscape). Cropping to the card's real
    aspect ratio fixes that; `crop_ratio` keeps its previous meaning (how
    much of the frame to keep, 0..1).
    """
    height, width = image.shape[:2]
    margin_ratio = max(0.0, min(0.5, 1.0 - crop_ratio))

    frame_ratio = width / float(height)
    if frame_ratio >= CARD_ASPECT_RATIO:
        crop_h = height
        crop_w = int(height * CARD_ASPECT_RATIO)
    else:
        crop_w = width
        crop_h = int(width / CARD_ASPECT_RATIO)

    crop_w = max(20, int(crop_w * (1.0 - margin_ratio)))
    crop_h = max(20, int(crop_h * (1.0 - margin_ratio)))

    x = max(0, (width - crop_w) // 2)
    y = max(0, (height - crop_h) // 2)

    return image[y:y + crop_h, x:x + crop_w]


def resize_to_max_width(image: np.ndarray, max_width: int = 1100) -> np.ndarray:
    """
    Only ever shrinks, never enlarges: upscaling a small source image
    digitally does not add real detail and tends to hurt OCR accuracy more
    than it helps.
    """
    height, width = image.shape[:2]

    if width > max_width:
        scale = max_width / float(width)
        new_height = int(height * scale)
        return cv2.resize(image, (max_width, new_height), interpolation=cv2.INTER_AREA)

    return image


# --------------------------------------------------------------------------
# Orientation correction
# --------------------------------------------------------------------------

def _flag_orientation_score(card_image: np.ndarray) -> float:
    """
    Score for 'the Iran flag icon (a small graphic with a green stripe
    above a red stripe) sits in the top-left corner, where it always is
    printed on a correctly oriented card'.

    This is the primary, most reliable orientation signal. Its colours
    are strongly saturated and the green-above-red arrangement is
    specific enough that it is very hard to confuse with anything else --
    unlike skin-tone or brightness-based checks, which testing showed can
    be fooled by a busy background bleeding into the crop's edge, or by a
    privacy-redacted (blacked out) photo that no longer looks like real
    skin.
    """
    h, w = card_image.shape[:2]
    corner = card_image[0:int(h * 0.26), 0:int(w * 0.16)]
    if corner.size == 0:
        return 0.0

    hsv = cv2.cvtColor(corner, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, (35, 60, 50), (85, 255, 255))
    red_mask = cv2.bitwise_or(
        cv2.inRange(hsv, (0, 60, 50), (10, 255, 255)),
        cv2.inRange(hsv, (170, 60, 50), (180, 255, 255)),
    )
    total = corner.shape[0] * corner.shape[1]
    green_px = cv2.countNonZero(green_mask)
    red_px = cv2.countNonZero(red_mask)
    if green_px < total * 0.01 or red_px < total * 0.005:
        return 0.0

    # The flag's green stripe must actually sit above its red stripe, not
    # just have some of each colour present somewhere in the corner.
    green_y = float(np.where(green_mask > 0)[0].mean())
    red_y = float(np.where(red_mask > 0)[0].mean())
    if green_y >= red_y:
        return 0.0

    return float(min(1.0, (green_px + red_px) / (total * 0.06)))


def _photo_side_score(card_image: np.ndarray, side: str) -> float:
    """
    Fallback orientation signal, used only when the flag check above is
    inconclusive: fraction of skin-toned pixels in the region where the
    portrait photo sits if the card is right-side up (side='left') or
    upside down (side='right'). Kept with a comfortable margin from the
    card's own edges -- sampling too close to the border risks picking up
    a sliver of background that bled in from an imperfect crop, which
    testing showed can otherwise read as strongly skin-toned itself (a
    patterned rug, for instance).
    """
    h, w = card_image.shape[:2]
    if side == "left":
        x0, x1 = int(w * 0.05), int(w * 0.33)
    else:
        x0, x1 = int(w * 0.67), int(w * 0.95)
    y0, y1 = int(h * 0.15), int(h * 0.85)
    region = card_image[y0:y1, x0:x1]
    if region.size == 0:
        return 0.0
    skin = _skin_mask(region)
    return cv2.countNonZero(skin) / skin.size


def _digit_zone_score(card_image: np.ndarray) -> float:
    """
    Optional extra signal, used only as a last-resort tie-breaker: reads
    the national-ID-number zone (top-right on a correctly oriented card)
    via the project's own digit_recognizer, if available. Returns 0 (i.e.
    "no opinion") whenever that module isn't present or errors, so its
    absence never blocks the self-contained checks above -- this used to
    be the ONLY orientation signal, which meant a missing/failing
    digit_recognizer silently disabled orientation correction entirely.
    """
    try:
        from digit_recognizer import recognize_digits
        h, w = card_image.shape[:2]
        zone = card_image[int(h * 0.18):int(h * 0.42), int(w * 0.45):int(w * 0.95)]
        digits, score = recognize_digits(zone, min_digits=6)
        return float(score) if digits else 0.0
    except Exception:  # noqa: BLE001
        return 0.0


def _ensure_orientation(card_image: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Make sure the card is right-side up -- concretely, that the portrait
    photo ends up on the LEFT, which is how this card is always printed.
    Tries three independent signals, in order of reliability, and stops
    at the first one that gives a clear answer:

      1. Flag icon (top-left corner, green stripe above red) -- see
         `_flag_orientation_score`. Reliable even when the photo itself
         has been redacted or the background bled slightly into the crop.
      2. Skin-tone fraction, left region vs right region -- only used if
         the flag check found no flag-like pattern on EITHER side (e.g.
         unusual lighting washed out its colours, or a card template
         without that icon).
      3. digit_recognizer, if the deployment has it available. Last
         resort only, since it depends on an external module this file
         doesn't control and can't assume is present or working.

    If none of the three signals gives a clear answer, the image is left
    exactly as detected rather than guessing -- an uninformed rotation is
    worse than no rotation.
    """
    rotated_image = cv2.rotate(card_image, cv2.ROTATE_180)

    upright_flag = _flag_orientation_score(card_image)
    rotated_flag = _flag_orientation_score(rotated_image)
    if upright_flag > 0 or rotated_flag > 0:
        if rotated_flag > upright_flag:
            return rotated_image, True
        return card_image, False

    upright_photo = _photo_side_score(card_image, "left")
    rotated_photo = _photo_side_score(card_image, "right")
    # Skin-tone is the least specific of the three signals, so only act
    # on a clear, meaningful gap rather than a marginal difference.
    biggest = max(upright_photo, rotated_photo)
    if biggest > 0.03 and abs(upright_photo - rotated_photo) > 0.3 * biggest:
        if rotated_photo > upright_photo:
            return rotated_image, True
        return card_image, False

    upright_digits = _digit_zone_score(card_image)
    rotated_digits = _digit_zone_score(rotated_image)
    if rotated_digits > upright_digits:
        return rotated_image, True

    return card_image, False



# --------------------------------------------------------------------------
# Full pipeline orchestrator
# --------------------------------------------------------------------------

def _touches_frame_edge(quad: np.ndarray, image_shape: tuple, margin_px: int = 6) -> bool:
    """
    True if any corner of the detected quad sits at (or past) the edge of
    the camera frame itself. When that happens the card was photographed
    too close / not fully inside the shot, so part of it was simply never
    captured -- no amount of detection or perspective-correction accuracy
    can recover pixels that don't exist in the source image. This is
    reported as a distinct warning so it isn't mistaken for a detection
    failure the pipeline could have avoided.
    """
    h, w = image_shape[:2]
    xs, ys = quad[:, 0], quad[:, 1]
    return bool(
        xs.min() <= margin_px or ys.min() <= margin_px
        or xs.max() >= w - 1 - margin_px or ys.max() >= h - 1 - margin_px
    )


def prepare_card_image(image: np.ndarray) -> tuple[np.ndarray, bool, str | None]:
    """
    Full pipeline: enhance lighting -> robust card detection -> fixed-size
    perspective correction -> texture sanity check -> orientation fix ->
    resize -> document-style OCR normalization.

    Returns: (final card image, whether a card quad was auto-detected, a
    warning string or None).
    """
    quality = assess_quality(image)
    enhanced = auto_enhance(image, quality)

    warning: str | None = None
    card_detected = False

    # -- Detect the card (fast, no OCR involved) ---------------------------
    card_quad = detect_card(enhanced)

    # A quad touching the camera frame's own border means part of the card
    # was cut off by the shot itself -- flag this distinctly, since no
    # perspective correction can reconstruct pixels that were never
    # captured. Still proceed with whatever was captured rather than
    # discarding it outright.
    if card_quad is not None and _touches_frame_edge(card_quad, enhanced.shape):
        warning = (
            "The card appears to extend beyond the edge of the photo, so part of it may be "
            "missing from the result. Move back slightly so the whole card is inside the frame."
        )

    # -- Perspective-correct to a fixed size so ROI boxes stay accurate ----
    if card_quad is not None:
        try:
            card_image = warp_card(enhanced, card_quad)
            card_detected = True
        except Exception:  # noqa: BLE001
            card_image = center_crop(enhanced)
            warning = "Perspective correction on the detected quad failed; used a guided center-crop instead."
    else:
        card_image = center_crop(enhanced)
        warning = (
            "Card was not detected automatically; assumed the camera frame is already "
            "aligned to the on-screen guide. For best accuracy, place the card on a plain "
            "background and keep it fully visible (not gripped over its edges)."
        )

    # -- Sanity check: a correctly cropped card region is not texture-flat -
    if card_detected:
        gray_card = cv2.cvtColor(card_image, cv2.COLOR_BGR2GRAY)
        if gray_card.std() < 12:
            card_image = center_crop(enhanced)
            card_detected = False
            warning = (warning + " " if warning else "") + (
                "The detected region had no card-like texture; used a guided center-crop instead."
            )

    # -- Debug aid: draw the detected quad on the enhanced frame -----------
    try:
        debug_frame = enhanced.copy()
        if card_quad is not None:
            cv2.polylines(debug_frame, [card_quad.astype(int)], True, (0, 255, 0), 3)
        cv2.imwrite("debug_card_quad.jpg", debug_frame)
    except Exception:  # noqa: BLE001
        pass

    # -- Orientation fix based on content (portrait photo goes on the left) -
    card_image, rotated = _ensure_orientation(card_image)
    if rotated:
        warning = (warning + " " if warning else "") + "Card orientation was corrected by 180 degrees."

    card_image = resize_to_max_width(card_image, 1100)
    card_image = normalize_card_for_ocr(card_image)
    return card_image, card_detected, warning


# --------------------------------------------------------------------------
# Per-field ROI cropping
# --------------------------------------------------------------------------

def crop_roi(
    image: np.ndarray,
    box: list[float],
    pad_x_ratio: float = 0.02,
    pad_y_ratio: float = 0.10,
    pad_ratio: float | None = None,
) -> np.ndarray | None:
    """Crop a field from the perspective-corrected card. `box` is
    [x1, y1, x2, y2] in normalized (0..1) coordinates."""
    if pad_ratio is not None:
        pad_x_ratio = pad_ratio
        pad_y_ratio = pad_ratio

    height, width = image.shape[:2]
    x1 = max(0, int(box[0] * width))
    y1 = max(0, int(box[1] * height))
    x2 = min(width, int(box[2] * width))
    y2 = min(height, int(box[3] * height))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = image[y1:y2, x1:x2]

    pad_x = max(2, int((x2 - x1) * pad_x_ratio))
    pad_y = max(1, int((y2 - y1) * pad_y_ratio))

    return cv2.copyMakeBorder(crop, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REPLICATE)


# --------------------------------------------------------------------------
# OCR preprocessing
# --------------------------------------------------------------------------

def binarize_for_ocr(gray: np.ndarray) -> np.ndarray:
    """
    Convert a small, already-isolated grayscale crop (single field) into
    pure black-on-white: no intermediate gray shades.

    Why this matters: when text and background sit in overlapping shades
    of gray (low-contrast printing, or a security pattern behind the
    text), the character recognizer has to separate the real glyph shape
    from background noise, and that gray "no-man's-land" is the main
    source of confusions like 0 vs 5. Removing it entirely gives a sharp,
    unambiguous glyph edge.

    Since this runs on a small, already-isolated crop (not the whole card
    with very different regions), both adaptive threshold and Otsu are
    reasonable: adaptive is tried first (more robust to local lighting,
    e.g. a shadow across one corner of the card under a phone light); if
    its result is extreme (near all-black or all-white, meaning its
    parameters didn't suit this particular crop), we fall back to Otsu (a
    single global threshold based on the whole crop's histogram).
    """
    denoised = cv2.medianBlur(gray, 3)
    block_size = 25 if min(gray.shape[:2]) > 30 else 15
    if block_size % 2 == 0:
        block_size += 1

    binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, 10
    )
    black_ratio = 1.0 - (np.count_nonzero(binary) / binary.size)
    if black_ratio < 0.02 or black_ratio > 0.75:
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary  # no morphologyEx here -- keeps Persian dot marks intact


def prepare_crop_for_ocr(crop: np.ndarray, min_height: int = 96) -> np.ndarray:
    """The card itself is already normalized at the whole-image level; this
    just upsizes small crops for readability."""
    height, width = crop.shape[:2]
    if height < min_height:
        scale = min_height / float(height)
        crop = cv2.resize(crop, (int(width * scale), min_height), interpolation=cv2.INTER_CUBIC)
    return crop


def resize_card_for_ocr(image: np.ndarray, min_width: int = 1200, max_width: int = 2000) -> np.ndarray:
    """Upscale a too-small card, downscale a too-large one."""
    height, width = image.shape[:2]
    if width < min_width:
        scale = min_width / float(width)
        return cv2.resize(image, (min_width, int(height * scale)), interpolation=cv2.INTER_CUBIC)
    if width > max_width:
        scale = max_width / float(width)
        return cv2.resize(image, (max_width, int(height * scale)), interpolation=cv2.INTER_AREA)
    return image


# -- Document-normalization tuning knobs -----------------------------------
# Each knob is independent; the meaningful/safe range is noted next to it.

# 1) Halftone (fine scan/print dot) removal, before any contrast boost.
DOC_DOWNSCALE = 2        # [1..4]   downscale factor; higher = stronger dot removal, less detail
DOC_HALFTONE_BLUR = 1    # [0.0..3.0] initial Gaussian sigma; higher = smoother

# 2) Contrast / background curve.
DOC_MID = 130.0          # [110..200] curve midpoint; lower = whiter bg + bolder text,
                         #            higher = finer text (raise if a 4 looks like a 6)
DOC_SLOPE = 0.077        # [0.03..0.20] curve steepness; higher = harsher contrast

# 3) Highlight/shadow clipping and extra boldness.
DOC_WHITE_CLIP = 235     # [200..255] above this -> pure white; lower = cleaner bg
DOC_BLACK_CLIP = 15      # [0..60]    below this -> pure black; lower = less boldening
DOC_BOLD = 0             # [0..100]   extra boldening (dark-side gamma); 0 = off,
                         #            raise (e.g. 20-40) only if text looks too thin

# 4) Final sharpness -- same convention as a typical sharpen slider.
DOC_SHARPNESS = 70       # [-100..100] negative = smoothing, positive = sharpen (unsharp mask)

# 5) Final speckle cleanup.
DOC_SPECKLE = 2          # [0..5] median kernel for leftover speckles; 0 = off


def normalize_card_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    Normalize the whole card into "document mode" for OCR.
    Order: halftone removal -> divide by estimated background -> contrast
    curve -> optional boldening -> clip both ends -> speckle cleanup ->
    sharpen (or smooth).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    # 1) Halftone removal.
    ds = max(1, int(DOC_DOWNSCALE))
    if ds > 1:
        small = cv2.resize(gray, (max(1, w // ds), max(1, h // ds)), interpolation=cv2.INTER_AREA)
        smooth = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    else:
        smooth = gray
    if DOC_HALFTONE_BLUR > 0:
        smooth = cv2.GaussianBlur(smooth, (0, 0), DOC_HALFTONE_BLUR)

    # 2) Background estimation + normalization division.
    k = max(15, int(w * 0.03))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    background = cv2.morphologyEx(smooth, cv2.MORPH_DILATE, kernel)
    background = cv2.GaussianBlur(background, (0, 0), 6)
    norm = smooth.astype(np.float32) * 255.0 / (background.astype(np.float32) + 1.0)
    norm = np.clip(norm, 0, 255).astype(np.uint8)

    # 3) Sigmoid contrast curve.
    lut = (255.0 / (1.0 + np.exp(-DOC_SLOPE * (np.arange(256) - DOC_MID)))).astype(np.uint8)
    out = cv2.LUT(norm, lut)

    # 4) Optional extra boldening (dark-side gamma).
    if DOC_BOLD > 0:
        gamma = 1.0 + (DOC_BOLD / 100.0)  # [1.0 .. 2.0]
        table = ((np.arange(256) / 255.0) ** gamma * 255).astype(np.uint8)
        out = cv2.LUT(out, table)

    # 5) Hard clip both ends of the histogram.
    out = np.where(out > DOC_WHITE_CLIP, 255, out)
    out = np.where(out < DOC_BLACK_CLIP, 0, out).astype(np.uint8)

    # 6) Speckle cleanup.
    if DOC_SPECKLE >= 3:
        ks = int(DOC_SPECKLE)
        if ks % 2 == 0:
            ks += 1
        out = cv2.medianBlur(out, ks)

    # 7) Sharpness (negative = smoothing, positive = sharpen).
    if DOC_SHARPNESS < 0:
        sigma = (abs(DOC_SHARPNESS) / 100.0) * 2.0  # max sigma of 2
        out = cv2.GaussianBlur(out, (0, 0), max(0.1, sigma))
    elif DOC_SHARPNESS > 0:
        amount = (DOC_SHARPNESS / 100.0) * 1.5  # max amount of 1.5
        blurred = cv2.GaussianBlur(out, (0, 0), 1.0)
        out = cv2.addWeighted(out, 1.0 + amount, blurred, -amount, 0)

    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
