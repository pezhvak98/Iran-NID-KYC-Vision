
from __future__ import annotations

import cv2
import re
from image_processing import crop_roi, prepare_crop_for_ocr, binarize_for_ocr
import ocr_engine
from card_layout import FIELD_LABELS, FIELD_ROIS
from itertools import combinations
from text_utils import (
    attempt_national_code_correction,
    clean_name,
    is_valid_national_code,
    normalize_text,
    only_digits,
    parse_jalali_date,
)

LOW_CONFIDENCE_THRESHOLD = 0.35

_REQUIRED_FIELDS = ("national_id", "first_name", "last_name", "father_name", "birth_date", "expiry_date")

_LABEL_PATTERNS = {
    "national_id": [r"شماره\s*ملی", r"کد\s*ملی"],
    "first_name": [r"^نام$"],
    "last_name": [r"نام\s*خانوادگی", r"خانوادگی"],
    "father_name": [r"نام\s*پدر", r"^پدر$"],
    "birth_date": [r"تاریخ\s*تولد", r"تولد"],
    "expiry_date": [r"پایان\s*اعتبار", r"تاریخ\s*انقضا", r"اعتبار", r"انقضا"],
}


def _empty_data() -> dict:
    return {
        "national_id": None,
        "first_name": None,
        "last_name": None,
        "father_name": None,
        "birth_date": {"jalali": None, "iso": None, "raw": None},
        "expiry_date": {"jalali": None, "iso": None, "raw": None},
    }


def _ensure_date_dict(value):
    """Returns to the standard dictionary if for any reason the date value is a string."""
    if isinstance(value, dict):
        return value
    return {
        "jalali": None,
        "iso": None,
        "raw": value if isinstance(value, str) else None,
    }



_LABEL_NOISE_RE = re.compile(
    r"([نتد]ام|نام|خانواد|وادگ|پدر|پذر|ملی|شماره|تولد|تاریخ|اعتبار|پایان|جمهوری|اسلامی|ایران|کارت)"
)


def _clean_name_value(raw: str) -> str | None:
    """Remove leaked tags and 1-2 character noise chunks from the name value."""
    cleaned = clean_name(raw)
    if not cleaned:
        return None

    tokens = [t for t in cleaned.split() if not _LABEL_NOISE_RE.search(t)]

    if len(tokens) > 1:
        tokens = [t for t in tokens if len(t) > 2]

    return " ".join(tokens) if tokens else None


def _repair_national_05(candidate: str) -> str | None:
    """ 
    0↔5 combination search (up to 3 digits) for national number. 
    It only returns when exactly one valid candidate is found, never 
    Do not choose a chance between two valid guesses. 
    """
    if not candidate or len(candidate) != 10 or not candidate.isdigit():
        return None

    positions = [i for i, ch in enumerate(candidate) if ch in "05"]
    if not positions or len(positions) > 4:
        return None

    valids = set()
    for r in range(1, min(len(positions), 3) + 1):
        for combo in combinations(positions, r):
            chars = list(candidate)
            for i in combo:
                chars[i] = "0" if chars[i] == "5" else "5"
            alt = "".join(chars)
            if is_valid_national_code(alt):
                valids.add(alt)

    if len(valids) == 1:
        return valids.pop()
    return None




def _swap05_variants(s: str) -> list[str]:
    out = [s]
    for i, ch in enumerate(s):
        if ch == "5":
            out.append(s[:i] + "0" + s[i + 1:])
        elif ch == "0":
            out.append(s[:i] + "5" + s[i + 1:])
    return out


def _parse_date_robust(raw: str):
    """outpus: (jalali, iso, conf, warnings, candidates)"""
    jalali, iso, conf, warns = parse_jalali_date(raw)
    if jalali:
        year = int(jalali.split("/")[0])
        if 1250 <= year <= 1420:
            return jalali, iso, conf, warns, []

    parts = [only_digits(p) for p in re.split(r"[^0-9۰-۹٠-٩]+", raw or "") if only_digits(p)]
    candidates: set[str] = set()

    ordered = None
    if len(parts) == 3 and len(parts[0]) == 4:
        ordered = (parts[0], parts[1], parts[2])      
    elif len(parts) == 3 and len(parts[2]) == 4:
        ordered = (parts[2], parts[1], parts[0])     

    if ordered:
        y, mo, d = ordered
        for y2 in _swap05_variants(y):
            if not (1250 <= int(y2) <= 1420):
                continue
            for mo2 in _swap05_variants(mo):
                if not (1 <= int(mo2) <= 12):
                    continue
                for d2 in _swap05_variants(d):
                    if 1 <= int(d2) <= 31:
                        candidates.add(f"{y2}/{mo2}/{d2}")
    else:
        digits = only_digits(raw)
        for m in re.finditer(r"(1[234]\d{2})(\d{2})(\d{2})", digits):
            y, mo, d = m.groups()
            for y2 in _swap05_variants(y):
                if not (1250 <= int(y2) <= 1420):
                    continue
                for mo2 in _swap05_variants(mo):
                    if not (1 <= int(mo2) <= 12):
                        continue
                    for d2 in _swap05_variants(d):
                        if 1 <= int(d2) <= 31:
                            candidates.add(f"{y2}/{mo2}/{d2}")

    if len(candidates) == 1:
        chosen = next(iter(candidates))
        return chosen, None, 0.6, warns + ["تاریخ با تعمیر اختلال ۰/۵ بازسازی شد."], []
    if not candidates:
        return None, None, 0.0, warns, []
    return None, None, 0.0, warns, sorted(candidates)







def _enumerate_date_candidates(raw: str) -> set[str]:
    """Returns all possible valid interpretations (month/day order + fix 0.5)."""
    parts = [only_digits(p) for p in re.split(r"[^\d]+", raw or "") if only_digits(p)]
    parts = [p for p in parts if 1 <= len(p) <= 4]

    candidates: set[str] = set()
    year_parts = [p for p in parts if len(p) == 4]
    rest = [p for p in parts if len(p) <= 2]

    if year_parts and len(rest) >= 2:
        for y in year_parts[:2]:
            for y2 in _swap05_variants(y):
                if not (1250 <= int(y2) <= 1420):
                    continue
                r2 = rest[:2]
                for mo_src, d_src in ((r2[0], r2[1]), (r2[1], r2[0])):
                    for mo2 in _swap05_variants(mo_src):
                        if not (1 <= int(mo2) <= 12):
                            continue
                        for d2 in _swap05_variants(d_src):
                            if 1 <= int(d2) <= 31:
                                candidates.add(f"{y2}/{mo2}/{d2}")
    else:
        digits = only_digits(raw)
        for m in re.finditer(r"(1[234]\d{2})(\d{2})(\d{2})", digits):
            y, mo, d = m.groups()
            for y2 in _swap05_variants(y):
                if not (1250 <= int(y2) <= 1420):
                    continue
                for mo2 in _swap05_variants(mo):
                    if not (1 <= int(mo2) <= 12):
                        continue
                    for d2 in _swap05_variants(d):
                        if 1 <= int(d2) <= 31:
                            candidates.add(f"{y2}/{mo2}/{d2}")

    return candidates


def _parse_date_with_repair(raw: str):

    candidates = _enumerate_date_candidates(raw)

    jalali, iso, parse_conf, date_warnings = parse_jalali_date(raw)
    if jalali:
        year = int(jalali.split("/")[0])
        if 1250 <= year <= 1420:
            candidates.add(jalali)
        else:
            jalali = None

    if len(candidates) == 1:
        chosen = next(iter(candidates))
        notes = list(date_warnings)
        if chosen != jalali:
            notes.append("تاریخ با تعمیر اختلال ۰/۵ بازسازی شد.")
        return chosen, None, 0.6, notes, []

    if len(candidates) == 0:
        return None, None, 0.0, date_warnings, []

    return None, None, 0.0, date_warnings, sorted(candidates)






def _apply_value(data: dict, confidence: dict, warnings: list, field: str, raw: str | None, conf: float) -> None:
    """Writes the OCRed raw value depending on the type of parsing/clearing field and writes it in data."""
    if not raw:
        return

    if field == "national_id":
        digits = only_digits(raw)
        match = re.search(r"\d{10}", digits)
        candidate = match.group() if match else (digits if len(digits) == 10 else None)

        if candidate:
            note = None
            if not is_valid_national_code(candidate):
                corrected, note = attempt_national_code_correction(candidate)
                if corrected:
                    candidate = corrected
                    conf *= 0.9  # تصحیح خودکار
                else:
                    fixed05 = _repair_national_05(candidate)
                    if fixed05:
                        candidate = fixed05
                        conf *= 0.85
                        note = "شماره ملی با تعمیر اختلال ۰/۵ بازسازی شد؛ لطفاً با کارت تطبیق دهید."
            if note:
                warnings.append(note)
            data["national_id"] = candidate
            confidence["national_id"] = round(conf, 3)
        elif digits:
            data["national_id"] = digits
            confidence["national_id"] = round(conf * 0.4, 3)
            warnings.append("شماره ملی کامل (۱۰ رقمی) خوانده نشد؛ مقدار ناقص برای بازبینی نمایش داده شد.")

    elif field in ("birth_date", "expiry_date"):
        jalali, iso, parse_conf, date_warnings, date_candidates = _parse_date_robust(raw)
        if jalali:
            data[field] = {"jalali": jalali, "iso": iso, "raw": raw}
            confidence[field] = round(conf * parse_conf, 3)
        else:
            entry = {"jalali": None, "iso": None, "raw": raw}
            if date_candidates:
                entry["candidates"] = date_candidates
                warnings.append(
                    f"{FIELD_LABELS[field]}: {len(date_candidates)} تفسیر معتبر؛ نیاز به انتخاب دستی: "
                    + "، ".join(date_candidates)
                )
            else:
                warnings.append(f"{FIELD_LABELS[field]}: تاریخ قابل استخراج نیست.")
            data[field] = entry
        for w in date_warnings:
            warnings.append(f"{FIELD_LABELS[field]}: {w}")

    else:  # first_name / last_name / father_name
        cleaned = _clean_name_value(raw)
        if cleaned:
            data[field] = cleaned
            confidence[field] = round(conf, 3)


def _extract_via_roi(card_image, data: dict, confidence: dict, warnings: list) -> None:
    for field, config in FIELD_ROIS.items():
        if field == "photo":
            continue

        crop = crop_roi(card_image, config["box"])
        if crop is None:
            warnings.append(f"ناحیهٔ «{FIELD_LABELS[field]}» قابل برش نبود.")
            continue

        # --- Debug: Saving the cut of each field ---
        cv2.imwrite(f"debug_crop_{field}.jpg", crop)
        # ----------------------------------

        def _get_ocr_result(prepared_img):
            if config["kind"] == "number":
                return ocr_engine.read_digits(prepared_img)
            elif config["kind"] == "date":
                return ocr_engine.read_date(prepared_img)
            else:
                return ocr_engine.read_name(prepared_img)

        best_raw, best_conf = None, 0.0
        readings: list[tuple[str, float, str]] = [] 

        def _try(prepared_img, strategy_name: str) -> None:
            nonlocal best_raw, best_conf
            raw, conf = _get_ocr_result(prepared_img)
            if raw:
                readings.append((raw, conf, strategy_name))
                if conf > best_conf:
                    best_raw, best_conf = raw, conf

        # ── Attempt 1: normalized crop (the card is normalized to the whole image level) ──
        _try(prepare_crop_for_ocr(crop), "soft")

        # ── Attempt 2: Second comment with real binary ── 
        # For figures/dates we always compare (main source of error 0.5 i.e 
        # is middle gray); For the text only when the first reading was weak.
        if config["kind"] in ("number", "date") or best_conf < 0.85:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape[:2]
            if h < 120:  # بزرگ‌نمایی قبل از باینری: جزئیات ریز حروف حفظ می‌شوند
                scale = 120.0 / h
                gray = cv2.resize(gray, (int(w * scale), 120), interpolation=cv2.INTER_CUBIC)
            binary = binarize_for_ocr(gray)
            _try(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), "binary")

        # Attempt 3
        if best_conf < 0.30:
            _try(crop, "raw")

        if not best_raw:
            warnings.append(f"در ناحیهٔ «{FIELD_LABELS[field]}» متنی خوانده نشد.")
            continue

        # Honest ambiguity detection: if two independent paths, different reading but both
        # Give it with confidence, it means it is a guess → more reliable choice + revision flag
        confident_others = [
            (raw, conf, name)
            for (raw, conf, name) in readings
            if raw != best_raw and conf > 0.5
        ]
        if confident_others:
            alt = max(confident_others, key=lambda item: item[1])
            warnings.append(
                f"«{FIELD_LABELS[field]}»: دو خوانش متفاوت از دو مسیر پیش‌پردازش "
                f"(«{best_raw}» در برابر «{alt[0]}»)؛ خوانشِ با اعتماد بالاتر انتخاب شد "
                "ولی بازبینی توصیه می‌شود."
            )

        _apply_value(data, confidence, warnings, field, best_raw, best_conf)


def _box_metrics(box: list) -> tuple[float, float, float, float]:
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return (sum(xs) / len(xs), sum(ys) / len(ys), max(xs) - min(xs), max(ys) - min(ys))


def _matches_any_label(normalized_text: str) -> bool:
    for patterns in _LABEL_PATTERNS.values():
        for pattern in patterns:
            if re.search(pattern, normalized_text):
                return True
    return False


def _recover_missing_fields(card_image, data: dict, confidence: dict) -> list[str]:
    
    missing = []
    for field in _REQUIRED_FIELDS:
        if field in ("birth_date", "expiry_date"):
            data[field] = _ensure_date_dict(data[field])
            if not data[field].get("jalali") or confidence.get(field, 0.0) < LOW_CONFIDENCE_THRESHOLD:
                missing.append(field)
        else:
            if not data.get(field) or confidence.get(field, 0.0) < LOW_CONFIDENCE_THRESHOLD:
                missing.append(field)

    if not missing:
        return []

    warnings: list[str] = []
    lines = ocr_engine.read_free_text(card_image)

    if not lines:
        return [f"مسیر پشتیبان (OCR کامل تصویر) نتیجه‌ای نداد؛ فیلدهای ناقص: "
                f"{', '.join(FIELD_LABELS.get(f, f) for f in missing)}"]

    
    label_lines: dict[str, dict] = {}
    for line in lines:
        text = (line.get("text") or "").strip()
        if not text or not line.get("box"):
            continue
        normalized = normalize_text(text)
        for field in missing:
            for pattern in _LABEL_PATTERNS.get(field, []):
                if re.search(pattern, normalized):
                    if field not in label_lines or line["confidence"] > label_lines[field]["confidence"]:
                        label_lines[field] = line
                    break

    for field in missing:
        label_line = label_lines.get(field)
        if not label_line:
            warnings.append(f"با روش پشتیبان هم برچسب «{FIELD_LABELS.get(field, field)}» پیدا نشد.")
            continue

        
        is_date_field = field in ("birth_date", "expiry_date")

        lx, ly, _, lh = _box_metrics(label_line["box"])
        candidates = []

        for line in lines:
            if line is label_line or not line.get("box"):
                continue

            text = (line.get("text") or "").strip()
            if not text or _matches_any_label(normalize_text(text)):
                continue

            # Type-aware filter: date must contain digits, name must not be numeric
            text_digits = only_digits(text)
            if is_date_field and len(text_digits) < 4:
                continue
            if not is_date_field and len(text_digits) > 2:
                continue

            cx, cy, _, _ = _box_metrics(line["box"])
            dy = abs(cy - ly)
            vertical_limit = max(lh * 2.5, 45)
            if dy > vertical_limit:
                continue

            score = float(line.get("confidence", 0.0)) - (dy / vertical_limit) * 0.3
            if cx < lx:  
                score += 0.15

            candidates.append((score, text, float(line.get("confidence", 0.0))))

        if not candidates:
            warnings.append(f"با روش پشتیبان هم مقداری برای «{FIELD_LABELS.get(field, field)}» پیدا نشد.")
            continue

        candidates.sort(key=lambda item: item[0], reverse=True)
        _, best_text, best_conf = candidates[0]
        _apply_value(data, confidence, warnings, field, best_text, best_conf * 0.85)

    return warnings


def extract_card_fields(card_image) -> dict:
   
    data = _empty_data()
    confidence = {field: 0.0 for field in _REQUIRED_FIELDS}
    warnings: list[str] = []

    _extract_via_roi(card_image, data, confidence, warnings)
    warnings.extend(_recover_missing_fields(card_image, data, confidence))

    national_id = data.get("national_id")
    validation = {
        "national_id": is_valid_national_code(national_id) if national_id else False,
    }

    if national_id and len(national_id) == 10 and not validation["national_id"]:
        warnings.append("شماره ملی پیدا شد اما رقم کنترلی آن معتبر نیست؛ لطفاً به‌صورت دستی بررسی شود.")

        # تعمیر دفاعی مقادیر تاریخ
    for field in ("birth_date", "expiry_date"):
        data[field] = _ensure_date_dict(data[field])

    missing_fields = []
    
    missing_fields = []
    for field in ("national_id", "first_name", "last_name", "father_name"):
        if not data.get(field):
            missing_fields.append(field)
    for field in ("birth_date", "expiry_date"):
        if not data[field].get("jalali"):
            missing_fields.append(field)

    needs_review = bool(missing_fields) or not validation["national_id"]

    return {
        "data": data,
        "confidence": {k: round(v, 3) for k, v in confidence.items()},
        "validation": validation,
        "warnings": warnings,
        "missing_fields": missing_fields,
        "needs_review": needs_review,
    }


