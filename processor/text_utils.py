"""
Common text utilities: digit/Persian text normalization, national ID validation,
Jalali date conversion, and allowlists for OCR.
"""

from __future__ import annotations

import re

try:
    import jdatetime
except ImportError:  # pragma: no cover
    jdatetime = None


# ----------------------------------------------------------------------
# Digits and Characters
# ----------------------------------------------------------------------

FA_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
AR_DIGITS = "٠١٢٣٤٥٦٧٨٩"
EN_DIGITS = "0123456789"

_DIGIT_TABLE = str.maketrans(FA_DIGITS + AR_DIGITS, EN_DIGITS + EN_DIGITS)

# Persian alphabet (includes older/Arabic letter forms that sometimes appear
# in printed fonts instead of their Persian equivalents). This list is used
# as an allowlist for the OCR engine to limit the character search space,
# making it more accurate and faster.
PERSIAN_LETTERS = (
    "ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهیءأإئآيك‌ آأإئؤء يكةة"
)

# For text fields (first name, last name, father's name): only Persian letters + space.
NAME_ALLOWLIST = PERSIAN_LETTERS + "ص "

# For numeric fields (national ID, dates): on the card itself, digits are printed
# with Latin/Western glyphs, but based on real tests, EasyOCR's Persian language
# vocabulary outputs Arabic/Persian glyphs (٠١٢٣٤٥٦٧٨٩ or ۰۱۲۳۴۵۶۷۸۹), not ASCII.
# When the allowlist only contained ASCII digits, the decoder had no valid
# characters for these positions and returned empty results (which is exactly
# why national ID and birth date were never read). Therefore, all three digit
# variants (Latin + Persian + Arabic) are allowed. normalize_digits/only_digits
# convert everything to Latin digits downstream, so no further changes are needed.
DIGIT_ALLOWLIST = EN_DIGITS + FA_DIGITS + AR_DIGITS

# For dates: digits from above + common separators (/ - .)
DATE_ALLOWLIST = DIGIT_ALLOWLIST + "/-."


def normalize_digits(text: str) -> str:
    """Convert Persian/Arabic digits to Latin digits."""
    if not text:
        return ""
    return text.translate(_DIGIT_TABLE)


def normalize_text(text: str) -> str:
    """Normalize Persian characters and remove punctuation for comparison/search."""
    text = normalize_digits(text or "")

    text = text.replace("ي", "ی")
    text = text.replace("ك", "ک")
    text = text.replace("ة", "ه")

    # Replace invisible characters (zero-width space, RTL mark) with a regular space.
    text = re.sub(r"[\u200c\u200f\u200e]", " ", text)

    text = text.replace("،", " ")
    text = text.replace("؛", " ")

    text = re.sub(r"[.,;:()!؟?\"'«»_/\\\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def only_digits(text: str) -> str:
    """Remove all non-digit characters (after normalizing Persian digits)."""
    return re.sub(r"\D", "", normalize_digits(text or ""))


def clean_name(text: str | None) -> str | None:
    """Clean raw OCR value for name fields."""
    if not text:
        return None

    normalized = normalize_text(text)
    # Names should not contain digits; if they do, the crop is probably wrong.
    normalized = re.sub(r"\d+", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    return normalized or None


# ----------------------------------------------------------------------
# Iranian National ID Validation
# ----------------------------------------------------------------------

def is_valid_national_code(code: str | None) -> bool:
    """
    Validate a 10-digit Iranian national ID using the official check-digit algorithm.
    """
    if not code:
        return False

    if not code.isdigit():
        return False

    if len(code) != 10:
        return False

    if len(set(code)) == 1:  # e.g., 0000000000 or 1111111111
        return False

    check_digit = int(code[-1])
    total = sum(int(code[i]) * (10 - i) for i in range(9))
    remainder = total % 11

    if remainder < 2:
        return check_digit == remainder

    return check_digit == 11 - remainder


# Digit pairs that were observed in real tests where the OCR model mistakes
# one digit for another (0 instead of 5). This list is intentionally kept short:
# simulations on thousands of valid national IDs show that adding extra pairs
# (even a common and seemingly harmless pair like 6↔8) significantly reduces
# the "unique correction" rate, because the check digit only provides one unit
# of redundancy; the more candidates are tested, the higher the chance of
# accidental (not real) matching:
#
#   Only 0↔5:            89.5% unique correction, 8.6% ambiguous
#   0↔5 + 6↔8:           72.2% unique correction, 26.0% ambiguous
#   0↔5 + 6↔8 + 3→8:     61.4% unique correction, 24.2% ambiguous
#
# If you observe another common error later (e.g., 3 instead of 8), repeat
# the simulation with that pair to ensure the ambiguity rate does not rise
# too high before adding it to this dictionary.
_CONFUSABLE_DIGITS: dict[str, str] = {
    "0": "5",
    "5": "0",
}


def attempt_national_code_correction(digits: str) -> tuple[str | None, str | None]:
    """
    If a 10-digit national ID has an invalid check digit, attempt to find a
    valid version by swapping a single digit (only among common OCR confusions,
    not arbitrary digits). This uses the check digit as an error detection/
    correction layer — exactly what it was designed for.

    Output: (corrected_code or None, explanatory_message or None)
    - If digits is already valid: (digits, None)
    - If exactly one valid correction is found: (corrected_code, message)
    - If more than one valid correction is found (real ambiguity): (None, warning)
    - If no correction is found: (None, None) — the caller should keep the
      raw value with a normal "invalid" warning.
    """
    if len(digits) != 10 or not digits.isdigit():
        return None, None

    if is_valid_national_code(digits):
        return digits, None

    candidates: set[str] = set()
    for i, ch in enumerate(digits):
        for alt in _CONFUSABLE_DIGITS.get(ch, ""):
            candidate = digits[:i] + alt + digits[i + 1:]
            if is_valid_national_code(candidate):
                candidates.add(candidate)

    if len(candidates) == 1:
        corrected = next(iter(candidates))
        return corrected, (
            f"شماره ملی خوانده‌شده ({digits}) رقم کنترلی نامعتبر داشت؛ با اصلاح یک رقم "
            f"(بر اساس اشتباهات رایج OCR) به {corrected} تصحیح شد. لطفاً صحت آن را تأیید کنید."
        )

    if len(candidates) > 1:
        return None, (
            f"شناسه ملی خوانده شده ({digits}) نامعتبر است و بیش از یک عدد ممکن است "
            f"اصلاح ({', '.join(sorted(candidates))}); لطفا به صورت دستی بررسی شود."
        )

    return None, None


# ----------------------------------------------------------------------
# Jalali Date
# ----------------------------------------------------------------------

# Reasonable Jalali year range for birth/expiry dates on ID cards
# (to reject obviously wrong OCR readings, not to reject edge cases).
_MIN_PLAUSIBLE_YEAR = 1280
_MAX_PLAUSIBLE_YEAR = 1450


def parse_jalali_date(raw_text: str | None) -> tuple[str | None, str | None, float, list[str]]:
    """
    Convert raw OCR text to a Jalali date (and Gregorian if possible).

    Output: (jalali "YYYY/MM/DD", Gregorian ISO or None, confidence, warnings)
    """
    warnings: list[str] = []

    if not raw_text:
        return None, None, 0.0, ["هیچ مقداری برای تاریخ یافت نشد."]

    normalized = normalize_digits(raw_text)

    # Case 1: Explicit separators between year/month/day (most common).
    match = re.search(r"(1[234]\d{2})\D{1,3}(\d{1,2})\D{1,3}(\d{1,2})", normalized)

    year_confidence = 0.85

    if not match:
        # Case 2: OCR missed the separators and only a digit string remains.
        digits_only = only_digits(normalized)

        if len(digits_only) == 8:
            match = re.match(r"(\d{4})(\d{2})(\d{2})", digits_only)
        elif len(digits_only) == 6:
            # The year isn't two digits, but sometimes the first digit (always 1)
            # is dropped; if it starts with 3 or 4, it's likely missing a '1'.
            if digits_only[0] in ("3", "4"):
                digits_only = "1" + digits_only
                match = re.match(r"(\d{4})(\d{1})(\d{1})", digits_only)
        if match:
            year_confidence = 0.55  # Less confident because no separators were present
            warnings.append(
                "جداکننده های تاریخ توسط OCR خوانده نشد. تاریخ استخراج شده از رشته رقم پیوسته"
            )

    if not match:
        return None, None, 0.0, ["تاریخ قابل استخراج نیست."]

    year = int(match.group(1))
    month = int(match.group(2))
    day = int(match.group(3))

    if not (1 <= month <= 12):
        return None, None, 0.0, ["مقدار ماه نامعتبر است."]

    if not (1 <= day <= 31):
        return None, None, 0.0, ["مقدار روز نامعتبر است."]

    if not (_MIN_PLAUSIBLE_YEAR <= year <= _MAX_PLAUSIBLE_YEAR):
        warnings.append(
            f"سال {year} خارج از محدوده معقول است. لطفا به صورت دستی بررسی کنید."
        )
        year_confidence *= 0.5

    jalali = f"{year:04d}/{month:02d}/{day:02d}"
    iso = None

    if jdatetime:
        try:
            iso = jdatetime.date(year, month, day).togregorian().isoformat()
        except ValueError:
            warnings.append(
                "ترکیب سال/ماه/روز در تقویم جلالی نامعتبر است "
                "(e.g., 31 اسفند)."
            )
            return jalali, None, year_confidence * 0.4, warnings
    else:
        warnings.append(
            "بسته jdatetime برای تبدیل جلالی به میلادی نصب نشده است."
        )

    return jalali, iso, year_confidence, warnings


def clean_persian_text(text: str) -> str:
    """
    Clean and normalize Persian text with common OCR corrections.
    """
    if not text:
        return text

    # 1. Remove distracting labels that may have been included in the crop
    labels_to_remove = [
        "نام", "خانوادگی", "پدر", "شمار", "ملی",
        "تاریخ", "تولد", "اعتبار", "ام"
    ]
    for label in labels_to_remove:
        text = text.replace(label, "")

    # 2. Normalize Arabic characters to Persian (EasyOCR often confuses these)
    text = (
        text.replace("أ", "ا")
        .replace("إ", "ا")
        .replace("ي", "ی")
        .replace("ك", "ک")
    )

    # 3. Fix very common EasyOCR errors on ID cards
    # (You can expand this dictionary based on your own tests)
    corrections = {
        "متحمد": "محمد",
        "اسرانی": "ایرانی",
        "روح‌ااا": "روح الله"
    }
    for wrong, right in corrections.items():
        text = text.replace(wrong, right)

    # 4. Remove strange characters: only letters, digits, spaces, and slashes allowed
    text = re.sub(r'[^\w\s/۰-۹0-9]', '', text)

    return text.strip()