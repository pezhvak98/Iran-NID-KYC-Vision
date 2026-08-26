# Iran-NID-KYC-Vision

**Automated Iranian National ID Card Extraction with Human-in-the-Loop Review**

Iran-NID-KYC-Vision is a CPU-friendly pipeline for extracting structured data from Iranian national ID cards. It pairs classical computer vision for reliable card detection with a dual-engine OCR system (EasyOCR + font-aware template matching), and ships with a web-based operator review interface so every result is either automatically validated or explicitly flagged for human confirmation.

The system is built for KYC (Know Your Customer) contexts, where silent misreads are unacceptable: ambiguous or low-confidence extractions are surfaced to a human operator rather than guessed.

---

## Table of Contents

- [Key Features](#key-features)
- [Architecture & Pipeline](#architecture--pipeline)
- [Quick Start (Docker)](#quick-start-docker)
- [Local Development](#local-development)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [ROI Calibration Tool](#roi-calibration-tool-roi_calibratorhtml)
- [Design Decisions](#design-decisions)
- [Roadmap & Limitations](#roadmap--limitations)

---

## Key Features

- **Robust Card Detection** — Multi-strategy classical CV (line-free print mapping, color masking, skin/hand removal) that performs reliably against cluttered backgrounds such as desks, technical drawings, or ruled paper.
- **Dual-Engine OCR**
  - *EasyOCR* for open-vocabulary Persian names.
  - *B-Yekan Template Matcher* for digits — CPU-only, millisecond latency, and immune to ۰/۵ confusion by design.
- **Domain Constraints as Arbiters** — National ID checksum validation, Jalali calendar range checks, and explicit candidate generation for ambiguous dates prevent silent misreads.
- **Human-in-the-Loop Workflow** — A web UI with alignment templates, editable review fields, the original source image for reference, and a final structured JSON export.
- **Capture Quality Gate** — Automatic blur/brightness/resolution assessment. Low-quality captures are flagged with `needs_review=true` rather than processed blindly.
- **Dockerized Deployment** — Single-command startup with pre-baked EasyOCR models for near-instant cold start on repeat runs.

---

## Architecture & Pipeline

1. **Capture** — MJPEG IP-camera proxy or file upload via the web UI.
2. **Quality Gate** — Assesses each input frame for blur, brightness, and resolution.
3. **Enhancement** — Quality-aware gamma correction, CLAHE, and contrast adjustment.
4. **Detection** — Morphological line removal, geometry validation, sub-pixel corner refinement, and a fixed-aspect perspective warp (1100×694).
5. **Orientation Fix** — A content-based digit-row readability test detects and corrects 180° rotation.
6. **Normalization** — Halftone removal, background division, and a tunable sigmoid contrast curve (`DOC_*` parameters).
7. **ROI Extraction** — Calibrated, per-field bounding boxes applied to the normalized card.
8. **Extraction & Validation** — Dual-engine OCR combined with checksum and calendar arbiters, plus 0/5 repair candidates for ambiguous digits.
9. **Review & Output** — The operator approves or corrects each field before a structured JSON record is exported.

A full breakdown of each stage, along with the reasoning behind key design choices, is documented in [`ARCHITECTURE.md`](./ARCHITECTURE.md).

---

## Quick Start (Docker)

The fastest way to run the full stack (backend, frontend, and OCR models):

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd Iran-NID-KYC-Vision

# 2. Build and run
docker compose up --build

# 3. Open the UI
# Frontend: http://127.0.0.1:8000/ui
# API Docs: http://127.0.0.1:8000/docs
```

> **Note:** The first build downloads PyTorch (CPU build) and EasyOCR models (~300 MB). Subsequent starts are near-instant.

---

## Local Development

If you prefer to run the project without Docker:

**Prerequisites**
- Python 3.10+
- FFmpeg (optional — required only for camera stream proxying)

**Installation**

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install PyTorch (CPU build recommended for a lightweight setup)
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu

# 3. Install project dependencies
pip install -r processor/requirements.txt

# 4. Pre-download EasyOCR models (optional — otherwise downloaded on first run)
python -c "import easyocr; easyocr.Reader(['fa','en'], gpu=False)"
```

**Running the Server**

```bash
cd processor
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Debugging ROI Layout Without the UI**

```bash
# Linux / macOS
curl -X POST -F "file=@[input-image.ext]" "http://127.0.0.1:8000/api/v1/debug/layout?annotate=true" --output [output-image.ext]

# Windows
curl.exe -X POST -F "file=@[input-image.ext]" "http://127.0.0.1:8000/api/v1/debug/layout?annotate=true" --output [output-image.ext]
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/process-card` | Upload an image (`multipart/form-data`) and extract structured card data. |
| `GET`  | `/api/v1/camera-stream` | Proxied MJPEG stream from the configured IP camera. |
| `GET`  | `/ui` | Web-based operator review interface. |
| `GET`  | `/docs` | Interactive OpenAPI (Swagger) documentation. |

### Example Response — `/api/v1/process-card`

```json
{
  "success": true,
  "data": {
    "national_id": "4890453032",
    "first_name": "محمد",
    "last_name": "ایرانی",
    "father_name": "روح الله",
    "birth_date": { "jalali": "1400/01/01", "iso": "2021-03-21" },
    "expiry_date": { "jalali": "1402/01/01", "iso": "2023-03-21" }
  },
  "validation": { "national_id": true },
  "needs_review": false,
  "warnings": []
}
```

---

## Configuration

**Document Normalization Knobs** (`processor/image_processing.py`)

Tune these constants to adapt to different lighting conditions or card print batches:

| Parameter | Purpose |
|---|---|
| `DOC_DOWNSCALE` | Halftone removal scale (default: `2`). |
| `DOC_MID` / `DOC_SLOPE` | Center and steepness of the normalization sigmoid curve. |
| `DOC_WHITE_CLIP` / `DOC_BLACK_CLIP` | Output clipping bounds. |
| `DOC_SHARPNESS` | Unsharp mask strength. |

**Camera Stream**

Configure your IP camera URL via `DEFAULT_CAMERA_URL` in `processor/main.py`, or set it directly from the UI input field.

---

## ROI Calibration Tool (`roi_calibrator.html`)

A standalone, browser-based utility for precise per-field calibration — no backend required.

- **Load** — Open the final cropped/normalized card image directly in the HTML file.
- **Calibrate** — Visually draw and measure the exact bounding box for each field (National ID, name, dates, etc.).
- **Export** — Copy the generated normalized coordinates into `processor/card_layout.py` to update the extraction ROIs.

---

## Design Decisions

| Decision | Reasoning |
|----------|-----------|
| Template matching for digits | CPU-only, millisecond latency, shape-based matching — ۰/۵ confusion is physically impossible. |
| Domain constraints as arbiters | CV/OCR output only governs when validated by checksum or calendar rules, preventing hallucinated fields. |
| Fixed-size perspective warp | ROI calibration is aspect-sensitive; a fixed 1100×694 warp eliminates alignment drift. |
| Capture quality gate | A blurry capture triggers mandatory human review instead of a silent, unreliable result. |
| Explicit candidates | Operators see genuine ambiguity (e.g., 1380 vs. 1385) and choose — the system never guesses silently. |

---

## Roadmap & Limitations

- **Name Character Confusion** — On severely blurry captures, single-character confusions (e.g., ص/م) can occur. Mitigated today by the quality gate; the planned production safeguard is a **Civil Registry Lookup (استعلام ثبت احوال)** cross-check by national ID.
- **GPU Acceleration** — Set `gpu=True` in `processor/ocr_engine.py` to move EasyOCR onto CUDA.
- **Face Detection** — Currently implemented with OpenCV Haar cascades; a lightweight deep-learning face detector is a candidate future upgrade.
