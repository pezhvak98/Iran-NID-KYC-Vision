# Iran-NID-KYC-Vision — Architecture

This document describes the end-to-end processing pipeline behind Iran-NID-KYC-Vision, the reasoning behind its key design decisions, and the API surface exposed by the service.

**Stack:** FastAPI · OpenCV · EasyOCR · B-Yekan Digit Matcher · Vanilla JS · Docker

The system extracts structured data from Iranian national ID cards and pairs it with a human-in-the-loop review workflow, so that every output is either automatically validated against domain rules or explicitly routed to an operator.

---

## Pipeline Overview

| Stage | Description |
|---|---|
| 1. Capture | Image acquired via MJPEG camera proxy or direct upload, guided by a UI alignment template. |
| 2. Quality Gate | Blur and brightness are assessed on the raw input; low-quality captures are flagged with `needs_review=true`. |
| 3. Enhancement | Quality-aware gamma correction and CLAHE improve contrast ahead of detection. |
| 4. Detection (Classical CV) | A line-free print map (morphological opening), color masking, and skin masking (for hand removal) feed geometry validation, sub-pixel corner detection, and a fixed-size perspective warp (1100×694). |
| 5. Orientation | A digit-row readability test (template matching) detects and corrects 180° rotation. |
| 6. Normalization | Halftone removal, background division, and a sigmoid contrast curve (tunable via `DOC_*` parameters) prepare the image for extraction. |
| 7. ROI Extraction | Per-field bounding boxes, calibrated against the normalized card layout, isolate each data field. |
| 8. Dual-Engine OCR | EasyOCR handles Persian names; template matching handles digits. Results are arbitrated by checksum validation, Jalali calendar range checks, and 0/5 repair candidates. |
| 9. Review | Extracted fields, source image, and any ambiguous candidates are presented to an operator, who approves or edits before final JSON export. |

---

## Key Design Decisions

| Decision | Reasoning |
|---|---|
| Template matching for digits | CPU-only, millisecond latency, and shape-based — eliminates 0/5 confusion by construction rather than by post-hoc correction. |
| Domain constraints as arbiters | Computer vision and OCR output are only trusted once validated by checksum or calendar logic, preventing hallucinated or unchecked fields. |
| Fixed-size perspective warp | ROI calibration is aspect-sensitive; standardizing on a 1100×694 warp removes alignment error as a variable. |
| Capture quality gate | A blurry or poorly lit capture routes to mandatory human review rather than producing an unreliable silent result. |
| Explicit candidates | Genuine ambiguity (e.g., a digit that could be 0 or 5, or a year that could be 1380 or 1385) is surfaced to the operator rather than resolved by silent guessing. |

---

## API

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/process-card` | Upload an image and extract structured card data. |
| `GET` | `/api/v1/camera-stream` | Proxied MJPEG stream from the configured IP camera. |
| `GET` | `/ui` | Web-based operator review interface. |
| `GET` | `/docs` | Interactive OpenAPI (Swagger) documentation. |

---

## Roadmap

- **Name character confusion** on blurry captures (e.g., ص/م) — mitigated by the quality gate, with a civil registry lookup planned as a production-grade cross-check.
- **GPU acceleration** — EasyOCR can switch to CUDA by setting `gpu=True`.

---

## Running the Project

```bash
docker compose up --build
# UI: http://127.0.0.1:8000/ui
```

For local (non-Docker) development, dependency installation, and debugging endpoints, see [`README.md`](./README.md).
