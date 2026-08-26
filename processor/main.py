import base64
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

import ocr_engine
from card_layout import draw_debug_overlay
from field_extraction import extract_card_fields
from image_processing import assess_quality, decode_image, prepare_card_image
from photo_extractor import encode_photo_png, extract_face_photo

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Warm up the EasyOCR model here, not on the first user request.
    Without this, the first real user would have to wait for model download/
    loading + first inference (which is always several times slower than
    subsequent ones).
    """
    await run_in_threadpool(ocr_engine.warmup)
    yield


app = FastAPI(
    title="IranID Extractor",
    description="National ID Card Image Processing",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "service": "IranID Vision",
        "version": "3.0.0",
        "phase": "roi-based-extraction",
        "gpu": ocr_engine.is_gpu_active(),
    }


@app.get("/api/v1/camera-stream")
def camera_stream():
    camera_url = os.getenv("CAMERA_URL", "http://192.168.1.103/video")

    try:
        import requests as http_requests

        response = http_requests.get(camera_url, stream=True, timeout=5)
        response.raise_for_status()

        content_type = response.headers.get(
            "Content-Type", "multipart/x-mixed-replace;boundary=frame"
        )

        def iter_stream():
            try:
                for chunk in response.iter_content(chunk_size=4096):
                    yield chunk
            finally:
                response.close()

        return StreamingResponse(
            iter_stream(),
            media_type=content_type,
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Camera not available: {str(e)}")


app.mount(
    "/ui",
    StaticFiles(
        directory=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "front"),
        html=True,
    ),
    name="ui",
)


async def _read_and_validate_upload(file: UploadFile) -> bytes:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file format: {ext}")

    content = await file.read()

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large (max 10MB)")

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    return content


def _process_card_sync(content: bytes) -> dict:
    """
    All CPU-bound and blocking heavy operations (OpenCV + EasyOCR) are executed
    here so they can be offloaded from FastAPI's main event loop using
    run_in_threadpool. Without this, a single OCR request would block the
    entire server for all concurrent requests during its execution.
    """
    image = decode_image(content)
    quality = assess_quality(image)

    card_image, card_detected, card_warning = prepare_card_image(image)
    cv2.imwrite("debug_card_transformed.jpg", card_image)
    card_quality = assess_quality(card_image)

    extraction = extract_card_fields(card_image)

    try:
        photo_result = extract_face_photo(card_image)
    except Exception as error:  
        # Photo extraction is a secondary feature; if it fails for any reason,
        # the entire request (including successfully extracted text fields)
        # should not be discarded.
        photo_result = {
            "image": None,
            "face_detected": False,
            "warning": f"Photo extraction error: {error}"
        }

    photo_bytes = encode_photo_png(photo_result["image"]) if photo_result["image"] is not None else None
    photo_base64 = base64.b64encode(photo_bytes).decode("ascii") if photo_bytes else None

    warnings = list(extraction["warnings"])
    if card_warning:
        warnings.append(card_warning)
    if photo_result.get("warning"):
        warnings.append(photo_result["warning"])

    # ── Capture Quality Gate ─────────────────────────────────────────────
    # A KYC system needs to know where it cannot trust itself: if the input
    # frame is blurry/low-resolution/poorly lit, the extracted values are
    # still displayed as suggestions by default, but human approval is forced.
    # (The quality judgment is based on the "input" image, not card_quality,
    # which is deliberately brightened/whitened after normalization.)
    capture_issues = []
    if quality["is_blurry"]:
        capture_issues.append("Blurry image")
    if not quality["resolution_ok"]:
        capture_issues.append("Low resolution")
    if quality["is_dark"]:
        capture_issues.append("Low light")
    if quality["is_overexposed"]:
        capture_issues.append("Overexposed")

    if capture_issues:
        warnings.append(
            "Input image quality is not optimal ("
            + ", ".join(capture_issues)
            + "); extracted values are displayed as suggestions and must be "
            "verified against the actual card."
        )
        extraction["needs_review"] = True
    else:
        extraction["needs_review"] = bool(extraction.get("needs_review", False))
    # ──────────────────────────────────────────────────────────────────

    return {
        "quality": quality,
        "card_quality": card_quality,
        "card_detected": card_detected,
        "card_size": {"width": card_image.shape[1], "height": card_image.shape[0]},
        "extraction": extraction,
        "photo_base64": photo_base64,
        "photo_face_detected": photo_result["face_detected"],
        "warnings": warnings,
    }


@app.post("/api/v1/process-card")
async def process_card(file: UploadFile = File(...)):
    request_id = str(uuid.uuid4())
    content = await _read_and_validate_upload(file)

    try:
        result = await run_in_threadpool(_process_card_sync, content)
    except ValueError as e:
        # decode_image raises ValueError with a Persian message when the file
        # cannot be read as an image.
        raise HTTPException(status_code=400, detail=f"Cannot decode image: {str(e)}")

    extraction = result["extraction"]

    return {
        "success": True,
        "request_id": request_id,
        "phase": "roi-based-extraction",
        "message": (
            "Image processed and fields extracted based on pre-calibrated regions."
        ),
        "file": {
            "filename": file.filename,
            "content_type": file.content_type,
            "size_bytes": len(content),
        },
        "quality": result["quality"],
        "card_quality": result["card_quality"],
        "card": {
            "detected": result["card_detected"],
            "width": result["card_size"]["width"],
            "height": result["card_size"]["height"],
        },
        "ocr": {
            "engine": "easyocr",
            "mode": "roi-per-field",
            "gpu": ocr_engine.is_gpu_active(),
        },
        "data": extraction["data"],
        "field_confidence": extraction["confidence"],
        "validation": extraction["validation"],
        "missing_fields": extraction["missing_fields"],
        "photo": {
            "available": result["photo_base64"] is not None,
            "face_detected": result["photo_face_detected"],
            "base64_png": result["photo_base64"],
        },
        "warnings": result["warnings"],
        "needs_review": extraction["needs_review"],
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/v1/debug/layout")
async def debug_layout(file: UploadFile = File(...), annotate: bool = True):
    """
    Calibration tool. Two modes:
    - annotate=true (default): Returns the processed card image with current
      ROI boxes overlaid (for quick visual inspection).
    - annotate=false: Returns the processed image without any overlays —
      this is the exact image you should load in the browser-based
      calibration tool, because it represents the exact pixels that
      FIELD_ROIS are applied to (after card detection/cropping and resize,
      not the raw uploaded file).
    """
    content = await _read_and_validate_upload(file)

    def _build_debug_image() -> bytes:
        image = decode_image(content)
        card_image, _, _ = prepare_card_image(image)
        output_image = draw_debug_overlay(card_image) if annotate else card_image
        success, buffer = cv2.imencode(".png", output_image)
        if not success:
            raise ValueError("Unable to build debug image.")
        return buffer.tobytes()

    try:
        png_bytes = await run_in_threadpool(_build_debug_image)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(e))

    return Response(content=png_bytes, media_type="image/png")