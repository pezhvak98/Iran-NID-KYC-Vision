# ── IranID Vision: extractor کارت ملی ──
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=2

WORKDIR /app

# وابستگی‌های سیستمی OpenCV (headless)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# torch نسخهٔ CPU (قبل از easyocr تا wheel سنگین CUDA نیاید)
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
    torch torchvision

COPY processor/requirements.txt .
RUN pip install -r requirements.txt

# دانلود مدل‌های EasyOCR در زمان build → شروع آنی بدون اینترنت
RUN python -c "import easyocr; easyocr.Reader(['fa','en'], gpu=False, verbose=False)"

COPY processor/ ./processor/
COPY front/ ./front/

WORKDIR /app/processor
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]