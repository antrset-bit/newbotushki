# ===== Base image ============================================================
FROM python:3.11-slim

# ===== Env ===================================================================
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # OCR настройки по умолчанию — можно переопределить через Render env vars
    PDF_COMPRESS=1 \
    OCR_DPI=220 \
    OCR_MAX_PAGES=0 \
    TESS_LANG=rus+eng \
    # Tesseract ищет языковые данные здесь
    TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata

# ===== System deps (poppler + tesseract + utils) =============================
# - poppler-utils: pdftoppm/pdftocairo для pdf2image
# - tesseract-ocr + языки rus/eng для pytesseract
# - build-essential + gcc: на случай сборки колёс
# - fonts-dejavu: базовые шрифты (иногда влияют на рендер в poppler)
# - libglib2.0-0, libjpeg, zlib: частые зависимости Pillow/PyMuPDF
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential gcc \
    poppler-utils \
    tesseract-ocr tesseract-ocr-rus tesseract-ocr-eng \
    fonts-dejavu \
    libglib2.0-0 libjpeg62-turbo zlib1g \
    ca-certificates curl \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# ===== Workdir ===============================================================
WORKDIR /app

# ===== Python deps ===========================================================
# Скопируйте ваш requirements.txt рядом с Dockerfile перед сборкой
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
  && pip install -r /app/requirements.txt

# ===== App code ==============================================================
# Предполагается структура проекта с пакетом `app/` и файлом `app/main.py` (FastAPI)
COPY . /app

# ===== Non-root user (безопаснее) ===========================================
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# ===== Expose port for Render ===============================================
# Render обычно прокидывает переменную PORT — uvicorn её использует ниже.
EXPOSE 10000

# ===== Start command =========================================================
# Если ваш FastAPI-приложение — это объект "app" в app/main.py:
# Render задаёт переменную PORT, поэтому используем её в командe.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
