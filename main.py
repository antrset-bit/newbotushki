import asyncio
import logging
from pathlib import Path
import tempfile

from fastapi import FastAPI, Request, Response, UploadFile, File, HTTPException
from fastapi.responses import PlainTextResponse
from telegram import Update

from app.bot.handlers import build_application
from app.config import RUN_MODE, PUBLIC_BASE_URL, TELEGRAM_BOT_TOKEN

from app.ocr.ocr import ocr_file
from app.ocr.postprocess import postprocess

logger = logging.getLogger("semantic-bot")

# Единый экземпляр приложения FastAPI
app = FastAPI(title="Semantic Bot + OCR Service", version="1.2.0")

# --- Healthchecks ---
@app.head("/")
async def health_head():
    return Response(status_code=200)

@app.get("/")
async def health_get():
    return {"ok": True}

# --- Telegram Bot (python-telegram-bot) ---
application = build_application()

@app.on_event("startup")
async def _startup():
    # Инициализация PTB
    await application.initialize()

    if RUN_MODE == "polling":
        # Неблокирующий старт, чтобы не ловить таймаут gunicorn
        asyncio.create_task(application.start())
        logger.info("PTB polling started (background)")
    elif RUN_MODE == "webhook":
        if PUBLIC_BASE_URL and TELEGRAM_BOT_TOKEN:
            try:
                wh_url = f"{PUBLIC_BASE_URL}/telegram/{TELEGRAM_BOT_TOKEN}"
                await application.bot.set_webhook(url=wh_url, drop_pending_updates=True)
                logger.info("Webhook set to %s", wh_url)
            except Exception as e:
                logger.warning("Failed to set webhook: %r", e)
        logger.info("PTB initialized for webhook mode")
    else:
        logger.warning("Unknown RUN_MODE=%r; bot will not be started", RUN_MODE)

@app.on_event("shutdown")
async def _shutdown():
    try:
        if RUN_MODE == "polling":
            await application.stop()
            logger.info("PTB polling stopped")
        await application.shutdown()
    finally:
        logger.info("PTB shutdown complete")

# Вебхук-эндпоинт: токен в пути и проверка соответствия настроечному
@app.post("/telegram/{token}")
async def telegram_webhook_token(token: str, request: Request):
    if TELEGRAM_BOT_TOKEN and token != TELEGRAM_BOT_TOKEN:
        return Response(status_code=403)

    data = await request.json()
    update = Update.de_json(data, application.bot)
    await application.process_update(update)
    return {"ok": True}

# --- OCR endpoints ---
@app.post("/ocr", response_class=PlainTextResponse)
async def ocr_endpoint(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    with tempfile.NamedTemporaryFile(delete=True, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp.flush()
        text, _docx = ocr_file(tmp.name)
    return text

@app.post("/ocr/clean", response_class=PlainTextResponse)
async def clean_endpoint(raw_text: str = File(...)):
    # Принимаем уже распознанный текст и применяем постобработку
    return postprocess(raw_text)
