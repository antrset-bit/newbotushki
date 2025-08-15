# app/bot/handlers.py — clean build (fast /pdf_to_docx, dedup, safe timeouts)
from __future__ import annotations

import os, re, time, asyncio, logging, difflib
from pathlib import Path
from typing import Any, Optional

from telegram import Update, Document, ReplyKeyboardMarkup, InputFile
from telegram.ext import MessageHandler, CommandHandler, filters, ApplicationBuilder
from telegram.request import HTTPXRequest

from app.config import (
    TELEGRAM_BOT_TOKEN, DOC_FOLDER, INDEX_FILE, TEXTS_FILE, MANIFEST_FILE,
    DAILY_FREE_LIMIT, TELEGRAM_MSG_LIMIT, RUN_MODE, PUBLIC_BASE_URL, RETRIEVAL_K
)
from app.utils.usage import is_admin, get_usage, inc_usage
from app.utils.files import load_manifest, save_manifest, sha256_file
from app.services.indexing import index_file, retrieve_chunks
from app.services.generation import generate_direct_ai_answer, generate_answer_with_gemini
from app.services.tm import tm_process_search, ROW_MATCH_EXPERTISE, ROW_MATCH_REGISTERED, ROW_MATCH_KW
from app.services.extract import extract_text_from_pdf, extract_text_from_docx, extract_text_from_txt, extract_text_from_doc
from app.services.chunking import smart_split_text
from app.services.embeddings import get_embedding
from app.ocr.postprocess import postprocess

from docx import Document as DocxDocument
from docx.enum.text import WD_COLOR_INDEX

import fitz  # PyMuPDF
from pdf2image import convert_from_path
import pytesseract
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger("semantic-bot")

AI_LABEL   = "🤖 AI-чат"
DOCS_LABEL = "Часто задаваемые вопросы"
WORK_LABEL = "🗂️ Работа с документами"
TM_LABEL   = "🏷️ Товарные знаки"
MAIN_KB    = ReplyKeyboardMarkup([[AI_LABEL, WORK_LABEL, TM_LABEL], [DOCS_LABEL]], resize_keyboard=True)

TM_MODE = "tm"

# ---- Settings ----
FORCE_OCR = os.getenv("FORCE_OCR", "0") == "1"
OCR_DPI = int(os.getenv("OCR_DPI", "220"))
PDF_COMPRESS = os.getenv("PDF_COMPRESS", "1") == "1"
OCR_MAX_PAGES = int(os.getenv("OCR_MAX_PAGES", "0"))  # 0 = all
POPPLER_PATH = os.getenv("POPPLER_PATH", "").strip() or None
TESS_LANG = os.getenv("TESS_LANG", "rus")
TESS_CONFIG = os.getenv("TESS_CONFIG", "--oem 1 --psm 6 -c preserve_interword_spaces=1")
OCR_OVERALL_TIMEOUT = int(os.getenv("OCR_OVERALL_TIMEOUT", "480"))  # seconds

PDF_DEDUP_TTL = 120
INDEX_DEDUP_TTL = 600
CMD_DEDUP_TTL = 180

def _split_for_telegram(text: str, max_len: int = TELEGRAM_MSG_LIMIT - 200) -> list[str]:
    parts, buf, cur = [], [], 0
    for p in text.replace("\r\n", "\n").split("\n\n"):
        p = p.strip()
        if not p:
            chunk = "\n\n".join(buf).strip()
            if chunk: parts.append(chunk)
            buf, cur = [], 0
            continue
        need = len(p) + (2 if cur > 0 else 0)
        if cur + need <= max_len:
            buf.append(p); cur += need
        else:
            chunk = "\n\n".join(buf).strip()
            if chunk: parts.append(chunk)
            buf, cur = [], 0
            while len(p) > max_len:
                parts.append(p[:max_len]); p = p[max_len:]
            if p:
                buf, cur = [p], len(p)
    chunk = "\n\n".join(buf).strip()
    if chunk: parts.append(chunk)
    return parts

async def send_long(update: Update, text: str):
    text = (text or "").strip() or "⚠️ Пустой ответ. Попробуйте переформулировать запрос или загрузить документ заново."
    for c in _split_for_telegram(text):
        await update.message.reply_text(c, disable_web_page_preview=True)

def _normalize_ext(ext: str) -> str:
    mapping = {"с":"c","х":"x","о":"o","р":"p","а":"a","е":"e","к":"k","м":"m","т":"t","н":"h","в":"b"}
    return "".join(mapping.get(ch, ch) for ch in ext)

# ---- Dedup helpers ----
def _pdf_job_key(document: Document) -> str:
    return f"pdf::{document.file_unique_id or document.file_id}"

def _pdf_job_is_running(context, key: str) -> bool:
    jobs = context.user_data.setdefault("pdf_jobs", {})
    rec = jobs.get(key)
    if not rec: return False
    if time.time() - rec.get("ts", 0) > PDF_DEDUP_TTL:
        jobs.pop(key, None)
        return False
    return rec.get("running", False)

def _pdf_job_start(context, key: str, progress_msg_id: int | None):
    jobs = context.user_data.setdefault("pdf_jobs", {})
    jobs[key] = {"running": True, "ts": time.time(), "msg_id": progress_msg_id}

def _pdf_job_finish(context, key: str):
    jobs = context.user_data.setdefault("pdf_jobs", {})
    jobs.pop(key, None)

def _cmd_job_key(name: str, chat_id: int) -> str:
    return f"cmd::{name}::{chat_id}"

def _cmd_is_running(context, key: str) -> bool:
    jobs = context.user_data.setdefault("cmd_jobs", {})
    rec = jobs.get(key)
    if not rec: return False
    if time.time() - rec.get("ts", 0) > CMD_DEDUP_TTL:
        jobs.pop(key, None)
        return False
    return rec.get("running", False)

def _cmd_start(context, key: str, msg_id: int | None):
    jobs = context.user_data.setdefault("cmd_jobs", {})
    jobs[key] = {"running": True, "ts": time.time(), "msg_id": msg_id}

def _cmd_finish(context, key: str):
    jobs = context.user_data.setdefault("cmd_jobs", {})
    jobs.pop(key, None)

def _index_job_key(file_unique_id: str | None, file_hash: str | None) -> str:
    return f"idx::{file_unique_id or ''}::{file_hash or ''}"

def _idx_is_running(context, key: str) -> bool:
    jobs = context.bot_data.setdefault("index_jobs", {})
    rec = jobs.get(key)
    if not rec: return False
    if time.time() - rec.get("ts", 0) > INDEX_DEDUP_TTL:
        jobs.pop(key, None)
        return False
    return bool(rec.get("running"))

def _idx_start(context, key: str, msg_id: int | None):
    jobs = context.bot_data.setdefault("index_jobs", {})
    jobs[key] = {"running": True, "ts": time.time(), "msg_id": msg_id}

def _idx_finish(context, key: str):
    jobs = context.bot_data.setdefault("index_jobs", {})
    jobs.pop(key, None)

# ---------- Normalization ----------
def _normalize_confusables_ru(text: str) -> str:
    if not text:
        return text
    mapping = {
        "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "K": "К", "M": "М",
        "O": "О", "P": "Р", "T": "Т", "X": "Х", "Y": "У",
        "a": "а", "c": "с", "e": "е", "o": "о", "p": "р", "x": "х", "y": "у",
        "Ñ": "Н", "Í": "И", "ì": "и",
    }
    out = []
    for ch in text:
        out.append(mapping.get(ch, ch))
    fixed = "".join(out)
    fixed = re.sub(r"\s+([,.:;!?])", r"\1", fixed)
    fixed = fixed.replace("…", "...")
    return fixed

# ---------- PDF helpers ----------
def _compress_pdf_sync(src: str) -> str:
    try:
        doc = fitz.open(src)
        out = os.path.join(DOC_FOLDER, f"compressed_{Path(src).name}")
        doc.save(out, garbage=4, deflate=True, clean=True, linear=True)
        doc.close()
        return out
    except Exception as e:
        logger.warning("PDF compress failed for %s: %r", src, e)
        return src

def _ocr_pdf_to_text_fast(pdf_path: str, dpi: int, max_pages: int, poppler_path: Optional[str], tess_lang: str) -> str:
    # 1) text layer first
    try:
        doc = fitz.open(pdf_path)
        pieces = []
        pages = doc.page_count if (not max_pages or max_pages <= 0) else min(doc.page_count, max_pages)
        for i in range(pages):
            t = (doc.load_page(i).get_text("text") or "").strip()
            if t:
                pieces.append(t)
        quick = "\n".join(pieces).strip()
        if (not FORCE_OCR) and len(re.sub(r"\s+", "", quick)) > 200:
            return _normalize_confusables_ru(quick)
    except Exception as e:
        logger.debug("text-layer failed: %r", e)

    # 2) OCR with deadline
    start = time.time()
    kwargs = {"dpi": dpi, "fmt": "png"}
    if poppler_path:
        kwargs["poppler_path"] = poppler_path
    try:
        images = convert_from_path(pdf_path, **kwargs)
    except Exception as e:
        logger.error("render failed for %s: %r", pdf_path, e)
        return ""
    if max_pages and max_pages > 0:
        images = images[:max_pages]

    text_pages = [""] * len(images)

    def _ocr_one(idx_img):
        idx, img = idx_img
        return (idx, pytesseract.image_to_string(img, lang=tess_lang, config=TESS_CONFIG))

    with ThreadPoolExecutor(max_workers=min(6, (os.cpu_count() or 4))) as pool:
        futures = {pool.submit(_ocr_one, (i, im)): i for i, im in enumerate(images)]
        try:
            for fut in as_completed(futures, timeout=OCR_OVERALL_TIMEOUT):
                if time.time() - start > OCR_OVERALL_TIMEOUT:
                    break
                i = futures[fut]
                try:
                    idx, txt = fut.result()
                    text_pages[idx] = _normalize_confusables_ru(txt or "")
                except Exception as e:
                    logger.warning("OCR page %s failed/timeout: %r", i, e)
                    text_pages[i] = ""
        except Exception as e:
            logger.warning("OCR overall timeout or error: %r", e)

    return "\n".join(tp for tp in text_pages if tp).strip()

def _save_text_as_docx(text: str, base_name: str) -> str:
    os.makedirs(DOC_FOLDER, exist_ok=True)
    safe = re.sub(r"[^A-Za-zА-Яа-я0-9_.-]+", "_", base_name)[:60]
    out = os.path.join(DOC_FOLDER, f"{int(time.time())}_{safe}.docx")
    doc = DocxDocument()
    for line in (text or "").split("\n"):
        doc.add_paragraph(line)
    doc.save(out)
    return out

# ---------- Compare report (minimal) ----------
def _save_docx_report(filename: str, title: str, body: str) -> str:
    os.makedirs(DOC_FOLDER, exist_ok=True)
    path = os.path.join(DOC_FOLDER, filename)
    doc = DocxDocument()
    doc.add_heading(title, level=1)
    for para in (body or "").split("\n"):
        doc.add_paragraph(para)
    doc.save(path)
    return path

def _save_docx_compare_tables(filename: str, doc_name_a: str, doc_name_b: str, text_a: str, text_b: str) -> str:
    os.makedirs(DOC_FOLDER, exist_ok=True)
    path = os.path.join(DOC_FOLDER, filename)
    doc = DocxDocument()
    doc.add_heading(f"Изменения: {doc_name_a} → {doc_name_b}", level=1)
    a_lines = [l.rstrip() for l in (text_a or "").splitlines()]
    b_lines = [l.rstrip() for l in (text_b or "").splitlines()]
    sm = difflib.SequenceMatcher(a=a_lines, b=b_lines, autojunk=False)
    change_count = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        change_count += 1
        old_block = "\n".join(a_lines[i1:i2]).strip()
        new_block = "\n".join(b_lines[j1:j2]).strip()
        table = doc.add_table(rows=1, cols=2); table.style = 'Table Grid'
        cell_old = table.rows[0].cells[0]; cell_new = table.rows[0].cells[1]
        cell_old.text = old_block or "(пусто)"
        cell_new.text = new_block or "(пусто)"
        doc.add_paragraph("")
    if change_count == 0:
        doc.add_paragraph("Изменений не обнаружено.")
    doc.save(path)
    return path

# ---------- Helpers ----------
def _get_last_pdf_doc(context) -> dict | None:
    docs = (context.user_data.get("work_docs") or []) + (context.user_data.get("docs") or [])
    for rec in reversed(docs):
        name = rec.get("name") or ""
        if name.lower().endswith(".pdf"):
            return rec
    return None

# ---------- Handlers ----------
async def start(update: Update, context: Any):
    context.user_data["mode"] = "docs"
    usage_left = "∞" if is_admin(update.effective_user.id) else max(0, DAILY_FREE_LIMIT - get_usage(update.effective_user.id))
    msg = (
        "Привет!\n\n"
        "1) 🗂️ Работа с документами — загрузите DOC/DOCX/PDF; для PDF автоматически делаем сжатие+OCR и сохраняем в DOCX.\n"
        "2) Часто задаваемые вопросы — глобальный индекс.\n"
        "3) 🤖 AI-чат — свободный диалог.\n"
        "Команда: /pdf_to_docx — конвертация последнего загруженного PDF в DOCX (быстро, с таймаутами).\n\n"
        f"Сегодняшний лимит AI-чат: {usage_left} сообщений."
    )
    await update.message.reply_text(msg, reply_markup=MAIN_KB)

async def _work_handle_file(update: Update, context: Any, document: Document):
    raw = document.file_name or "file"
    fname = Path(raw).name
    base, ext = os.path.splitext(fname)
    ext = _normalize_ext(ext.lower().lstrip("."))
    if ext not in ("pdf", "docx", "doc"):
        await update.message.reply_text("Поддерживаются только PDF, DOCX, DOC для режима 'Работа с документами'.")
        return
    os.makedirs(DOC_FOLDER, exist_ok=True)
    ts = int(time.time())
    file_path = os.path.join(DOC_FOLDER, f"{base}_{ts}.{ext}")
    new_file = await context.bot.get_file(document.file_id)
    await new_file.download_to_drive(file_path)

    if ext == "pdf":
        key = _pdf_job_key(document)
        if _pdf_job_is_running(context, key):
            return
        sent = await update.message.reply_text("Принят PDF. Выполняю сжатие/извлечение…")
        _pdf_job_start(context, key, sent.message_id)
        try:
            def _job(pdf_path: str):
                pdf_use = _compress_pdf_sync(pdf_path) if PDF_COMPRESS else pdf_path
                quick = extract_text_from_pdf(pdf_use) or ""
                if (not FORCE_OCR) and len(quick.strip()) > 200:
                    use_text = _normalize_confusables_ru(quick)
                else:
                    use_text = _ocr_pdf_to_text_fast(pdf_use, OCR_DPI, OCR_MAX_PAGES, POPPLER_PATH, TESS_LANG)
                use_text = postprocess(use_text)
                docx_path = _save_text_as_docx(use_text, f"{Path(fname).stem}_ocr")
                chunks = smart_split_text(use_text) if use_text else []
                return use_text, chunks, docx_path, pdf_use
            text, chunks, derived_docx, pdf_final = await asyncio.to_thread(_job, file_path)
            context.user_data.setdefault("work_docs", []).append({
                "name": fname, "path": file_path, "text": text or "", "chunks": chunks or [],
                "from_pdf": True, "docx_path": derived_docx, "pdf_processed": pdf_final,
            })
            try:
                await update.effective_chat.edit_message_text(
                    message_id=sent.message_id,
                    text="PDF добавлен в контекст. Готово."
                )
            except Exception:
                pass
        finally:
            _pdf_job_finish(context, key)
        context.user_data["last_pdf_path"] = file_path
        context.user_data["last_pdf_name"] = fname
        return

    if ext == "docx":
        text = extract_text_from_docx(file_path)
    else:
        text = extract_text_from_doc(file_path)

    if not (text or "").strip():
        await update.message.reply_text("Не удалось извлечь текст из документа.")
        return

    chunks = smart_split_text(text)
    context.user_data.setdefault("work_docs", []).append({"name": fname, "path": file_path, "text": text, "chunks": chunks})
    await update.message.reply_text(f"Файл добавлен в контекст: {fname}. Документов в сессии: {len(context.user_data['work_docs'])}.")

async def handle_file(update: Update, context: Any):
    context.user_data.setdefault("work_docs", [])
    document: Document = update.message.document
    raw = document.file_name or "file"
    fname = Path(raw).name
    base, ext = os.path.splitext(fname)
    ext = _normalize_ext(ext.lower().lstrip("."))
    if ext not in ("pdf", "docx", "txt", "doc"):
        await update.message.reply_text("Поддерживаются только PDF, DOCX, DOC (и TXT для глобальной индексации).")
        return

    if context.user_data.get("mode") == "work":
        await _work_handle_file(update, context, document)
        if ext == "pdf":
            context.user_data["last_pdf_name"] = fname
        return

    os.makedirs(DOC_FOLDER, exist_ok=True)
    ts = int(time.time())
    file_path = os.path.join(DOC_FOLDER, f"{base}_{ts}.{ext}")
    new_file = await context.bot.get_file(document.file_id)
    await new_file.download_to_drive(file_path)

    if ext == "pdf":
        context.user_data["last_pdf_path"] = file_path
        context.user_data["last_pdf_name"] = fname

    try:
        file_hash = sha256_file(file_path)
    except Exception:
        file_hash = None

    manifest = load_manifest()
    if file_hash and file_hash in manifest.get("hashes", {}):
        await update.message.reply_text("Этот файл уже проиндексирован ранее. Можете задавать вопросы.")
        return

    key = _index_job_key(document.file_unique_id, file_hash)
    if _idx_is_running(context, key):
        return

    progress = await update.message.reply_text("Файл загружен. Индексация началась…")
    _idx_start(context, key, progress.message_id)

    try:
        def _index_job():
            try:
                added, total = index_file(file_path)
                return (True, added, total, None)
            except Exception as e:
                return (False, 0, 0, repr(e))
        ok, added, total, err = await asyncio.to_thread(_index_job)

        if ok:
            if added == 0:
                text = ("Файл загружен, но текст не извлечён. Возможно, это скан без текстового слоя.\n"
                        "Пришлите DOCX/TXT или PDF с текстом, либо включите OCR.")
            else:
                if file_hash:
                    manifest.setdefault("hashes", {})[file_hash] = {"fname": os.path.basename(file_path), "time": int(time.time())}
                    save_manifest(manifest)
                text = f"Индексация завершена. Добавлено фрагментов: {added}. Всего: {total}. Теперь можно задавать вопросы."
        else:
            text = f"❌ Ошибка индексации: {err}"

        try:
            await update.effective_chat.edit_message_text(message_id=progress.message_id, text=text)
        except Exception:
            await update.message.reply_text(text)
    finally:
        _idx_finish(context, key)

async def pdf_to_docx(update: Update, context: Any):
    key = _cmd_job_key("pdf_to_docx", update.effective_chat.id)
    if _cmd_is_running(context, key):
        return
    progress = await update.message.reply_text("Ищу последний PDF для конвертации…")
    _cmd_start(context, key, progress.message_id)
    try:
        rec = _get_last_pdf_doc(context)
        pdf_path = None
        if rec:
            pdf_path = rec.get("path") or rec.get("pdf_processed")
        if not pdf_path:
            pdf_path = context.user_data.get("last_pdf_path")
        if (not pdf_path) or (not os.path.isfile(pdf_path)):
            try:
                candidates = [p for p in sorted(Path(DOC_FOLDER).glob("*.pdf"), key=lambda x: x.stat().st_mtime, reverse=True)]
                if candidates:
                    pdf_path = str(candidates[0])
                    context.user_data["last_pdf_path"] = pdf_path
                    context.user_data["last_pdf_name"] = os.path.basename(pdf_path)
            except Exception:
                pdf_path = None
        if not pdf_path or not os.path.isfile(pdf_path):
            try:
                await update.effective_chat.edit_message_text(message_id=progress.message_id,
                    text="Не удалось найти загруженный PDF. Пришлите PDF и повторите команду.")
            except Exception:
                await update.message.reply_text("Не удалось найти загруженный PDF. Пришлите PDF и повторите команду.")
            return

        try:
            await update.effective_chat.edit_message_text(message_id=progress.message_id,
                text="Нашёл PDF. Быстрый анализ/извлечение…")
        except Exception:
            pass

        def _job(pp: str) -> str:
            use_pdf = _compress_pdf_sync(pp) if PDF_COMPRESS else pp
            text = _ocr_pdf_to_text_fast(use_pdf, OCR_DPI, OCR_MAX_PAGES, POPPLER_PATH, TESS_LANG)
            try:
                text = postprocess(text or "")
            except Exception:
                pass
            out_docx = _save_text_as_docx(text, f"{Path(pp).stem}_ocr")
            return out_docx

        loop = asyncio.get_running_loop()
        out_docx = await loop.run_in_executor(None, _job, pdf_path)

        # send once, with simple retry
        tries = 3
        ok = False
        for i in range(tries):
            try:
                with open(out_docx, "rb") as f:
                    await context.bot.send_document(
                        chat_id=update.effective_chat.id,
                        document=InputFile(f, filename=os.path.basename(out_docx)),
                        caption="Готово: PDF → DOCX (OCR, быстрый режим)."
                    )
                ok = True
                break
            except Exception:
                await asyncio.sleep(2 ** (i + 1))

        if ok:
            try:
                await update.effective_chat.edit_message_text(message_id=progress.message_id,
                    text="✅ Конвертация завершена. Отправил DOCX.")
            except Exception:
                pass
        else:
            if PUBLIC_BASE_URL:
                url = f"{PUBLIC_BASE_URL.rstrip('/')}/{os.path.basename(out_docx)}"
                txt = ("Не удалось отправить файл из-за сетевого таймаута Telegram. "
                       f"Скачайте по ссылке: {url}")
            else:
                txt = "Не удалось отправить файл из-за сетевого таймаута Telegram. Повторите команду позже."
            try:
                await update.effective_chat.edit_message_text(message_id=progress.message_id, text=txt)
            except Exception:
                await update.message.reply_text(txt)
    finally:
        _cmd_finish(context, key)

async def ai_mode(update: Update, context: Any):
    context.user_data["mode"] = "ai"
    await update.message.reply_text("Режим: AI-чат.", reply_markup=MAIN_KB)

async def docs_mode(update: Update, context: Any):
    context.user_data["mode"] = "docs"
    await update.message.reply_text("Режим: документы.", reply_markup=MAIN_KB)

async def work_mode(update: Update, context: Any):
    context.user_data["mode"] = "work"
    context.user_data.setdefault("work_docs", [])
    await update.message.reply_text("Режим: 🗂️ Работа с документами. Команда: /pdf_to_docx", reply_markup=MAIN_KB)

async def tm_mode(update: Update, context: Any):
    context.user_data["mode"] = TM_MODE
    await update.message.reply_text("Режим: 🏷️ Товарные знаки.", reply_markup=MAIN_KB)

async def handle_text(update: Update, context: Any):
    txt = (update.message.text or "").strip()
    if txt == AI_LABEL: return await ai_mode(update, context)
    if txt == DOCS_LABEL: return await docs_mode(update, context)
    if txt == WORK_LABEL: return await work_mode(update, context)
    if txt == TM_LABEL: return await tm_mode(update, context)
    await update.message.reply_text("Ок.")

async def error_handler(update: object, context: Any) -> None:
    logger.exception("Unhandled error while processing update: %s", update)

def build_application():
    request = HTTPXRequest(
        connect_timeout=30.0,
        read_timeout=300.0,
        write_timeout=300.0,
        pool_timeout=60.0
    )
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).request(request).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("ai", ai_mode))
    app.add_handler(CommandHandler("docs", docs_mode))
    app.add_handler(CommandHandler("tm", tm_mode))
    app.add_handler(CommandHandler("pdf_to_docx", pdf_to_docx))

    app.add_handler(MessageHandler(filters.Document.ALL, handle_file))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_error_handler(error_handler)
    return app
