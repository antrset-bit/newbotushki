import os, re, time, asyncio, logging, difflib, tempfile, gc
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
from docx.shared import Pt

import fitz  # PyMuPDF
import pytesseract
from pytesseract import Output as TessOutput
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger("semantic-bot")

AI_LABEL   = "🤖 AI-чат"
DOCS_LABEL = "Часто задаваемые вопросы"
WORK_LABEL = "🗂️ Работа с документами"
TM_LABEL   = "🏷️ Товарные знаки"
MAIN_KB    = ReplyKeyboardMarkup([[AI_LABEL, WORK_LABEL, TM_LABEL], [DOCS_LABEL]], resize_keyboard=True)

TM_MODE = "tm"

# ---- OCR defaults tuned for Russian docs + stability limits ----
FORCE_OCR = os.getenv("FORCE_OCR", "0") == "1"
OCR_DPI = int(os.getenv("OCR_DPI", "220"))
# Removed PDF_COMPRESS to avoid extra memory/cpu
OCR_MAX_PAGES = int(os.getenv("OCR_MAX_PAGES", "0"))  # 0 = авто-лимит по OCR_DEFAULT_CAP
OCR_DEFAULT_CAP = int(os.getenv("OCR_DEFAULT_CAP", "10"))  # если OCR_MAX_PAGES=0
POPPLER_PATH = os.getenv("POPPLER_PATH", "").strip() or None
TESS_LANG = os.getenv("TESS_LANG", "rus")
TESS_CONFIG = os.getenv("TESS_CONFIG", "--oem 1 --psm 6 -c preserve_interword_spaces=1")

# Stability guards
MAX_PIXELS_PER_PAGE = int(os.getenv("MAX_PIXELS_PER_PAGE", str(8_000_000)))     # ~8 MP
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(48 * 1024 * 1024)))      # 48 MB safety
OCR_OVERALL_TIMEOUT = int(os.getenv("OCR_OVERALL_TIMEOUT", "480"))              # seconds
OCR_THREADS_CAP = int(os.getenv("OCR_THREADS_CAP", "4"))

# Dedup windows
PDF_DEDUP_TTL = 120
INDEX_DEDUP_TTL = 600

def _split_for_telegram(text: str, max_len: int = TELEGRAM_MSG_LIMIT - 200) -> list[str]:
    parts, buf, cur = [], [], 0
    for p in text.replace("\\r\\n", "\\n").split("\\n\\n"):
        p = p.strip()
        if not p:
            chunk = "\\n\\n".join(buf).strip()
            if chunk: parts.append(chunk)
            buf, cur = [], 0
            continue
        need = len(p) + (2 if cur > 0 else 0)
        if cur + need <= max_len:
            buf.append(p); cur += need
        else:
            chunk = "\\n\\n".join(buf).strip()
            if chunk: parts.append(chunk)
            buf, cur = [], 0
            while len(p) > max_len:
                parts.append(p[:max_len]); p = p[max_len:]
            if p:
                buf, cur = [p], len(p)
    chunk = "\\n\\n".join(buf).strip()
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

# ---------- Post-OCR normalization: Latin look-alikes -> Cyrillic ----------
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
    fixed = re.sub(r"\\s+([,.:;!?])", r"\\1", fixed)
    fixed = fixed.replace("…", "...")
    return fixed

# ---------- DOCX helpers ----------
def _save_docx_report(filename: str, title: str, body: str) -> str:
    os.makedirs(DOC_FOLDER, exist_ok=True)
    path = os.path.join(DOC_FOLDER, filename)
    doc = DocxDocument()
    doc.add_heading(title, level=1)
    for para in (body or "").split("\\n"):
        doc.add_paragraph(para)
    doc.save(path)
    return path

def _add_table_change(doc: DocxDocument, old_text: str, new_text: str, caption: str | None = None):
    if caption:
        pcap = doc.add_paragraph()
        pcap.add_run(caption).bold = True
    table = doc.add_table(rows=1, cols=2)
    table.style = 'Table Grid'
    cell_old = table.rows[0].cells[0]
    old_lines = (old_text or '').splitlines() or ['(пусто)']
    for i, line in enumerate(old_lines):
        p = cell_old.paragraphs[0] if i == 0 else cell_old.add_paragraph()
        p.add_run(line)
    cell_new = table.rows[0].cells[1]
    new_lines = (new_text or '').splitlines() or ['(пусто)']
    for i, line in enumerate(new_lines):
        p = cell_new.paragraphs[0] if i == 0 else cell_new.add_paragraph()
        r = p.add_run(line)
        r.font.highlight_color = WD_COLOR_INDEX.YELLOW
    doc.add_paragraph("")

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
        old_block = "\\n".join(a_lines[i1:i2]).strip()
        new_block = "\\n".join(b_lines[j1:j2]).strip()
        if tag == "replace":
            _add_table_change(doc, old_block, new_block, caption=f"Различие {change_count}")
        elif tag == "insert":
            _add_table_change(doc, "", new_block, caption=f"Добавлено {change_count}")
        elif tag == "delete":
            _add_table_change(doc, old_block, "", caption=f"Удалено {change_count}")
    if change_count == 0:
        doc.add_paragraph("Изменений не обнаружено.")
    doc.save(path)
    return path

# ---------- Новый алгоритм сравнения: слева только отличия (отсутствуют в DOCX), справа — полный DOCX ----------
def _normalize_ru(s: str) -> str:
    s = (s or "").replace("\\r\\n", "\\n").replace("\\r", "\\n")
    s = re.sub(r"[ \\t]+", " ", s)
    s = re.sub(r"\\n{2,}", "\\n\\n", s)
    return s.strip()

def _sentences(text: str) -> list[str]:
    text = (text or "").replace("…", ".")
    parts: list[str] = []
    for block in re.split(r"\\n\\s*\\n", text):
        block = block.strip()
        if not block:
            continue
        subs = re.split(r"(?<=[\\.\\!\\?])\\s+(?=[А-ЯA-Z])", block)
        if not subs:
            subs = [block]
        for s in subs:
            s = s.strip()
            if s:
                parts.append(s)
    return parts

def _is_similar(a: str, b: str, threshold: float = 0.88) -> bool:
    ra = a.casefold()
    rb = b.casefold()
    return difflib.SequenceMatcher(None, ra, rb).ratio() >= threshold

def _compute_only_differences(pdf_text: str, docx_text: str) -> list[str]:
    pdf_text_n = _normalize_ru(pdf_text)
    docx_text_n = _normalize_ru(docx_text)
    pdf_sents  = _sentences(pdf_text_n)
    docx_sents = _sentences(docx_text_n)
    docx_keys = set()
    for s in docx_sents:
        k = re.sub(r"[^А-Яа-я0-9]+", "", s.casefold())
        if k:
            docx_keys.add(k)
    diffs: list[str] = []
    for s in pdf_sents:
        key = re.sub(r"[^А-Яа-я0-9]+", "", s.casefold())
        present = key in docx_keys
        if present:
            continue
        similar = any(_is_similar(s, t, 0.88) for t in docx_sents)
        if not similar:
            diffs.append(s)
    return diffs

def save_docx_two_column_compare(output_path: str, pdf_text: str, docx_text: str) -> str:
    diffs = _compute_only_differences(pdf_text, docx_text)
    doc = DocxDocument()
    p = doc.add_paragraph("Сравнение документов: отличия PDF vs ОРИГИНАЛ (DOCX)")
    if p.runs:
        p.runs[0].bold = True
        p.runs[0].font.size = Pt(12)
    table = doc.add_table(rows=1, cols=2)
    hdr = table.rows[0].cells
    hdr[0].text = "Отличается от DOCX (показываем только отсутствующее)"
    hdr[1].text = "Оригинал: текст DOCX"
    row = table.add_row().cells
    left = "Отличий не обнаружено — всё содержимое PDF присутствует в DOCX." if not diffs else ("\\n• " + "\\n• ".join(diffs))
    row[0].text = left
    row[1].text = _normalize_ru(docx_text)
    doc.save(output_path)
    return output_path

# ---------- OCR helpers with stability ----------
def _page_has_good_text(page: "fitz.Page") -> bool:
    txt = page.get_text("text") or ""
    txt = txt.strip()
    return len(re.sub(r"\\s+", "", txt)) >= 80

def _adaptive_zoom_for_page(page: "fitz.Page", base_dpi: int) -> float:
    # limit raster size by pixels and bytes
    # estimate a zoom that keeps width*height under MAX_PIXELS_PER_PAGE
    rect = page.rect
    w_pt, h_pt = rect.width, rect.height  # points (1/72 inch)
    # base zoom from DPI
    zoom = max(72.0, float(base_dpi)) / 72.0
    # compute pixel dims
    px_w = int(w_pt * zoom)
    px_h = int(h_pt * zoom)
    # shrink while exceeding pixel limit
    while px_w * px_h > MAX_PIXELS_PER_PAGE and zoom > 0.5:
        zoom *= 0.85
        px_w = int(w_pt * zoom)
        px_h = int(h_pt * zoom)
    return max(0.5, min(zoom, 4.0))

def _extract_or_ocr_page(doc: "fitz.Document", page_index: int, dpi: int, lang: str) -> list[str]:
    page = doc.load_page(page_index)
    if _page_has_good_text(page) and not FORCE_OCR:
        blocks = page.get_text("blocks") or []
        blocks.sort(key=lambda b: (round(b[1]), round(b[0])))
        paras = []
        for b in blocks:
            t = (b[4] or "").strip()
            if t:
                t = _normalize_confusables_ru(t)
                for par in re.split(r"\\n\\s*\\n", t):
                    par = postprocess(par.strip())
                    if par:
                        paras.append(par)
        return paras
    # OCR path
    zoom = _adaptive_zoom_for_page(page, dpi)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    # memory safety
    est_bytes = pix.width * pix.height * 3
    if est_bytes > MAX_IMAGE_BYTES:
        scale = (MAX_IMAGE_BYTES / est_bytes) ** 0.5
        zoom *= scale
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)

    img = None
    data = None
    try:
        from PIL import Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        data = pytesseract.image_to_data(img, lang=lang, config=TESS_CONFIG, output_type=TessOutput.DICT)
    except Exception:
        fd, tmp = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            pix.save(tmp)
            data = pytesseract.image_to_data(tmp, lang=lang, config=TESS_CONFIG, output_type=TessOutput.DICT)
        finally:
            try: os.remove(tmp)
            except Exception: pass

    paras: list[str] = []
    if not data or not data.get("text"):
        try:
            ocr_text = pytesseract.image_to_string(img, lang=lang, config=TESS_CONFIG) if img else ""
        except Exception:
            ocr_text = ""
        ocr_text = postprocess(_normalize_confusables_ru(ocr_text))
        for line in filter(None, (l.strip() for l in ocr_text.splitlines())):
            paras.append(line)
        gc.collect()
        return paras

    n = len(data["text"])
    buckets: dict[tuple[int,int], dict[int, list[tuple[int,str]]]] = {}
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue
        conf_raw = str(data.get("conf", ["-1"]*n)[i])
        try:
            conf = int(conf_raw)
        except Exception:
            conf = -1
        if conf < 0:
            continue
        b = int(data.get("block_num", [0]*n)[i] or 0)
        p = int(data.get("par_num",   [0]*n)[i] or 0)
        l = int(data.get("line_num",  [0]*n)[i] or 0)
        x = int(data.get("left",      [0]*n)[i] or 0)
        buckets.setdefault((b, p), {}).setdefault(l, []).append((x, txt))
    for (b,p) in sorted(buckets.keys()):
        lines = buckets[(b,p)]
        para_lines = []
        for ln in sorted(lines.keys()):
            words = sorted(lines[ln], key=lambda t: t[0])
            line_text = " ".join(w for _, w in words)
            line_text = postprocess(_normalize_confusables_ru(line_text))
            if line_text:
                para_lines.append(line_text)
        para = "\\n".join(para_lines).strip()
        if para:
            paras.append(para)
    gc.collect()
    return paras

def _ocr_pdf_to_docx_layout(pdf_path: str, dpi: int, max_pages: int, _poppler_path: Optional[str], lang: str) -> str:
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise RuntimeError(f"Не удалось открыть PDF: {e!r}")

    if doc.is_encrypted:
        try:
            doc.authenticate("")  # попытка открыть пустым паролем
        except Exception:
            pass
        if doc.is_encrypted:
            raise RuntimeError("PDF зашифрован и не может быть открыт без пароля.")

    total_pages = doc.page_count
    if max_pages and max_pages > 0:
        pages = list(range(min(total_pages, max_pages)))
    else:
        pages = list(range(min(total_pages, OCR_DEFAULT_CAP)))

    paras_per_page: dict[int, list[str]] = {}
    ocr_needed = []
    for i in pages:
        page = doc.load_page(i)
        if _page_has_good_text(page) and not FORCE_OCR:
            paras_per_page[i] = _extract_or_ocr_page(doc, i, dpi, lang)
        else:
            ocr_needed.append(i)

    if ocr_needed:
        cpu = max(1, (os.cpu_count() or 2) // 3)
        workers = max(1, min(cpu, OCR_THREADS_CAP))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_extract_or_ocr_page, doc, i, dpi, lang): i for i in ocr_needed}
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    paras_per_page[i] = fut.result()
                except Exception as e:
                    logger.warning("OCR failed p%s: %r", i, e)
                    paras_per_page[i] = []

    out = DocxDocument()
    for idx, i in enumerate(pages):
        for para in paras_per_page.get(i, []):
            for part in re.split(r"\\n{2,}", para):
                part = part.strip()
                if part:
                    out.add_paragraph(part)
        if idx < len(pages)-1:
            out.add_page_break()

    os.makedirs(DOC_FOLDER, exist_ok=True)
    safe_base = re.sub(r"[^A-Za-zА-Яа-я0-9_.-]+", "_", Path(pdf_path).stem)[:60]
    out_path = os.path.join(DOC_FOLDER, f"{int(time.time())}_{safe_base}_ocr.docx")
    out.save(out_path)
    return out_path

# ---------- Local fallbacks (no LLM) ----------
def _fallback_summary(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return "Документ пуст или не распознан."
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    head = lines[:100]
    def grab(label_variants):
        for lv in label_variants:
            m = re.search(rf"{lv}([^\\n\\r]{{0,200}})", text, flags=re.IGNORECASE)
            if m:
                return m.group(0).strip()
        return ""
    parties = grab(["стороны", "заказчик", "исполнитель", "арендодатель", "арендатор"])
    term    = grab(["срок", "период действия", "дата окончания"])
    price   = grab(["цена", "стоимость", "вознаграждение", "оплата"])
    subject = grab(["предмет договора", "предмет", "цель договора"])

    bullets = []
    if subject: bullets.append(f"• Предмет: {subject}")
    if parties: bullets.append(f"• Стороны: {parties}")
    if term:    bullets.append(f"• Срок: {term}")
    if price:   bullets.append(f"• Цена/оплата: {price}")
    if not bullets:
        bullets.append("• Ключевые положения не найдены автоматически — возможно, документ нестандартного формата.")

    preview = "\\n".join(head[:10])
    return "ИТОГ:\\n- Краткое резюме по основным полям.\\n\\n" + "\\n".join(bullets) + (f"\\n\\nФрагменты:\\n{preview}" if preview else "")

def _fallback_check(text: str) -> str:
    text = (text or "").lower()
    if not text:
        return "Документ пуст или не распознан."
    checks = [
        ("Стороны/реквизиты", all(k in text for k in ["реквиз", "подпис", "адрес"]) ),
        ("Предмет договора", "предмет" in text),
        ("Срок и расторжение", any(k in text for k in ["срок", "расторж", "прекращ"])),
        ("Ответственность/неустойка", any(k in text for k in ["ответствен", "неусто", "штраф"])),
        ("Конфиденциальность/персональные данные", any(k in text for k in ["конфиденц", "персональн", "пдн"])),
        ("Форс-мажор", any(k in text for k in ["форс", "непреодол"])),
        ("Порядок оплаты", any(k in text for k in ["оплат", "стоимост", "цена"])),
        ("Права на РИД/рез-ты работ", any(k in text for k in ["исключительн", "право", "интеллектуал"])),
    ]
    lines = [f"• {name}: {'OK' if ok else 'ПРОВЕРИТЬ/ОТСУТСТВУЕТ'}" for name, ok in checks]
    return "ПРОВЕРКА ДОГОВОРА (эвристика без ИИ):\\n" + "\\n".join(lines)

# --- надёжная отправка документов с ретраями ---
async def _send_document_with_retry(bot, chat_id: int, file_path: str, filename: str, caption: str | None = None, tries: int = 3):
    last_err = None
    for attempt in range(tries):
        try:
            with open(file_path, "rb") as f:
                await bot.send_document(
                    chat_id=chat_id,
                    document=InputFile(f, filename=filename),
                    caption=caption
                )
            return True
        except Exception as e:
            last_err = e
            await asyncio.sleep(2 ** (attempt + 1))
    logger.warning("send_document failed after retries: %r", last_err)
    return False

# ---------- Handlers ----------
async def start(update: Update, context: Any):
    context.user_data["mode"] = "docs"
    usage_left = "∞" if is_admin(update.effective_user.id) else max(0, DAILY_FREE_LIMIT - get_usage(update.effective_user.id))
    msg = (
        "Привет!\\n\\n"
        "1) 🗂️ Работа с документами — загрузите DOC/DOCX/PDF; для PDF автоматически делаем OCR и сохраняем в DOCX.\\n"
        "2) Часто задаваемые вопросы — классический режим с глобальным индексом (PDF/DOCX/TXT).\\n"
        "3) 🤖 AI-чат — свободный диалог.\\n"
        "4) 🏷️ Товарные знаки — поиск по Google Sheets.\\n"
        "Команда: /pdf_to_docx — конвертация последнего загруженного PDF в DOCX (OCR с сохранением порядка).\\n\\n"
        f"Сегодняшний лимит AI-чат: {usage_left} сообщений."
    )
    await update.message.reply_text(msg, reply_markup=MAIN_KB)

async def ai_mode(update: Update, context: Any):
    context.user_data["mode"] = "ai"
    usage_left = "∞" if is_admin(update.effective_user.id) else max(0, DAILY_FREE_LIMIT - get_usage(update.effective_user.id))
    await update.message.reply_text(
        f"Режим: AI-чат. Спросите что угодно. Доступно сегодня: {usage_left}.", reply_markup=MAIN_KB
    )

async def docs_mode(update: Update, context: Any):
    context.user_data["mode"] = "docs"
    await update.message.reply_text(
        "Режим: часто задаваемые вопросы по проиндексированным документам. Пришлите файл и задавайте вопрос.", reply_markup=MAIN_KB
    )

async def work_mode(update: Update, context: Any):
    context.user_data["mode"] = "work"
    context.user_data.setdefault("work_docs", [])
    await update.message.reply_text(
        "Режим: 🗂️ Работа с документами.\\n\\n"
        "Пришлите DOC/DOCX/PDF — добавлю в контекст. Для PDF: OCR → сохраняем как DOCX.\\n"
        "Команды:\\n"
        "• /doc_summary — резюме последнего документа\\n"
        "• /doc_check — проверить договор на ошибки/риски\\n"
        "• /doc_compare — сравнить два последних загруженных файла (в левой колонке — только то, чего нет в DOCX; справа — весь DOCX)\\n"
        "• /pdf_to_docx — конвертировать последний загруженный PDF в DOCX (OCR с сохранением порядка)\\n"
        "• /doc_clear — удалить все ранее загруженные документы\\n"
        "Любой вопрос в этом режиме — ответ с опорой на загруженные файлы.",
        reply_markup=MAIN_KB
    )

async def tm_mode(update: Update, context: Any):
    context.user_data["mode"] = TM_MODE
    await update.message.reply_text(
        "Режим: 🏷️ Товарные знаки.\\n\\n"
        "Отправьте название/ключевые слова — найду строки в Google Sheets и пришлю карточки.\\n"
        "Команды:\\n"
        "• /tm_reg — записи, где статус содержит «регистрация»\\n"
        "• /tm_exp — записи, где статус содержит «экспертиза»",
        reply_markup=MAIN_KB
    )

async def tm_cmd_reg(update: Update, context: Any):
    await tm_process_search(update.effective_chat.id, ROW_MATCH_REGISTERED, context)

async def tm_cmd_exp(update: Update, context: Any):
    await tm_process_search(update.effective_chat.id, ROW_MATCH_EXPERTISE, context)

async def tm_handle_text(update: Update, context: Any):
    user_text = (update.message.text or "").strip()
    kws = re.split(r"\\s+", user_text)
    await tm_process_search(update.effective_chat.id, lambda row: ROW_MATCH_KW(row, kws), context)

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
        sent = await update.message.reply_text("Принят PDF. Выполняю OCR…")
        _pdf_job_start(context, key, sent.message_id)
        try:
            def _job(pdf_path: str):
                pdf_use = pdf_path  # без сжатия ради скорости
                quick = extract_text_from_pdf(pdf_use) or ""
                if (not FORCE_OCR) and len(quick.strip()) > 80:
                    use_text = _normalize_confusables_ru(quick)
                    docx_path = _save_docx_report(f"{int(time.time())}_{Path(fname).stem}_from_text.docx", "Текст из PDF (без OCR)", use_text)
                else:
                    docx_path = _ocr_pdf_to_docx_layout(pdf_use, OCR_DPI, OCR_MAX_PAGES, POPPLER_PATH, TESS_LANG)
                    use_text = quick if quick.strip() else ""
                use_text = postprocess(use_text)
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
                    text="PDF добавлен в контекст. OCR/извлечение завершены."
                )
            except Exception:
                await update.message.reply_text("PDF добавлен в контекст. OCR/извлечение завершены.")
        finally:
            _pdf_job_finish(context, key)
        return

    if ext == "docx":
        text = extract_text_from_docx(file_path)
    else:
        text = extract_text_from_doc(file_path)

    if not (text or "").strip():
        await update.message.reply_text("Не удалось извлечь текст из документа. Убедитесь, что это не пустой/защищённый файл.")
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
        return

    os.makedirs(DOC_FOLDER, exist_ok=True)
    ts = int(time.time())
    file_path = os.path.join(DOC_FOLDER, f"{base}_{ts}.{ext}")
    new_file = await context.bot.get_file(document.file_id)
    await new_file.download_to_drive(file_path)

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
                text = ("Файл загружен, но текст не извлечён. Возможно, это скан без текстового слоя.\\n"
                        "Пришлите DOCX/TXT или PDF с текстом, либо включите OCR (tesseract+poppler/Docker).")
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

async def _work_retrieve(docs: list, query: str, k: int = 6) -> list[str]:
    if not docs: return []
    all_chunks = [c for d in docs for c in d.get("chunks", [])]
    if not all_chunks: return []
    import numpy as np
    try:
        q = get_embedding(query)
    except Exception:
        return all_chunks[:k]
    embs = []
    for c in all_chunks:
        try: embs.append(get_embedding(c))
        except Exception: embs.append(None)
    scored = []
    for c, e in zip(all_chunks, embs):
        if e is None: continue
        denom = (np.linalg.norm(q) * np.linalg.norm(e)) or 1.0
        scored.append((float(q @ e / denom), c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:k]] if scored else all_chunks[:k]

async def _reply_with_docx(update: Update, title: str, content: str, base_name: str):
    safe_base = re.sub(r"[^A-Za-zА-Яа-я0-9_.-]+", "_", base_name)[:60] or "report"
    filename = f"{int(time.time())}_{safe_base}.docx"
    path = _save_docx_report(filename, title, content)
    await _send_document_with_retry(update.get_bot(), update.effective_chat.id, path, filename)

def _get_last_pdf_doc(context) -> dict | None:
    docs = (context.user_data.get("work_docs") or []) + (context.user_data.get("docs") or [])
    for rec in reversed(docs):
        name = rec.get("name") or ""
        if name.lower().endswith(".pdf"):
            return rec
    return None

# Новая команда: /pdf_to_docx
async def pdf_to_docx(update: Update, context: Any):
    rec = _get_last_pdf_doc(context)
    if not rec:
        await update.message.reply_text("Не найден загруженный PDF. Пришлите PDF-файл и повторите.")
        return
    pdf_path = rec.get("path") or rec.get("pdf_processed") or ""
    if not pdf_path or not os.path.isfile(pdf_path):
        await update.message.reply_text("Не удалось определить путь к PDF. Пришлите файл ещё раз.")
        return

    async def _do_ocr() -> str:
        return await asyncio.to_thread(_ocr_pdf_to_docx_layout, pdf_path, OCR_DPI, OCR_MAX_PAGES, POPPLER_PATH, TESS_LANG)

    try:
        out_docx = await asyncio.wait_for(_do_ocr(), timeout=OCR_OVERALL_TIMEOUT)
    except asyncio.TimeoutError:
        await update.message.reply_text("⏳ OCR прерван по таймауту. Попробуйте уменьшить число страниц или установить OCR_MAX_PAGES.")
        return
    except Exception as e:
        await update.message.reply_text(f"Ошибка OCR/DOCX: {e!r}")
        return

    ok = await _send_document_with_retry(
        bot=context.bot,
        chat_id=update.effective_chat.id,
        file_path=out_docx,
        filename=os.path.basename(out_docx),
        caption="Готово: PDF → DOCX (OCR)."
    )
    if not ok:
        if PUBLIC_BASE_URL:
            url = f"{PUBLIC_BASE_URL.rstrip('/')}/{os.path.basename(out_docx)}"
            await update.message.reply_text(
                "Не удалось отправить файл из-за сетевого таймаута Telegram. "
                f"Можно скачать по ссылке: {url}"
            )
        else:
            await update.message.reply_text(
                "Не удалось отправить файл из-за сетевого таймаута Telegram. Повторите команду позже."
            )

async def doc_summary(update: Update, context: Any):
    docs = context.user_data.get("work_docs") or []
    if not docs:
        await update.message.reply_text("Нет документов в контексте. Пришлите DOC/DOCX/PDF и повторите.")
        return
    last = docs[-1]
    retrieved = await _work_retrieve([last], "краткое резюме документа с фокусом на тип работ/услуг/поставки/агентских/аренды", k=12)
    prompt = (
        "Ты — юридический ассистент. На основе КОНТЕКСТА составь краткое, но точное резюме договора.\\n"
        "Обязательно:\\n"
        "• Определи ТИП правоотношения и предмет: что именно делает исполнитель для заказчика (выбери из: выполнение работ, оказание услуг, поставка, агентские услуги, аренда), укажи формулировку из документа.\\n"
        "• Стороны (наименования), срок, цена/порядок оплаты, ответственность/неустойки, расторжение, конфиденциальность/ПДн, права на результаты (если есть).\\n"
        "• Избегай общих фраз «предмет договора» — пиши конкретное действие.\\n"
        "Формат:\\n"
        "— Тип/Предмет: …\\n— Стороны: …\\n— Срок: …\\n— Цена/оплата: …\\n— Ответственность: …\\n— Расторжение: …\\n— Конфиденциальность/ПДн: …\\n— Права на РИД: …\\n"
        "Используй только факты из контекста, без выдумок."
    )
    answer = generate_answer_with_gemini(prompt, retrieved) or ""
    if not answer.strip() or answer.strip().startswith("⚠️"):
        answer = _fallback_summary(last["text"])
    await send_long(update, f"📄 Резюме по: {last['name']}\\n\\n{answer}")
    await _reply_with_docx(update, f"Резюме: {last['name']}", answer, f"summary_{Path(last['name']).stem}")

async def doc_check(update: Update, context: Any):
    docs = context.user_data.get("work_docs") or []
    if not docs:
        await update.message.reply_text("Нет документов в контексте. Сначала загрузите файл.")
        return
    last = docs[-1]
    retrieved = await _work_retrieve([last], "проверка договора на ошибки/риски/пробелы/несоответствия", k=16)
    prompt = (
        "Ты — опытный договорной юрист. Проверь договор на ошибки и риски строго по КОНТЕКСТУ.\\n"
        "Укажи: спорные формулировки, односторонние условия, пробелы (не хватает условий), конфликты разделов,\\n"
        "риски по цене/срокам/ответственности/расторжению/РИД/конфиденциальности/ПДн и дай конкретные правки (как переписать пункты).\\n"
        "Структура:\\n1) Краткий итог рисков (3–6 пунктов)\\n2) Детальный разбор по разделам с цитатами\\n3) Список предложенных правок (bullet list)"
    )
    answer = generate_answer_with_gemini(prompt, retrieved) or ""
    if not answer.strip() or answer.strip().startswith("⚠️"):
        answer = _fallback_check(last["text"])
    await send_long(update, f"🔎 Проверка договора: {last['name']}\\n\\n{answer}")
    await _reply_with_docx(update, f"Проверка договора: {last['name']}", answer, f"check_{Path[last['name']).stem}")

async def doc_compare(update: Update, context: Any):
    """
    В ЛЕВОЙ колонке: показываем только то, чего НЕТ в DOCX вообще (даже если в PDF это просто «переехало»).
    В ПРАВОЙ колонке: полный текст DOCX.
    """
    docs = context.user_data.get("work_docs") or []
    if len(docs) < 2:
        await update.message.reply_text("Нужно минимум два документа. Загрузите ещё один и повторите /doc_compare.")
        return
    a, b = docs[-2], docs[-1]
    name_a = (a.get("name") or "").lower()
    name_b = (b.get("name") or "").lower()
    if name_a.endswith(".docx") and not name_b.endswith(".docx"):
        text_docx = a.get("text") or ""
        text_pdf  = b.get("text") or ""
        base_left = f"{Path(b['name']).stem}"
        base_docx = f"{Path(a['name']).stem}"
    elif name_b.endswith(".docx") and not name_a.endswith(".docx"):
        text_docx = b.get("text") or ""
        text_pdf  = a.get("text") or ""
        base_left = f"{Path(a['name']).stem}"
        base_docx = f"{Path(b['name']).stem}"
    else:
        text_pdf  = a.get("text") or ""
        text_docx = b.get("text") or ""
        base_left = f"{Path(a['name']).stem}"
        base_docx = f"{Path(b['name']).stem}"
    text_pdf  = postprocess(text_pdf or "")
    text_docx = postprocess(text_docx or "")

    os.makedirs(DOC_FOLDER, exist_ok=True)
    filename = f"{int(time.time())}_compare_{base_left}_vs_{base_docx}.docx"
    out_path = os.path.join(DOC_FOLDER, filename)
    out_path = save_docx_two_column_compare(out_path, text_pdf, text_docx)
    await _send_document_with_retry(update.get_bot(), update.effective_chat.id, out_path, filename)

async def doc_clear(update: Update, context: Any):
    docs = context.user_data.get("work_docs") or []
    deleted_count = 0
    for d in docs:
        for key in ("path", "docx_path", "pdf_processed"):
            p = d.get(key)
            if p and os.path.exists(p):
                try:
                    os.remove(p); deleted_count += 1
                except Exception:
                    pass
    context.user_data["work_docs"] = []
    await update.message.reply_text(f"Очищено файлов: {deleted_count}. Контекст теперь пуст.")

async def handle_text(update: Update, context: Any):
    user_query = (update.message.text or "").strip()
    mode = context.user_data.get("mode", "docs")

    if user_query == AI_LABEL:
        await ai_mode(update, context); return
    if user_query == DOCS_LABEL:
        await docs_mode(update, context); return
    if user_query == TM_LABEL:
        await tm_mode(update, context); return
    if user_query == WORK_LABEL:
        await work_mode(update, context); return

    if mode == "ai":
        uid = update.effective_user.id
        if not is_admin(uid):
            used = get_usage(uid)
            if used >= DAILY_FREE_LIMIT:
                await update.message.reply_text(
                    "Достигнут дневной лимит бесплатных обращений к ИИ (10). Возвращайтесь завтра или обратитесь к администратору.",
                    reply_markup=MAIN_KB,
                )
                return
        def _ai_job():
            try:
                return generate_direct_ai_answer(user_query)
            except Exception as e:
                return f"⚠️ Ошибка при обработке запроса: {repr(e)}"
        answer = await asyncio.to_thread(_ai_job)
        if answer and not is_admin(uid):
            inc_usage(uid)
        await send_long(update, answer)
        if not is_admin(uid):
            left = max(0, DAILY_FREE_LIMIT - get_usage(uid))
            await update.message.reply_text(f"Остаток на сегодня: {left}.")
        return

    if mode == "work":
        docs = context.user_data.get("work_docs") or []
        if not docs:
            await update.message.reply_text("Пока нет документов в контексте. Пришлите DOC/DOCX/PDF.")
            return
        retrieved = await _work_retrieve(docs, user_query, k=RETRIEVAL_K)
        answer = generate_answer_with_gemini(user_query, retrieved) or ""
        if not answer.strip() or answer.strip().startswith("⚠️"):
            answer = "Краткая справка по загруженным документам:\\n" + _fallback_summary("\\n\\n".join(retrieved) if retrieved else docs[-1]["text"])
        await send_long(update, answer)
        return

    if mode == "tm":
        low = user_query.lower()
        if low.startswith("/tm_reg"):
            await tm_cmd_reg(update, context); return
        if low.startswith("/tm_exp"):
            await tm_cmd_exp(update, context); return
        await tm_handle_text(update, context)
        return

    if not (os.path.exists(INDEX_FILE) and os.path.exists(TEXTS_FILE)):
        await update.message.reply_text("Нет проиндексированных документов. Сначала загрузите файл.")
        return

    def _answer_job():
        try:
            chunks = retrieve_chunks(user_query, k=RETRIEVAL_K)
            return generate_answer_with_gemini(user_query, chunks)
        except Exception as e:
            return f"⚠️ Ошибка при обработке запроса: {repr(e)}"

    answer = await asyncio.to_thread(_answer_job)
    if not answer:
        await update.message.reply_text("⚠️ Пустой ответ.")
    else:
        await send_long(update, answer)

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
    app.add_handler(CommandHandler("tm_reg", tm_cmd_reg))
    app.add_handler(CommandHandler("tm_exp", tm_cmd_exp))

    app.add_handler(CommandHandler("doc_summary", doc_summary))
    app.add_handler(CommandHandler("doc_check", doc_check))
    app.add_handler(CommandHandler("doc_compare", doc_compare))
    app.add_handler(CommandHandler("pdf_to_docx", pdf_to_docx))
    app.add_handler(CommandHandler("doc_clear", doc_clear))

    app.add_handler(MessageHandler(filters.TEXT & filters.Regex(f"^{re.escape(TM_LABEL)}$"), tm_mode))
    app.add_handler(MessageHandler(filters.TEXT & filters.Regex(f"^{re.escape(WORK_LABEL)}$"), work_mode))
    app.add_handler(MessageHandler(filters.TEXT & filters.Regex(f"^{re.escape(DOCS_LABEL)}$"), docs_mode))
    app.add_handler(MessageHandler(filters.TEXT & filters.Regex(f"^{re.escape(AI_LABEL)}$"), ai_mode))

    app.add_handler(MessageHandler(filters.Document.ALL, handle_file))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_error_handler(error_handler)
    return app
