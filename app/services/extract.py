import os
import logging
import subprocess
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict

import fitz  # PyMuPDF
from pdf2image import convert_from_path
import pytesseract

logger = logging.getLogger("semantic-bot")

# Постобработка текста (как в проекте)
try:
    from app.ocr.postprocess import postprocess
except Exception:
    def postprocess(text: str) -> str:
        return text

# Импорт полного OCR (ВНЕ try/except)
from app.ocr.ocr import ocr_pdf_to_text

# === Настройки ===
TESS_LANG = os.getenv("TESS_LANG", "rus+eng")
TESS_CONFIG = os.getenv("TESS_CONFIG", "--oem 1 --psm 6")
POPPLER_PATH = os.getenv("POPPLER_PATH", "").strip() or None
OCR_DPI = int(os.getenv("OCR_DPI", "200"))
MAX_WORKERS = max(1, int(os.getenv("OCR_THREADS", str(os.cpu_count() or 4))))
# Безопасная «с保险ка»: ограничим количество страниц под OCR
OCR_MAX_PAGES = int(os.getenv("OCR_MAX_PAGES", "200"))  # чтобы не ушатать RAM на огромных PDF
# Флаг принудительного полного OCR всего PDF (как в «Работа с документами»)
FORCE_OCR = (os.getenv("FORCE_OCR", "0") == "1")


def _text_quality_score(text: str) -> float:
    if not text:
        return 0.0
    letters = sum(ch.isalpha() for ch in text)
    if letters == 0:
        return 0.0
    cyr = sum("\u0400" <= ch <= "\u04FF" for ch in text)
    return 0.5 * (letters / max(1, len(text))) + 0.5 * (cyr / letters)


def _ocr_single_page(pdf_path: str, page_index_zero: int,
                     dpi: int = OCR_DPI, lang: str = TESS_LANG, config: str = TESS_CONFIG) -> str:
    """OCR одной конкретной страницы (индекс с 0). Рендерим только нужную страницу — безопасно по памяти."""
    first = page_index_zero + 1
    try:
        try:
            images = convert_from_path(
                pdf_path,
                dpi=dpi,
                first_page=first,
                last_page=first,
                fmt="jpeg",
                grayscale=True,
                poppler_path=POPPLER_PATH,
            )
        except TypeError:
            images = convert_from_path(pdf_path, dpi=dpi, first_page=first, last_page=first)
        if not images:
            return ""
        img = images[0]
        return pytesseract.image_to_string(img, lang=lang, config=config) or ""
    except Exception as e:
        logger.warning("OCR single page error (page=%s): %r", page_index_zero, e)
        return ""


def _ocr_pages_parallel(pdf_path: str, page_nums: List[int]) -> Dict[int, str]:
    """Параллельный OCR только нужных страниц, каждая страница рендерится отдельно (без больших диапазонов)."""
    if not page_nums:
        return {}
    # Страховка: ограничим количество страниц
    if len(page_nums) > OCR_MAX_PAGES:
        logger.warning("Too many pages for OCR (%d), truncating to %d", len(page_nums), OCR_MAX_PAGES)
        page_nums = page_nums[:OCR_MAX_PAGES]

    results: Dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(_ocr_single_page, pdf_path, i): i for i in page_nums}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                results[i] = fut.result() or ""
            except Exception as e:
                logger.warning("OCR future error on page %s: %r", i, e)
                results[i] = ""
    return results


def extract_text_from_pdf(file_path: str) -> str:
    """
    Быстрое извлечение текста из PDF:
      1) Сначала PyMuPDF постранично;
      2) OCR только пустых/битых страниц (safe-параллельный, поштучный рендер);
      3) Склейка и постобработка.

    Если FORCE_OCR=1 — выполняется ПОЛНЫЙ OCR всего PDF (как в «Работа с документами») сразу.
    """
    pages_text: List[str] = []
    pages_need_ocr: List[int] = []

    # 0) Принудительный полный OCR всего файла
    if FORCE_OCR:
        try:
            full = ocr_pdf_to_text(
                file_path,
                dpi=OCR_DPI,
                max_pages=OCR_MAX_PAGES,
                poppler_path=POPPLER_PATH,
                tess_lang=TESS_LANG
            )
            return postprocess(full)
        except Exception as e:
            logger.warning("FORCE_OCR failed, fallback to hybrid pipeline: %r", e)

    # 1) PyMuPDF постранично
    try:
        with fitz.open(file_path) as doc:
            for i, page in enumerate(doc):
                t = (page.get_text() or "").strip()
                if t:
                    # При желании можно проверять "качество" и отправлять низкое в OCR
                    # if _text_quality_score(t) < 0.5: ...
                    pages_text.append(t)
                else:
                    pages_text.append("")
                    pages_need_ocr.append(i)
    except Exception as e:
        logger.warning("PyMuPDF failed on %r; fallback to full OCR. Error: %r", file_path, e)
        # Простой и надёжный откат — полный OCR всего документа
        try:
            full = ocr_pdf_to_text(
                file_path,
                dpi=OCR_DPI,
                max_pages=OCR_MAX_PAGES,
                poppler_path=POPPLER_PATH,
                tess_lang=TESS_LANG
            )
            return postprocess(full)
        except Exception as ee:
            logger.error("Full OCR fallback failed: %r", ee)
            return ""

    # 2) OCR только нужных страниц
    if pages_need_ocr:
        try:
            ocr_map = _ocr_pages_parallel(file_path, pages_need_ocr)
            for i in pages_need_ocr:
                pages_text[i] = ocr_map.get(i, "")
        except Exception as e:
            logger.error("OCR pipeline failed: %r", e)

    # 3) Склейка и постобработка
    text = "\n\n".join(pages_text).strip()
    try:
        return postprocess(text)
    except Exception:
        return text


def extract_text_from_docx(file_path: str) -> str:
    try:
        from docx import Document as DocxDocument
        doc = DocxDocument(file_path)
        paras = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
        return "\n".join(paras).strip()
    except Exception as e:
        logger.error("Ошибка чтения DOCX: %r", e)
        return ""


def extract_text_from_txt(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except UnicodeDecodeError:
        try:
            with open(file_path, "r", encoding="cp1251") as f:
                return f.read().strip()
        except Exception:
            with open(file_path, "r", encoding="cp1251", errors="ignore") as f:
                return f.read().strip()
    except Exception as e:
        logger.error("Ошибка чтения TXT: %r", e)
        return ""


def extract_text_from_doc(file_path: str) -> str:
    """Чтение .doc через antiword (если установлен)."""
    try:
        if shutil.which("antiword") is None:
            logger.warning("antiword не найден — .doc не будет распознан.")
            return ""
        out = subprocess.run(["antiword", file_path], capture_output=True, text=True, timeout=60)
        if out.returncode == 0 and out.stdout:
            return out.stdout.strip()
        logger.error("antiword вернул код %s, stderr=%r", out.returncode, out.stderr)
        return ""
    except Exception as e:
        logger.error("Ошибка чтения DOC (antiword): %r", e)
        return ""
