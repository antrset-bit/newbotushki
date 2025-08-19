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

# Используем вашу функцию пост-обработки
try:
    from app.ocr.postprocess import postprocess
except Exception:  # на всякий случай, чтобы модуль не падал
    def postprocess(text: str) -> str:
        return text

# Настройки по умолчанию для OCR
TESS_LANG = os.getenv("TESS_LANG", "rus+eng")
# Быстрый и точный компромисс для договоров: LSTM-only + одна колонка
TESS_CONFIG = os.getenv("TESS_CONFIG", "--oem 1 --psm 6")
# Poppler путь (если требуется в контейнере)
POPPLER_PATH = os.getenv("POPPLER_PATH", "").strip() or None


def _text_quality_score(text: str) -> float:
    """
    Грубая эвристика «качества»: доля кириллицы, доля букв в целом.
    Возвращает 0..1 (выше — лучше).
    """
    if not text:
        return 0.0
    letters = sum(ch.isalpha() for ch in text)
    if letters == 0:
        return 0.0
    cyr = sum("\u0400" <= ch <= "\u04FF" for ch in text)
    return 0.5 * (letters / max(1, len(text))) + 0.5 * (cyr / letters)


def _ocr_page(img, lang: str = TESS_LANG, config: str = TESS_CONFIG) -> str:
    """
    Быстрый OCR одной страницы.
    Важно: img должен быть уже серым/умеренного размера для скорости.
    """
    try:
        return pytesseract.image_to_string(img, lang=lang, config=config) or ""
    except Exception as e:
        logger.warning("OCR error: %r", e)
        return ""


def _ocr_pages_parallel(pdf_path: str, page_nums: List[int], dpi: int = 200,
                        lang: str = TESS_LANG, config: str = TESS_CONFIG) -> Dict[int, str]:
    """
    Параллельный OCR для подмножества страниц.
    page_nums — индексы страниц с 0.
    Возвращает мапу {page_index: text}.
    """
    if not page_nums:
        return {}

    first, last = min(page_nums) + 1, max(page_nums) + 1
    try:
        images = convert_from_path(
            pdf_path,
            dpi=dpi,
            first_page=first,
            last_page=last,
            fmt="jpeg",
            grayscale=True,
            poppler_path=POPPLER_PATH,
        )
    except TypeError:
        # Если установленная версия pdf2image не поддерживает named args
        images = convert_from_path(pdf_path, dpi=dpi)

    base = first - 1
    results: Dict[int, str] = {}

    # Кол-во потоков = кол-ву CPU (или минимум 4)
    max_workers = os.cpu_count() or 4
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {}
        for i in page_nums:
            # Берём соответствующую картинку: images[i - base]
            idx = i - base
            if 0 <= idx < len(images):
                fut = ex.submit(_ocr_page, images[idx], lang, config)
                futs[fut] = i

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
    1) Сначала PyMuPDF постранично.
    2) OCR только для страниц без текста.
    3) Склейка и лёгкая пост-обработка.
    """
    # 1) PyMuPDF
    try:
        doc = fitz.open(file_path)
        pages_text: List[str] = []
        pages_need_ocr: List[int] = []

        for i, page in enumerate(doc):
            t = (page.get_text() or "").strip()
            if t:
                pages_text.append(t)
            else:
                pages_text.append("")  # заполнитель
                pages_need_ocr.append(i)

        # Опционально: если текст найден, но "битый", можем отметить страницы на OCR по порогу качества
        # (оставил простую версию для скорости; при желании можно усложнить).

    except Exception as e:
        logger.warning("PyMuPDF failed to read PDF (%r). Falling back to OCR for all pages.", e)
        pages_text = []
        pages_need_ocr = list(range(0, 10**9))  # sentinel; будет перезаписано ниже

    # 2) Если есть страницы для OCR — запускаем параллельно
    if pages_need_ocr:
        try:
            ocr_map = _ocr_pages_parallel(file_path, pages_need_ocr, dpi=200, lang=TESS_LANG, config=TESS_CONFIG)
            for i in pages_need_ocr:
                pages_text[i] = ocr_map.get(i, "")
        except Exception as e:
            logger.error("OCR pipeline failed: %r", e)

    # Если PyMuPDF целиком рухнул — pages_text могли не определить корректно.
    if not pages_text:
        try:
            # OCR для всех страниц (просто вызываем без ограничения)
            total_pages = fitz.open(file_path).page_count
            all_pages = list(range(total_pages))
            ocr_map = _ocr_pages_parallel(file_path, all_pages, dpi=200, lang=TESS_LANG, config=TESS_CONFIG)
            pages_text = [ocr_map.get(i, "") for i in all_pages]
        except Exception as e:
            logger.error("Full OCR fallback failed: %r", e)
            return ""

    text = "\n\n".join(pages_text).strip()
    try:
        return postprocess(text)
    except Exception:
        return text


def extract_text_from_docx(file_path: str) -> str:
    """Чтение DOCX быстро и просто."""
    try:
        from docx import Document as DocxDocument
        doc = DocxDocument(file_path)
        paras = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
        return "\n".join(paras).strip()
    except Exception as e:
        logger.error("Ошибка чтения DOCX: %r", e)
        return ""


def extract_text_from_txt(file_path: str) -> str:
    """Чтение TXT c fallback-ами по кодировке."""
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
    """
    Чтение .doc через antiword (если установлен).
    Возвращает пустую строку, если утилита недоступна/ошибка.
    """
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
