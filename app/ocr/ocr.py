# app/ocr/ocr.py — OCR + нормализация + опциональная доводка Gemini
import os, re, time, logging
from pathlib import Path
from typing import Optional, Tuple, List

import fitz  # PyMuPDF
from pdf2image import convert_from_path
import pytesseract
from docx import Document as DocxDocument

from app.config import (
    DOC_FOLDER, OCR_DPI, OCR_MAX_PAGES, POPPLER_PATH, TESS_LANG, TESS_CONFIG, PDF_COMPRESS
)
from app.services.llm_clean import clean_with_gemini
from app.ocr.postprocess import postprocess

logger = logging.getLogger("semantic-bot")

# ---------------------- Latin -> Cyrillic mapping (for strict cleanup) ----------------------
L2C = {
    # Upper
    "A":"А","B":"В","C":"С","D":"Д","E":"Е","F":"Ф","G":"Г","H":"Н","I":"І","J":"Й","K":"К","L":"Л","M":"М",
    "N":"Н","O":"О","P":"Р","Q":"К","R":"Р","S":"С","T":"Т","U":"И","V":"В","W":"Ш","X":"Х","Y":"У","Z":"З",
    # Lower
    "a":"а","b":"в","c":"с","d":"д","e":"е","f":"ф","g":"г","h":"н","i":"і","j":"й","k":"к","l":"л","m":"м",
    "n":"н","o":"о","p":"р","q":"к","r":"г","s":"с","t":"т","u":"и","v":"в","w":"ш","x":"х","y":"у","z":"з",
}

def _keep_char(ch: str) -> bool:
    o = ord(ch)
    if 0x0400 <= o <= 0x04FF:  # Cyrillic
        return True
    if ch.isdigit() or ch.isspace():
        return True
    if ch in ",.;:!?—–-()[]{}«»\"'/%№@&*+<>|=_":
        return True
    return False

SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.:;!?%])")
MULTI_DOTS_RE = re.compile(r"\.{3,}")
BAD_EQ_RE = re.compile(r"[^\S\r\n]*={2,}[^\S\r\n]*")

def map_and_strip_latin(text: str) -> str:
    out = []
    for ch in text:
        o = ord(ch)
        if (0x0041 <= o <= 0x005A) or (0x0061 <= o <= 0x007A):  # Latin
            mapped = L2C.get(ch, "")
            if mapped:
                out.append(mapped)
            # else: drop char
        else:
            if _keep_char(ch):
                out.append(ch)
    return "".join(out)

def normalize_text(text: str) -> str:
    if not text:
        return text
    # Замена N123 на №123
    text = re.sub(r"(?:(?<=\s)|^)N\s*(?=\d)", "№", text)
    # Приводим дефисы к единому виду
    text = text.replace("—", "-").replace("–", "-").replace("−", "-")
    # Убираем лишние знаки равенства вокруг текста
    text = re.sub(r"\s*={2,}\s*", " ", text)
    # Убираем пробел перед пунктуацией
    text = re.sub(r"\s+([,.:;!?%])", r"\1", text)
    # Сжимаем многоточия
    text = re.sub(r"\.{3,}", "...", text)
    # Удаляем мусор в виде подряд идущих не букв/цифр
    text = re.sub(r"[^\w\sА-Яа-яЁё.,:;!?()№\-]{2,}", " ", text)
    # Маппинг латиницы в кириллицу
    text = map_and_strip_latin(text)
    # Убираем повторяющиеся пробелы
    text = re.sub(r"[ \t]+", " ", text)
    # Убираем пробелы в начале/конце строк
    text = "\n".join(line.strip() for line in text.splitlines())
    # Убираем пустые строки в начале/конце и лишние переносы
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def deskew_and_binarize(img):
    try:
        import cv2, numpy as np
    except Exception:
        return None
    arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    arr = cv2.medianBlur(arr, 3)
    arr = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    coords = cv2.findNonZero(255 - arr)
    if coords is not None:
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        if angle < -45: angle = -(90 + angle)
        else: angle = -angle
        (h, w) = arr.shape[:2]
        import cv2 as _cv2
        M = _cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        arr = _cv2.warpAffine(arr, M, (w, h), flags=_cv2.INTER_CUBIC, borderMode=_cv2.BORDER_REPLICATE)
    return arr

def compress_pdf(src: str) -> str:
    if not PDF_COMPRESS:
        return src
    try:
        doc = fitz.open(src)
        out = os.path.join(DOC_FOLDER, f"compressed_{Path(src).name}")
        doc.save(out, garbage=4, deflate=True, clean=True, linear=True)
        doc.close()
        return out
    except Exception as e:
        logger.warning("PDF compress failed for %s: %r", src, e)
        return src

def _tess(img, lang: str, config: str) -> str:
    try:
        return pytesseract.image_to_string(img, lang=lang, config=config)
    except Exception as e:
        logger.warning("tesseract failed: %r", e)
        return ""

def ocr_pdf_to_text(pdf_path: str, dpi: int = OCR_DPI, max_pages: int = OCR_MAX_PAGES,
                    poppler_path: Optional[str] = POPPLER_PATH, tess_lang: str = TESS_LANG) -> str:
    try:
        kwargs = {"dpi": dpi, "fmt": "png"}
        if poppler_path:
            kwargs["poppler_path"] = poppler_path
        images = convert_from_path(pdf_path, **kwargs)
        if max_pages and max_pages > 0:
            images = images[:max_pages]
        pages_text: List[str] = []
        for img in images:
            try:
                arr = deskew_and_binarize(img)
                txt = _tess(arr if arr is not None else img, lang=tess_lang, config=TESS_CONFIG)
                # retry for columns / mixed lang
                cyr_ratio = sum('\u0400' <= ch <= '\u04FF' for ch in txt) / max(1, len(txt))
                if cyr_ratio < 0.3:
                    cfg2 = TESS_CONFIG
                    if "--psm" in cfg2:
                        import re as _re
                        cfg2 = _re.sub(r"--psm\s+\d+", "--psm 4", cfg2)
                    else:
                        cfg2 += " --psm 4"
                    alt = _tess(arr if arr is not None else img, lang="rus+eng", config=cfg2)
                    if len(alt) > len(txt):
                        txt = alt
                txt = normalize_text(txt)
            except Exception as e:
                logger.warning("OCR page failed: %r", e)
                txt = ""
            pages_text.append(txt)
        return "\n".join(pages_text).strip()
    except Exception as e:
        logger.error("OCR failed for %s: %r", pdf_path, e)
        return ""

def save_text_as_docx(text: str, base_name: str) -> str:
    os.makedirs(DOC_FOLDER, exist_ok=True)
    safe = re.sub(r"[^A-Za-zА-Яа-я0-9_.-]+", "_", base_name)[:60]
    out = os.path.join(DOC_FOLDER, f"{int(time.time())}_{safe}.docx")
    doc = DocxDocument()
    for para in (text or "").split("\n"):
        doc.add_paragraph(para)
    doc.save(out)
    return out

USE_GEMINI_NORMALIZE = (os.getenv("USE_GEMINI_NORMALIZE") or "1") == "1"

def ocr_file(path: str) -> Tuple[str, str]:
    use_pdf = compress_pdf(path)
    raw_text = ocr_pdf_to_text(use_pdf)
    text = normalize_text(raw_text)
    text = postprocess(text)
    if USE_GEMINI_NORMALIZE and (text or "").strip():
        text = clean_with_gemini(text, max_chars=6000, temperature=0.1)
    docx_path = save_text_as_docx(text, Path(path).stem + "_ocr")
    return text, docx_path
