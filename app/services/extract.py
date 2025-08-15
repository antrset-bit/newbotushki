import logging, subprocess, shutil
import fitz  # PyMuPDF
from pdf2image import convert_from_path
import pytesseract
from docx import Document as DocxDocument

logger = logging.getLogger("semantic-bot")
from app.ocr.postprocess import postprocess

def extract_text_from_pdf(file_path: str) -> str:
    # Сначала PyMuPDF, если пусто — OCR через tesseract
    try:
        doc = fitz.open(file_path)
        pages_text = [page.get_text().strip() for page in doc]
        text = "\n".join(pages_text).strip()
        if text:
            return postprocess(text)
    except Exception as e:
        logger.warning("PyMuPDF не смог извлечь текст (%s). Переходим к OCR.", repr(e))
    try:
        images = convert_from_path(file_path)
        ocr_texts = [pytesseract.image_to_string(img) for img in images]
        return "\n".join(ocr_texts).strip()
    except Exception as e:
        logger.error("Ошибка OCR/convert_from_path: %s", repr(e))
        return ""

def extract_text_from_docx(file_path: str) -> str:
    try:
        doc = DocxDocument(file_path)
        return "\n".join(p.text for p in doc.paragraphs).strip()
    except Exception as e:
        logger.error("Ошибка чтения DOCX: %s", repr(e))
        return ""

def extract_text_from_txt(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="cp1251", errors="ignore") as f:
            return f.read().strip()
    except Exception as e:
        logger.error("Ошибка чтения TXT: %s", repr(e))
        return ""

def extract_text_from_doc(file_path: str) -> str:
    """Чтение .doc через antiword (должен быть установлен в контейнере)."""
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
        logger.error("Ошибка чтения DOC (antiword): %s", repr(e))
        return ""
