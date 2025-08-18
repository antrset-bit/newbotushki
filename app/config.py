import os
import logging
from pathlib import Path
from typing import Tuple

logger = logging.getLogger("semantic-bot")

_FALLBACK_WARNED = False
_SILENCE_PATH_WARN = (os.getenv("SILENCE_PATH_WARN") or "0") == "1"


def _env(k: str, default: str = "") -> str:
    v = os.getenv(k, default)
    return v.strip() if isinstance(v, str) else v


def _ensure_dir(path_str: str) -> bool:
    try:
        Path(path_str).mkdir(parents=True, exist_ok=True)
        return True
    except (PermissionError, FileNotFoundError) as e:
        if not _SILENCE_PATH_WARN:
            logger.warning("No write access to %s: %r", path_str, e)
        return False


def _prepare_paths_or_fallback(paths: list[str]) -> list[str]:
    ok_all = True
    for p in paths:
        parent = Path(p).parent if Path(p).suffix else Path(p)
        ok_all &= _ensure_dir(str(parent))
    if ok_all:
        return paths

    fallback_base = "/tmp/appdata"
    _ensure_dir(fallback_base)

    rebased = []
    for p in paths:
        pp = Path(p)
        name = pp.name if pp.suffix else (pp.name or "data")
        rebased.append(str(Path(fallback_base) / name))

    for p in rebased:
        parent = Path(p).parent if Path(p).suffix else Path(p)
        _ensure_dir(str(parent))

    global _FALLBACK_WARNED
    if not _FALLBACK_WARNED and not _SILENCE_PATH_WARN:
        logger.warning("Switched data paths to fallback under %s due to permission issues.", fallback_base)
        _FALLBACK_WARNED = True
    return rebased


# ===== Tokens / models
TELEGRAM_BOT_TOKEN = _env("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN не задан")

TEXT_MODEL_NAME  = _env("TEXT_MODEL_NAME")  or "gemini-2.5-flash"
EMBEDDING_MODEL  = _env("EMBEDDING_MODEL")  or "gemini-embedding-001"
GEMINI_API_KEY   = _env("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY не задан")

# ===== Storage
BASE_DATA      = (_env("BASE_DATA") or "/var/tmp/appdata").rstrip("/")
DOC_FOLDER     = _env("DOC_FOLDER")    or f"{BASE_DATA}/documents"
INDEX_FILE     = _env("FAISS_INDEX")   or f"{BASE_DATA}/index.faiss"
TEXTS_FILE     = _env("TEXTS_FILE")    or f"{BASE_DATA}/texts.pkl"
MANIFEST_FILE  = _env("MANIFEST_FILE") or f"{BASE_DATA}/manifest.json"
USAGE_FILE     = _env("USAGE_FILE")    or f"{BASE_DATA}/usage.json"
DOCMETA_FILE   = _env("DOCMETA_FILE")  or f"{BASE_DATA}/docmeta.json"

# Ensure or fallback
DOC_FOLDER, INDEX_FILE, TEXTS_FILE, MANIFEST_FILE, USAGE_FILE, DOCMETA_FILE = _prepare_paths_or_fallback(
    [DOC_FOLDER, INDEX_FILE, TEXTS_FILE, MANIFEST_FILE, USAGE_FILE, DOCMETA_FILE]
)

# ===== Limits & retrieval
DAILY_FREE_LIMIT   = int(_env("DAILY_FREE_LIMIT") or "10")
ADMIN_USER_IDS     = set(int(x.strip()) for x in _env("ADMIN_USER_IDS").split(",") if x.strip().isdigit())

MAX_OUTPUT_TOKENS  = int(_env("MAX_OUTPUT_TOKENS") or "2048")
RETRIEVAL_K        = int(_env("RETRIEVAL_K") or "6")

CHUNK_MAX_CHARS    = int(_env("CHUNK_MAX_CHARS") or "2000")
CHUNK_MIN_CHARS    = int(_env("CHUNK_MIN_CHARS") or "400")
SUBCHUNK_MAX_CHARS = int(_env("SUBCHUNK_MAX_CHARS") or "1600")
TELEGRAM_MSG_LIMIT = 4096

# ===== TM
TM_SHEET_ID        = _env("TM_SHEET_ID")
TM_SHEET_NAME      = _env("TM_SHEET_NAME") or "Лист1"
TM_SHEET_GID       = _env("TM_SHEET_GID") or "0"
TM_ENABLE          = (_env("TM_ENABLE") or "1") == "1"
TM_SHEET_CSV_URL   = _env("TM_SHEET_CSV_URL")
TM_DEBUG           = (_env("TM_DEBUG") or "0") == "1"

# ===== Modes
RUN_MODE           = (_env("RUN_MODE") or "polling").lower()
PUBLIC_BASE_URL    = (_env("PUBLIC_BASE_URL") or "").rstrip("/")

# ===== OCR defaults (moved here so other modules can import)
OCR_DPI        = int(_env("OCR_DPI") or "300")
OCR_MAX_PAGES  = int(_env("OCR_MAX_PAGES") or "0")  # 0 = all pages
PDF_COMPRESS   = (_env("PDF_COMPRESS") or "1") == "1"
POPPLER_PATH   = (_env("POPPLER_PATH") or "").strip() or None
TESS_LANG      = _env("TESS_LANG") or "rus"
TESS_CONFIG    = _env("TESS_CONFIG") or "--oem 3 --psm 6 -c preserve_interword_spaces=1"
FORCE_OCR = os.getenv("FORCE_OCR", "0") == "1"
