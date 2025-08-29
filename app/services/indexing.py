# -*- coding: utf-8 -*-
from __future__ import annotations
import os, time, json, logging
from typing import List, Tuple
import numpy as np
import faiss

from app.config import INDEX_FILE, TEXTS_FILE, DOCMETA_FILE
from app.services.extract import extract_text_from_pdf, extract_text_from_docx, extract_text_from_txt
from app.services.chunking_sections import split_by_semantic_sections, serialize_section_chunks
from app.services.embeddings import get_embedding
from app.utils.files import load_texts, save_texts, sha256_file

logger = logging.getLogger("semantic-bot")

# Хранилище метаданных чанков (рядом с TEXTS_FILE)
METAS_FILE = TEXTS_FILE + ".meta.json"

MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "400000"))
MAX_CHUNKS_PER_DOC = int(os.getenv("MAX_CHUNKS_PER_DOC", "300"))
EMB_SLEEP_SEC = float(os.getenv("EMB_SLEEP_SEC", "0.01"))

def _sanitize_text(t: str) -> str:
    import re, unicodedata
    t = unicodedata.normalize("NFKC", t or "")
    t = t.replace("\x00", " ")
    t = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]", " ", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    lines = []
    for line in t.splitlines():
        lines.append(line[:10000])
    t = "\n".join(lines)
    if len(t) > MAX_TEXT_CHARS:
        logger.warning("Text too long (%s), truncating to %s", len(t), MAX_TEXT_CHARS)
        t = t[:MAX_TEXT_CHARS]
    return t

def _safe_get_embedding(ch: str) -> np.ndarray:
    v = get_embedding(ch)
    arr = np.asarray(v, dtype="float32").reshape(-1)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("Empty/invalid embedding")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Non-finite values in embedding")
    return arr

def load_docmeta() -> dict:
    if os.path.exists(DOCMETA_FILE):
        try:
            with open(DOCMETA_FILE, "r", encoding="utf-8") as f:
                return json.load(f) or {}
        except Exception:
            return {}
    return {}

def save_docmeta(meta: dict) -> None:
    os.makedirs(os.path.dirname(DOCMETA_FILE), exist_ok=True)
    with open(DOCMETA_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def load_metas() -> List[dict]:
    if os.path.exists(METAS_FILE):
        try:
            with open(METAS_FILE, "r", encoding="utf-8") as f:
                return json.load(f) or []
        except Exception:
            return []
    return []

def save_metas(metas: List[dict]) -> None:
    os.makedirs(os.path.dirname(METAS_FILE), exist_ok=True)
    with open(METAS_FILE, "w", encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False)

def load_index() -> faiss.Index | None:
    if os.path.exists(INDEX_FILE):
        return faiss.read_index(INDEX_FILE)
    return None

def save_index(index: faiss.Index) -> None:
    os.makedirs(os.path.dirname(INDEX_FILE), exist_ok=True)
    faiss.write_index(index, INDEX_FILE)

def ensure_index(dim: int) -> faiss.Index:
    index = load_index()
    try:
        if index is not None:
            cur_dim = getattr(index, 'd', None)
            if cur_dim is None or cur_dim != dim:
                logger.warning("FAISS dim mismatch: have %s, need %s — rebuilding index.", cur_dim, dim)
                index = faiss.IndexFlatL2(dim)
        else:
            index = faiss.IndexFlatL2(dim)
    except Exception as e:
        logger.warning("Error loading existing FAISS index, creating new: %r", e)
        index = faiss.IndexFlatL2(dim)
    return index

def index_file(file_path: str) -> Tuple[int, int]:
    ext = (file_path.lower().rsplit(".", 1)[-1] if "." in file_path else "").strip()
    if ext == "pdf":
        text = extract_text_from_pdf(file_path)
    elif ext == "docx":
        text = extract_text_from_docx(file_path)
    elif ext == "txt":
        text = extract_text_from_txt(file_path)
    else:
        raise ValueError("Неподдерживаемое расширение файла.")

    # Сдампим сырой текст (для отладки)
    try:
        _dump_base = os.environ.get("DUMP_FOLDER") or "/var/tmp/appdata/dumps"
        os.makedirs(_dump_base, exist_ok=True)
        _fname = os.path.basename(file_path)
        _base, _ = os.path.splitext(_fname)
        _hash = sha256_file(file_path)[:8]
        _dump_name = f"{_base}_{_hash}.txt"
        _dump_path = os.path.join(_dump_base, _dump_name)
        with open(_dump_path, "w", encoding="utf-8") as _f:
            _f.write(text or "")
        logger.info("Dumped raw text before indexing: %s", _dump_path)
    except Exception as _e:
        logger.warning("Failed to dump raw text before indexing: %r", _e)

    text = _sanitize_text(text or "")
    if not text.strip():
        cur = load_index()
        return (0, getattr(cur, "ntotal", 0) if cur else 0)

    # Разбиение по разделам и окна
    section_chunks = split_by_semantic_sections(text)
    chunks, metas = serialize_section_chunks(section_chunks, os.path.basename(file_path))
    if not chunks:
        cur = load_index()
        return (0, getattr(cur, "ntotal", 0) if cur else 0)
    if len(chunks) > MAX_CHUNKS_PER_DOC:
        chunks = chunks[:MAX_CHUNKS_PER_DOC]
        metas = metas[:MAX_CHUNKS_PER_DOC]

    # Первая размерность
    first_vec = None
    for ch in chunks:
        try:
            first_vec = _safe_get_embedding(ch)
            break
        except Exception:
            continue
    if first_vec is None:
        cur = load_index()
        return (0, getattr(cur, "ntotal", 0) if cur else 0)

    index = ensure_index(len(first_vec))
    texts = load_texts()
    all_metas = load_metas()

    # Потоковое добавление
    added = 0
    for i, ch in enumerate(chunks):
        try:
            vec = _safe_get_embedding(ch).reshape(1, -1)
            try:
                index.add(vec)
            except Exception:
                index = faiss.IndexFlatL2(vec.shape[1])
                index.add(vec)
            texts.append(ch)
            all_metas.append(metas[i])
            added += 1
            if EMB_SLEEP_SEC > 0:
                time.sleep(EMB_SLEEP_SEC)
        except Exception:
            continue

    save_index(index)
    save_texts(texts)
    save_metas(all_metas)

    # Привязка чанков к документу
    file_hash = sha256_file(file_path)
    meta = load_docmeta()
    base_ntotal = getattr(index, "ntotal", 0) - added
    if base_ntotal < 0: base_ntotal = 0
    added_ids = list(range(base_ntotal, base_ntotal + added))
    rec = meta.get(file_hash, {"fname": os.path.basename(file_path), "time": int(time.time()), "chunks": []})
    rec["fname"] = os.path.basename(file_path)
    rec["time"] = int(time.time())
    rec["chunks"] = sorted(set((rec.get("chunks") or []) + added_ids))
    meta[file_hash] = rec
    save_docmeta(meta)

    total = getattr(index, "ntotal", len(texts))
    return (added, total)


# --- Backwards-compat shim for old imports ---
def retrieve_chunks(query: str, k: int = 6):
    """
    Совместимость со старым импортом: app.services.indexing.retrieve_chunks
    Возвращает список текстов-чанков (top-k), используя гибридный ретривер.
    """
    try:
        from app.services.retrieval import retrieve_hybrid
        ids, texts, metas = retrieve_hybrid(query, top_k=int(k) if k else 6, pool_k=max(10, int(k)*8 if k else 48))
        return texts
    except Exception as e:
        # как запасной вариант — простой векторный поиск, если retrieval недоступен
        try:
            index = load_index()
            texts_all = load_texts()
            if index is None or not texts_all:
                return []
            from app.services.embeddings import get_embedding
            import numpy as np
            q = np.asarray(get_embedding(query), dtype="float32").reshape(1, -1)
            k2 = max(1, min(int(k) if k else 6, len(texts_all)))
            D, I = index.search(q, k=k2)
            ids = [i for i in I[0] if 0 <= i < len(texts_all)]
            return [texts_all[i] for i in ids]
        except Exception:
            return []

