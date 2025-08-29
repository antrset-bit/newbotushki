from app.services.chunking_sections import split_by_semantic_sections, serialize_section_chunks, expand_query
import os, time, json, logging
from typing import List, Tuple
import numpy as np
import faiss

from app.config import INDEX_FILE, TEXTS_FILE, DOC_FOLDER, DOCMETA_FILE
from app.services.extract import extract_text_from_pdf, extract_text_from_docx, extract_text_from_txt
from app.services.chunking import smart_split_text
from app.services.embeddings import get_embedding
from app.utils.files import load_texts, save_texts, sha256_file
logger = logging.getLogger("semantic-bot")

def load_docmeta() -> dict:
    if os.path.exists(DOCMETA_FILE):
        try:
            with open(DOCMETA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_docmeta(meta: dict):
    with open(DOCMETA_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def load_index() -> faiss.Index | None:
    if os.path.exists(INDEX_FILE):
        return faiss.read_index(INDEX_FILE)
    return None

def save_index(index: faiss.Index):
    faiss.write_index(index, INDEX_FILE)

def ensure_index(dim: int) -> faiss.Index:
    index = load_index()
    if index is None:
        index = faiss.IndexFlatL2(dim)
    return index

def index_file(file_path: str) -> Tuple[int, int]:
    ext = file_path.lower().split(".")[-1]
    if ext == "pdf":
        text = extract_text_from_pdf(file_path)
    elif ext == "docx":
        text = extract_text_from_docx(file_path)
    elif ext == "txt":
        text = extract_text_from_txt(file_path)
    else:
        raise ValueError("Неподдерживаемое расширение файла.")
    # --- dump recognized/extracted text before vectorization (for QA) ---
    try:
        # Base for dumps: env DUMP_FOLDER or alongside DOC_FOLDER
        try:
            from app.config import DOC_FOLDER as _DOCFOLDER
            _dump_base = os.environ.get("DUMP_FOLDER") or os.path.join(os.path.dirname(_DOCFOLDER), "dumps")
        except Exception:
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
    # --- end dump ---

    if not text:
        logger.warning("Пустой текст после извлечения: %s", os.path.basename(file_path))
        return (0, 0)

    section_chunks = split_by_semantic_sections(text)
chunks, metas = serialize_section_chunks(section_chunks, os.path.basename(file_path))
    if not chunks:
        return (0, 0)

    texts = load_texts()
    first_vec = None
    for ch in chunks:
        try:
            first_vec = get_embedding(ch)
            break
        except Exception as e:
            logger.warning("Пропущен чанк при определении размерности: %s", repr(e))
            continue
    if first_vec is None:
        raise RuntimeError("Не удалось получить эмбеддинги.")
    index = ensure_index(len(first_vec))
    new_embeddings, new_texts = [], []
    for ch in chunks:
        try:
            emb = get_embedding(ch)
            new_embeddings.append(emb)
            new_texts.append(ch)
            time.sleep(0.02)
        except Exception as e:
            logger.warning("Пропущен чанк из-за ошибки эмбеддинга: %s", repr(e))
            continue
    if not new_embeddings:
        existing = load_index()
        return (0, existing.ntotal if existing else 0)

    base_ntotal = 0
    existing_index = load_index()
    if existing_index is not None:
        base_ntotal = getattr(existing_index, "ntotal", 0)

    mat = np.vstack(new_embeddings).astype("float32")
    index.add(mat)
    save_index(index)

    texts.extend(new_texts)
    save_texts(texts)

    file_hash = sha256_file(file_path)
    meta = load_docmeta()
    added_ids = list(range(base_ntotal, base_ntotal + len(new_embeddings)))
    rec = meta.get(file_hash, {"fname": os.path.basename(file_path), "time": int(time.time()), "chunks": []})
    rec["fname"] = os.path.basename(file_path)
    rec["time"] = int(time.time())
    rec["chunks"] = sorted(set((rec.get("chunks") or []) + added_ids))
    meta[file_hash] = rec
    save_docmeta(meta)

    total = index.ntotal if hasattr(index, "ntotal") else len(texts)
    return (len(new_texts), total)

def retrieve_chunks(query: str, k: int = 6) -> List[str]:
    if not (os.path.exists(INDEX_FILE) and os.path.exists(TEXTS_FILE)):
        return []
    from app.services.embeddings import get_embedding as _ge
    q_emb = _ge(query)
    index = load_index()
    texts = load_texts()
    if index is None or len(texts) == 0:
        return []
    D, I = index.search(np.array([q_emb], dtype="float32"), k=min(k, len(texts)))
    ids = [i for i in I[0] if 0 <= i < len(texts)]
    return [texts[i] for i in ids]

def rebuild_faiss_from_texts(texts: List[str]) -> int:
    if not texts:
        if os.path.exists(INDEX_FILE):
            os.remove(INDEX_FILE)
        save_texts([])
        return 0

    dim_vec = None
    for t in texts:
        try:
            from app.services.embeddings import get_embedding as _ge
            dim_vec = _ge(t)
            break
        except Exception:
            continue
    if dim_vec is None:
        raise RuntimeError("Не удалось получить эмбеддинг ни для одного чанка.")

    index = faiss.IndexFlatL2(len(dim_vec))
    embs = []
    for t in texts:
        try:
            from app.services.embeddings import get_embedding as _ge
            embs.append(_ge(t))
            time.sleep(0.01)
        except Exception as e:
            logger.warning("rebuild: пропущен чанк из-за ошибки эмбеддинга: %r", e)
            continue
    if not embs:
        raise RuntimeError("rebuild: нет ни одного валидного эмбеддинга.")
    mat = np.vstack(embs).astype("float32")
    index.add(mat)
    save_index(index)
    save_texts(texts)
    return index.ntotal

def delete_document_from_training(file_hash: str, also_remove_file: bool = False) -> tuple[bool, str]:
    meta = load_docmeta()
    if file_hash not in meta:
        return False, "Не найдено метаданных по этому файлу. Воспользуйтесь /admin_rebuild для полной пересборки."

    fname_hint = meta[file_hash].get("fname", "")
    texts = load_texts()
    drop_ids = set(meta[file_hash].get("chunks") or [])
    if not texts:
        return False, "texts.pkl пуст — нечего удалять."
    keep_texts = [t for i, t in enumerate(texts) if i not in drop_ids]

    try:
        total = rebuild_faiss_from_texts(keep_texts)
    except Exception as e:
        return False, f"Ошибка пересборки индекса: {e!r}"

    meta.pop(file_hash, None)
    for k in list(meta.keys()):
        if "chunks" in meta[k]:
            meta[k].pop("chunks", None)
    save_docmeta(meta)

    if also_remove_file and fname_hint:
        try:
            for name in os.listdir(DOC_FOLDER):
                if name == fname_hint:
                    os.unlink(os.path.join(DOC_FOLDER, name))
                    break
        except Exception as e:
            pass

    return True, f"Документ удалён из обучения. Текущих векторов в индексе: {total}."

def full_reindex_all_documents() -> tuple[int, list[str]]:
    if os.path.exists(INDEX_FILE):
        os.remove(INDEX_FILE)
    save_texts([])
    save_docmeta({})

    errors = []
    total_added = 0
    for name in sorted(os.listdir(DOC_FOLDER)):
        path = os.path.join(DOC_FOLDER, name)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext not in (".pdf", ".docx", ".txt"):
            continue
        try:
            added, _ = index_file(path)
            total_added += added
            time.sleep(0.05)
        except Exception as e:
            errors.append(f"{name}: {e!r}")
    idx = load_index()
    total_vectors = getattr(idx, "ntotal", 0) if idx else 0
    return total_vectors, errors
