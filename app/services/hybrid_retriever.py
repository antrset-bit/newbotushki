# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Dict, Tuple, Optional
import numpy as np
import re

from app.services.indexing import load_index, load_metas
from app.utils.files import load_texts
from app.services.embeddings import get_embedding
from app.services.chunking_sections import expand_query
from app.services.hybrid_retriever import BM25, fuse_scores, mmr_select, ensure_section_boost

def _safe_unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype="float32").reshape(-1)
    n = float(np.linalg.norm(v)) or 1.0
    return (v / n).astype("float32")

def _faiss_topn_cosine(q_vec: np.ndarray, top_n: int) -> Tuple[List[int], List[float]]:
    index = load_index()
    texts = load_texts()
    if index is None or not texts:
        return [], []
    k = max(1, min(top_n, len(texts)))
    D, I = index.search(np.array([q_vec], dtype="float32"), k=k)
    ids = [int(i) for i in I[0] if 0 <= i < len(texts)]
    sims = []
    for d in D[0][:len(ids)]:
        sims.append(1.0 / (1.0 + float(d)))  # L2 -> pseudo-cosine
    return ids, sims

def retrieve_hybrid(user_query: str,
                    top_k: int = 6,
                    pool_k: int = 50,
                    section_boost: bool = True) -> Tuple[List[int], List[str], List[Dict]]:
    texts = load_texts()
    metas = load_metas()
    if not texts:
        return [], [], []
    q_expanded, target_canon = expand_query(user_query)

    q_vec = _safe_unit(np.array(get_embedding(q_expanded), dtype="float32"))
    faiss_ids, faiss_sims = _faiss_topn_cosine(q_vec, top_n=min(pool_k, len(texts)))
    if not faiss_ids:
        return [], [], []

    bm = BM25(texts)
    bm_scores_all = bm.score(q_expanded)
    bm_scores = [bm_scores_all[i] for i in faiss_ids]

    fused = fuse_scores(faiss_sims, bm_scores, alpha=0.65)
    ranked_global = [(faiss_ids[i], s) for i, s in enumerate(fused)]
    selected_global = mmr_select(ranked_global, texts, top_k=top_k, lambda_=0.7)

    if section_boost and target_canon and metas:
        selected_global = ensure_section_boost(selected_global, metas, texts, target_canon, limit_boost=2)

    out_ids = [i for i in selected_global if 0 <= i < len(texts)]
    out_texts = [texts[i] for i in out_ids]
    out_metas  = [metas[i] if i < len(metas) else {} for i in out_ids]
    return out_ids, out_texts, out_metas

def build_section_answer(user_query: str, max_chars: int = 700) -> Optional[str]:
    ids, texts, metas = retrieve_hybrid(user_query, top_k=8, pool_k=80, section_boost=True)
    if not ids:
        return None

    _, target_canon = expand_query(user_query)

    # Выбираем документ и (по возможности) нужный канон раздела
    best_doc = None
    best_section = None
    for i, m in zip(ids, metas):
        if not m: 
            continue
        canon = m.get("section_canonical")
        if target_canon and canon == target_canon:
            best_doc = m.get("doc_name")
            best_section = canon
            break
    if best_doc is None and metas:
        best_doc = metas[0].get("doc_name")

    texts_all = load_texts()
    all_metas = load_metas()
    indices = [j for j, m in enumerate(all_metas) if (m.get("doc_name")==best_doc) and (best_section is None or m.get("section_canonical")==best_section)]

    # сортируем по номеру/порядку чанка в разделе
    def _as_int(x):
        try: return int(x)
        except Exception: return 0
    indices_sorted = sorted(indices, key=lambda j: (all_metas[j].get("section_number") or "", _as_int(all_metas[j].get("chunk_index_in_section") or 0)))

    # собираем выдержку
    buf = []
    total = 0
    for j in indices_sorted:
        t = (texts_all[j] or "").strip()
        if not t: 
            continue
        if total + len(t) > max_chars and buf:
            break
        buf.append(t)
        total += len(t)
        if total >= max_chars:
            break
    excerpt = " ".join(buf).strip()
    excerpt = re.sub(r"\s+", " ", excerpt)

    doc_name = best_doc or "Документ"
    if not excerpt:
        return f"Название документа: {doc_name}.\nВыдержка из документа: (не найдено)"
    return f"Название документа: {doc_name}.\nВыдержка из документа: {excerpt}"
