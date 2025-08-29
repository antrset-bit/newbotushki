
# -*- coding: utf-8 -*-
"""
Hybrid retrieval utilities: simple BM25, score fusion, MMR, and section-boosting.
Drop into app/services/hybrid_retriever.py
"""
from __future__ import annotations
import math, re
from typing import List, Dict, Tuple, Optional

WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9_]+", re.UNICODE)

def tokenize(text: str) -> List[str]:
    return [t.lower() for t in WORD_RE.findall(text or "")]

class BM25:
    def __init__(self, docs: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs = docs
        self.doc_tokens = [tokenize(d) for d in docs]
        self.df: Dict[str, int] = {}
        for toks in self.doc_tokens:
            for w in set(toks):
                self.df[w] = self.df.get(w, 0) + 1
        self.N = len(docs)
        self.avgdl = sum(len(t) for t in self.doc_tokens) / max(1, self.N)

    def idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def score(self, query: str) -> List[float]:
        q = tokenize(query)
        scores = [0.0] * self.N
        for i, toks in enumerate(self.doc_tokens):
            dl = len(toks)
            tf = {}
            for w in toks:
                tf[w] = tf.get(w, 0) + 1
            s = 0.0
            for term in q:
                if term not in tf:
                    continue
                idf = self.idf(term)
                denom = tf[term] + self.k1 * (1 - self.b + self.b * dl / max(1, self.avgdl))
                s += idf * (tf[term] * (self.k1 + 1)) / denom
            scores[i] = s
        return scores

def fuse_scores(cos: List[float], bm: List[float], alpha: float = 0.65) -> List[float]:
    if not bm or len(bm) != len(cos):
        return cos
    if max(bm) > 0:
        bm_norm = [x / max(bm) for x in bm]
    else:
        bm_norm = bm
    if max(cos) > 0:
        cos_norm = [x / max(cos) for x in cos]
    else:
        cos_norm = cos
    return [alpha * c + (1 - alpha) * b for c, b in zip(cos_norm, bm_norm)]

def mmr_select(items: List[Tuple[int, float]], texts: List[str], top_k: int = 5, lambda_: float = 0.7) -> List[int]:
    """
    items: list of (idx, score) after fusion, higher is better.
    returns selected indices (subset of item indices)
    """
    selected: List[int] = []
    cand = [i for i, _ in sorted(items, key=lambda x: x[1], reverse=True)]
    if not cand:
        return selected
    selected.append(cand.pop(0))
    while cand and len(selected) < top_k:
        best_i = None
        best_val = -1e9
        for i in cand:
            # similarity proxy: Jaccard over tokens (no embedding access here)
            a = set(tokenize(texts[i]))
            scores = []
            for j in selected:
                b = set(tokenize(texts[j]))
                inter = len(a & b)
                union = len(a | b) or 1
                sim = inter / union
                scores.append(sim)
            max_sim = max(scores) if scores else 0.0
            # original score:
            orig = next(s for idx, s in items if idx == i)
            val = lambda_ * orig - (1 - lambda_) * max_sim
            if val > best_val:
                best_val = val
                best_i = i
        if best_i is None:
            break
        selected.append(best_i)
        cand.remove(best_i)
    return selected

def ensure_section_boost(selected_ids: List[int],
                         metas: List[Dict],
                         texts: List[str],
                         target_canon: Optional[str],
                         limit_boost: int = 2) -> List[int]:
    """Ensure that chunks from a target canonical section are present among results."""
    if not target_canon:
        return selected_ids
    present = [i for i in selected_ids if (metas[i].get("section_canonical") == target_canon)]
    if present:
        return selected_ids
    # find best candidates of that section (by position heuristic: earlier chunks first)
    candidates = [i for i, m in enumerate(metas) if m.get("section_canonical") == target_canon]
    # add up to limit_boost candidates at front (stable unique)
    out = list(dict.fromkeys(candidates[:limit_boost] + selected_ids))
    return out[:max(len(selected_ids), limit_boost)]
