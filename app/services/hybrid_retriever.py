
"""
HybridRetriever — улучшенная логика поиска по проиндексированным документам,
ориентирована на «точно найти то, что есть», без зависимостей.

Подход:
1) Нормализация и токенизация (рус/eng), лемматизация-псевдо (стемминг по суффиксам).
2) BM25 (Okapi) как первичный скоринг.
3) TF‑IDF косинусное сходство как вторичный скоринг.
4) Реранк: score = α * bm25 + (1-α) * (100 * cos), плюс бусты за точные фразы и совпадения в заголовках.
5) Поиск устойчив к формам слов (мн/падежи) благодаря лёгкому стеммингу.
6) Поддержка сохранения/загрузки индекса в JSON.

Как использовать:
    from app.retriever import HybridRetriever

    retr = HybridRetriever(alpha=0.7, k1=1.6, b=0.68)
    # docs = [{"id": "doc1#0", "text": "...chunk text..."}, ...]
    retr.build(docs)               # или retr.load(path)
    hits = retr.search("запрос пользователя", top_k=5)
    # hits -> список словарей: {"id","score","text","highlights"}

Совет:
- Индексируйте чанки, где заголовок склеен с контентом (как в новом chunker.py).
- Давайте запросу несколько синонимов через OR: "договор ИЛИ контракт ИЛИ соглашение".
"""

import re, math, json
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional

WORD = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+", re.UNICODE)

RU_SUFFIXES = [
    "ими","ами","ями","иях","иях","иях","ого","ему","ому","ыми","ыми",
    "ыми","ами","ям","ев","ов","ие","ые","ая","яя","ое","ее","ий","ый",
    "ой","ем","им","ям","ах","ях","ом","ах","ах","ам","ям","ю","е","а","ы","и","о","у","я"
]

EN_SUFFIXES = ["ing","ed","ly","es","s","er","or","al","ive","tion","ions","ment","ments"]

def normalize(text: str) -> str:
    t = text.replace("\r\n","\n").replace("\r","\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def simple_stem(token: str) -> str:
    t = token.lower()
    if re.match(r"[A-Za-z]", t):
        for suf in EN_SUFFIXES:
            if t.endswith(suf) and len(t) - len(suf) >= 3:
                return t[: -len(suf)]
        return t
    else:
        for suf in RU_SUFFIXES:
            if t.endswith(suf) and len(t) - len(suf) >= 3:
                return t[: -len(suf)]
        return t

def tokenize(text: str) -> List[str]:
    return [m.group(0) for m in WORD.finditer(text)]

def analyze(text: str) -> List[str]:
    return [simple_stem(t) for t in tokenize(text)]

def split_heading(text: str) -> Tuple[str, str]:
    # Если первая строка похожа на заголовок — вернём (heading, body)
    lines = text.strip().split("\n")
    if not lines:
        return "", text
    first = lines[0].strip()
    if (len(first) <= 120 and sum(1 for c in first if c.isupper()) >= max(1, len(first)//5)) or first.startswith("#"):
        return first, "\n".join(lines[1:]).strip()
    return "", text

class HybridRetriever:
    def __init__(self, alpha: float = 0.7, k1: float = 1.6, b: float = 0.68, phrase_boost: float = 8.0, heading_boost: float = 1.25):
        self.alpha = alpha
        self.k1 = k1
        self.b = b
        self.phrase_boost = phrase_boost
        self.heading_boost = heading_boost

        self.docs: List[Dict] = []
        self.N = 0
        self.avgdl = 0.0
        self.df: Dict[str, int] = defaultdict(int)         # document frequency
        self.postings: Dict[str, List[Tuple[int,int]]] = defaultdict(list)  # term -> list of (doc_id, tf)
        self.doc_vecs: List[Dict[str, float]] = []         # TF-IDF vectors
        self.doc_norms: List[float] = []
        self.headings: List[str] = []

    def build(self, docs: List[Dict[str,str]]):
        self.docs = []
        self.postings.clear(); self.df.clear()
        self.doc_vecs.clear(); self.doc_norms.clear(); self.headings.clear()

        lengths = []
        for idx, d in enumerate(docs):
            text = normalize(d.get("text",""))
            heading, _ = split_heading(text)
            self.headings.append(heading or "")
            terms = analyze(text)
            tf = Counter(terms)
            self.docs.append({"id": d.get("id", str(idx)), "text": text, "tf": tf, "len": sum(tf.values())})
            lengths.append(self.docs[-1]["len"])

        self.N = len(self.docs)
        self.avgdl = (sum(lengths) / self.N) if self.N else 0.0

        # DF & postings
        for doc_id, d in enumerate(self.docs):
            for t, f in d["tf"].items():
                self.df[t] += 1
        for doc_id, d in enumerate(self.docs):
            for t, f in d["tf"].items():
                self.postings[t].append((doc_id, f))

        # Build TF-IDF vectors
        for doc_id, d in enumerate(self.docs):
            vec = {}
            for t, f in d["tf"].items():
                idf = math.log( (self.N - self.df[t] + 0.5) / (self.df[t] + 0.5) + 1.0 )
                vec[t] = (1 + math.log(1+f)) * idf
            norm = math.sqrt(sum(v*v for v in vec.values())) or 1.0
            self.doc_vecs.append(vec)
            self.doc_norms.append(norm)

    def save(self, path: str):
        data = {
            "alpha": self.alpha, "k1": self.k1, "b": self.b,
            "phrase_boost": self.phrase_boost, "heading_boost": self.heading_boost,
            "N": self.N, "avgdl": self.avgdl,
            "docs": [{"id": d["id"], "text": d["text"]} for d in self.docs]
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def load(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.alpha = data.get("alpha", self.alpha)
        self.k1 = data.get("k1", self.k1)
        self.b = data.get("b", self.b)
        self.phrase_boost = data.get("phrase_boost", self.phrase_boost)
        self.heading_boost = data.get("heading_boost", self.heading_boost)
        docs = data["docs"]
        self.build(docs)

    # --- Scoring pieces ---
    def _bm25_scores(self, query_terms: List[str]) -> List[float]:
        scores = [0.0]*self.N
        q_tf = Counter(query_terms)
        for t in q_tf.keys():
            if t not in self.df:
                continue
            idf = math.log( (self.N - self.df[t] + 0.5) / (self.df[t] + 0.5) + 1.0 )
            for doc_id, f in self.postings[t]:
                dl = self.docs[doc_id]["len"] or 1
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                score = idf * (f * (self.k1 + 1)) / denom
                scores[doc_id] += score
        return scores

    def _tfidf_cos_scores(self, query_terms: List[str]) -> List[float]:
        q_tf = Counter(query_terms)
        q_vec = {}
        for t, f in q_tf.items():
            if t not in self.df:
                continue
            idf = math.log( (self.N - self.df[t] + 0.5) / (self.df[t] + 0.5) + 1.0 )
            q_vec[t] = (1 + math.log(1+f)) * idf
        q_norm = math.sqrt(sum(v*v for v in q_vec.values())) or 1.0

        scores = [0.0]*self.N
        for doc_id in range(self.N):
            dot = 0.0
            dvec = self.doc_vecs[doc_id]
            for t, w in q_vec.items():
                if t in dvec:
                    dot += w * dvec[t]
            scores[doc_id] = dot / (q_norm * self.doc_norms[doc_id])
        return scores

    def _phrase_boosts(self, query: str) -> List[float]:
        boosts = [0.0]*self.N
        q = query.strip().lower()
        if len(q) < 3:
            return boosts
        for i, d in enumerate(self.docs):
            text = d["text"].lower()
            if q in text:
                boosts[i] += self.phrase_boost
        return boosts

    def _heading_boosts(self, query_terms: List[str]) -> List[float]:
        boosts = [0.0]*self.N
        for i, h in enumerate(self.headings):
            if not h: continue
            head_terms = set(analyze(h))
            if any(t in head_terms for t in query_terms):
                boosts[i] += self.heading_boost
        return boosts

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        query_norm = normalize(query)
        query_terms = analyze(query_norm)
        if not query_terms:
            return []

        bm25 = self._bm25_scores(query_terms)
        tfidf = self._tfidf_cos_scores(query_terms)
        phrase = self._phrase_boosts(query_norm)
        hboost = self._heading_boosts(query_terms)

        results = []
        for i in range(self.N):
            score = self.alpha * bm25[i] + (1 - self.alpha) * (100.0 * tfidf[i]) + phrase[i] + hboost[i]
            results.append((i, score))

        results.sort(key=lambda x: x[1], reverse=True)
        out = []
        for idx, sc in results[:top_k]:
            out.append({
                "id": self.docs[idx]["id"],
                "score": round(sc, 4),
                "text": self.docs[idx]["text"],
                "highlights": self._make_highlights(self.docs[idx]["text"], query_norm)
            })
        return out

    def _make_highlights(self, text: str, query: str, window: int = 120) -> List[str]:
        # Простая подсветка: берём окна вокруг точных вхождений слов из запроса (в исходной форме)
        low = text.lower()
        q_tokens = [q for q in set(tokenize(query)) if len(q) >= 3]
        spans = []
        for qt in q_tokens:
            start = 0
            while True:
                pos = low.find(qt.lower(), start)
                if pos == -1: break
                a = max(0, pos - window); b = min(len(text), pos + len(qt) + window)
                spans.append((a,b)); start = pos + len(qt)
        # Слить пересекающиеся
        spans.sort()
        merged = []
        for s in spans:
            if not merged or s[0] > merged[-1][1] + 10:
                merged.append(list(s))
            else:
                merged[-1][1] = max(merged[-1][1], s[1])
        return [text[a:b] for a,b in merged] or [text[:min(len(text), 2*window)]]
