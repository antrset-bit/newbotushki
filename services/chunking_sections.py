
# -*- coding: utf-8 -*-
"""
Semantic, section-aware chunking for Russian legal/business documents.
Drop this file into app/services/chunking_sections.py
"""
from __future__ import annotations
import re
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict, Iterable

# --- Section ontology & synonyms ------------------------------------------------

SECTION_SYNONYMS: Dict[str, Dict[str, List[str]]] = {
    "вознаграждение": {
        "aliases": [
            "стоимость услуг", "оплата", "цена", "гонорар", "финансовые условия",
            "порядок оплаты", "расчеты", "компенсация", "тариф", "стоимость работ"
        ],
        "keywords": ["оплата", "стоимость", "цена", "руб", "₽", "безнал", "счет", "акт", "срок оплаты", "предоплата", "отсрочка", "инвойс"]
    },
    "предмет договора": {
        "aliases": ["предмет", "услуги", "работы", "объем услуг", "описание услуг", "scope"],
        "keywords": ["оказание услуг", "исполнитель", "заказчик", "работы", "услуги"]
    },
    "порядок сдачи-приемки": {
        "aliases": ["сдача и приемка", "приемка услуг", "акты", "акты выполненных работ", "приемка"],
        "keywords": ["акт", "приемка", "сдача", "подписания акта"]
    },
    "ответственность": {
        "aliases": ["ответственность сторон", "штраф", "неустойка", "пеня", "санкции"],
        "keywords": ["неустойка", "штраф", "пеня", "ответственность"]
    },
    "срок действия": {
        "aliases": ["срок действия договора", "сроки", "срок исполнения", "срок"],
        "keywords": ["срок", "дата", "период", "пролонгация"]
    },
    "конфиденциальность": {
        "aliases": ["неразглашение", " NDA ", "конфиденциальная информация"],
        "keywords": ["конфиденциаль", "неразглашен", " NDA "]
    },
    "банковские реквизиты": {
        "aliases": ["реквизиты", "платежные реквизиты"],
        "keywords": ["р/с", "к/с", "ИНН", "КПП", "БИК", "банк"]
    },
    "прочие условия": {
        "aliases": ["заключительные положения", "общие положения", "прочие", "общие условия"],
        "keywords": []
    },
}

# Precompute inverse alias map for quick lookup
def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower().strip(' .,:;–—"«»()[]')

ALIASES_MAP: Dict[str, str] = {}
for canon, data in SECTION_SYNONYMS.items():
    ALIASES_MAP[_normalize(canon)] = canon
    for a in data.get("aliases", []):
        ALIASES_MAP[_normalize(a)] = canon

# --- Regexes -------------------------------------------------------------------

SECTION_HEADER_RE = re.compile(
    r'^\s*(?:раздел|глава|статья)?\s*'
    r'(?P<num>(?:[IVXLC]+|\d+(?:\.\d+)*))?\s*'
    r'[\.\):-]?\s*'
    r'(?P<title>[A-ZА-ЯЁ][^\n]{2,120})\s*$',
    re.IGNORECASE
)

# Lines that are very likely "service" endings / signatures etc.
TRAILING_SECTION_HINTS = re.compile(
    r'(подписи|реквизиты|банковские реквизиты|application|appendix|приложение)',
    re.IGNORECASE
)

@dataclass
class Section:
    title_raw: str
    title_canonical: Optional[str]
    number: Optional[str]
    start_line: int
    end_line: Optional[int] = None

@dataclass
class SectionChunk:
    text: str
    section_title_raw: str
    section_canonical: Optional[str]
    section_number: Optional[str]
    chunk_index_in_section: int

# --- Core utilities ------------------------------------------------------------

def canonical_from_title(title: str) -> Optional[str]:
    norm = _normalize(title)
    if norm in ALIASES_MAP:
        return ALIASES_MAP[norm]
    # fuzzy contains both ways
    for alias_norm, canon in ALIASES_MAP.items():
        if alias_norm in norm or norm in alias_norm:
            return canon
    return None

def find_sections(lines: List[str]) -> List[Section]:
    sections: List[Section] = []
    for i, line in enumerate(lines):
        # Skip too-short lines or service tails
        if len(line.strip()) < 3:
            continue
        if SECTION_HEADER_RE.match(line) and not line.strip().endswith(":"):
            m = SECTION_HEADER_RE.match(line)
            title = m.group("title").strip() if m else line.strip()
            num = m.group("num") if m else None
            sections.append(Section(
                title_raw=title,
                title_canonical=canonical_from_title(title),
                number=num,
                start_line=i
            ))
    # Close sections by next start or end of doc
    for j in range(len(sections) - 1):
        sections[j].end_line = sections[j + 1].start_line
    if sections:
        sections[-1].end_line = len(lines)
        # Heuristic: if last sections are signatures/requisites, keep them but let downstream filter if needed
    return sections

def window_chunks(text: str, max_chars: int = 1600, overlap: int = 250) -> List[str]:
    text = text.strip()
    if not text:
        return []
    parts: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        j = min(n, i + max_chars)
        # try to end at sentence boundary
        k = text.rfind('.', i, j)
        if k == -1:
            k = j
        else:
            k = min(n, k + 1)
        parts.append(text[i:k].strip())
        if k >= n:
            break
        i = max(0, k - overlap)
    return [p for p in parts if p]

def split_by_semantic_sections(text: str) -> List[SectionChunk]:
    lines = text.splitlines()
    secs = find_sections(lines)
    if not secs:
        # Fallback to previous sentence-based split (compatible signature)
        try:
            # Lazy import to avoid hard dependency
            from app.services.chunking import smart_split_text  # type: ignore
            raw_chunks = smart_split_text(text)
        except Exception:
            # naïve fallback
            raw_chunks = re.split(r'(?<=[\.\!\?…])\s+', text)
        out: List[SectionChunk] = []
        for i, c in enumerate(raw_chunks):
            if c and c.strip():
                out.append(SectionChunk(c.strip(), "generic", None, None, i))
        return out

    out: List[SectionChunk] = []
    for s in secs:
        body = "\n".join(lines[s.start_line + 1: s.end_line]).strip()
        if not body:
            continue
        windows = window_chunks(body, max_chars=1600, overlap=250)
        for idx, w in enumerate(windows):
            out.append(SectionChunk(
                text=w,
                section_title_raw=s.title_raw,
                section_canonical=s.title_canonical,
                section_number=s.number,
                chunk_index_in_section=idx
            ))
    return out

def serialize_section_chunks(chunks: List[SectionChunk], doc_name: str) -> Tuple[List[str], List[Dict]]:
    """Return parallel lists: texts, metas for indexing pipelines."""
    texts: List[str] = []
    metas: List[Dict] = []
    for ch in chunks:
        texts.append(ch.text)
        metas.append({
            "doc_name": doc_name,
            "section_title_raw": ch.section_title_raw,
            "section_canonical": ch.section_canonical,
            "section_number": ch.section_number,
            "chunk_index_in_section": ch.chunk_index_in_section,
        })
    return texts, metas

# --- Query expansion helpers ---------------------------------------------------

def expand_query(q: str) -> Tuple[str, Optional[str]]:
    """
    Expand user query with canonical section names & keywords if we detect aliases.
    Returns (expanded_query, detected_canonical_or_None)
    """
    qn = _normalize(q)
    detected = None
    bag: List[str] = []
    for canon, data in SECTION_SYNONYMS.items():
        terms = [canon] + data.get("aliases", [])
        if any(_normalize(t) in qn for t in terms):
            detected = canon if detected is None else detected
            bag.extend(terms + data.get("keywords", []))
    if bag:
        unique = sorted(set(bag), key=lambda x: x.lower())
        return q + " || " + " || ".join(unique), detected
    return q, None
