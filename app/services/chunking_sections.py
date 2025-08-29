# -*- coding: utf-8 -*-
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Optional, Dict

# Ключевые разделы и их синонимы/ключевые слова
SECTION_SYNONYMS: Dict[str, Dict[str, List[str]]] = {
    "вознаграждение": {
        "aliases": [
            "стоимость услуг", "оплата", "цена", "гонорар", "финансовые условия",
            "порядок оплаты", "расчеты", "компенсация", "тариф", "стоимость работ"
        ],
        "keywords": ["оплата", "стоимость", "цена", "руб", "₽", "безнал", "счет", "акт", "срок оплаты", "предоплата", "отсрочка", "инвойс"]
    },
    "срок действия": {
        "aliases": [
            "срок действия договора", "сроки действия", "срок соглашения", "длительность", "период действия",
            "вступает в силу", "вступление в силу", "прекращение", "расторжение", "односторонний отказ"
        ],
        "keywords": [
            "вступает в силу", "действует до", "расторгнут", "расторжение", "прекращение", "срок", "календарных дней",
            "уведомление", "односторонний", "соглашение сторон"
        ]
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
    "банковские реквизиты": {
        "aliases": ["реквизиты", "платежные реквизиты"],
        "keywords": ["р/с", "к/с", "ИНН", "КПП", "БИК", "банк"]
    },
}

def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower().strip(' .,:;–—\"«»()[]')

ALIASES_MAP: Dict[str, str] = {}
for canon, data in SECTION_SYNONYMS.items():
    ALIASES_MAP[_normalize(canon)] = canon
    for a in data.get("aliases", []):
        ALIASES_MAP[_normalize(a)] = canon

SECTION_HEADER_RE = re.compile(
    r'^\s*(?:раздел|глава|статья)?\s*'
    r'(?P<num>(?:[IVXLC]+|\d+(?:\.\d+)*))?\s*'
    r'[\.\):-]?\s*'
    r'(?P<title>[A-ZА-ЯЁ][^\n]{2,120})\s*$',
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

def canonical_from_title(title: str) -> Optional[str]:
    norm = _normalize(title)
    if norm in ALIASES_MAP:
        return ALIASES_MAP[norm]
    for alias_norm, canon in ALIASES_MAP.items():
        if alias_norm in norm or norm in alias_norm:
            return canon
    return None

def find_sections(lines: List[str]) -> List[Section]:
    sections: List[Section] = []
    for i, line in enumerate(lines):
        s = line.strip()
        if len(s) < 3:
            continue
        m = SECTION_HEADER_RE.match(s)
        if m and not s.endswith(":"):
            title = m.group("title").strip()
            num = m.group("num")
            sections.append(Section(
                title_raw=title,
                title_canonical=canonical_from_title(title),
                number=num,
                start_line=i
            ))
    for j in range(len(sections) - 1):
        sections[j].end_line = sections[j + 1].start_line
    if sections:
        sections[-1].end_line = len(lines)
    return sections

def window_chunks(text: str, max_chars: int = 1600, overlap: int = 250) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    max_chars = max(300, int(max_chars))
    overlap = max(0, min(int(overlap), max_chars - 100))
    parts: List[str] = []
    i, n = 0, len(text)
    guard = 0
    while i < n and guard < 100000:
        guard += 1
        j = min(n, i + max_chars)
        k = text.rfind('.', i, j)
        if k == -1 or (j - k) > 200:
            k = j
        else:
            k = min(n, k + 1)
        chunk = text[i:k].strip()
        if chunk:
            parts.append(chunk[:max_chars])
        if k >= n:
            break
        i_next = max(i + 1, k - overlap)
        if i_next <= i:
            i_next = i + max(50, max_chars // 4)
        i = i_next
    parts = [p for p in parts if len(p) >= 20]
    return parts[:10000]

def split_by_semantic_sections(text: str) -> List[SectionChunk]:
    lines = text.splitlines()
    secs = find_sections(lines)
    if not secs:
        parts = [p.strip() for p in re.split(r'(?<=[\.\!\?…])\s+', text) if p.strip()]
        return [SectionChunk(p, "generic", None, None, i) for i, p in enumerate(parts)]
    out: List[SectionChunk] = []
    for s in secs:
        body = "\n".join(lines[s.start_line + 1: s.end_line]).strip()
        if not body:
            continue
        windows = window_chunks(body, max_chars=1600, overlap=250)
        if len(windows) > 1000:
            windows = windows[:1000]
        for idx, w in enumerate(windows):
            out.append(SectionChunk(
                text=w,
                section_title_raw=s.title_raw,
                section_canonical=s.title_canonical,
                section_number=s.number,
                chunk_index_in_section=idx
            ))
    return out

def serialize_section_chunks(chunks: List[SectionChunk], doc_name: str):
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

def expand_query(q: str) -> tuple[str, str | None]:
    """
    Возвращает (расширенный_запрос, целевой_канон_если_обнаружен).
    """
    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", s or "").strip().lower()
    qn = _norm(q)
    detected = None
    bag: List[str] = []
    for canon, data in SECTION_SYNONYMS.items():
        terms = [canon] + data.get("aliases", [])
        if any(_norm(t) in qn for t in terms):
            if detected is None:
                detected = canon
            bag.extend(terms + data.get("keywords", []))
    if bag:
        uniq = []
        seen = set()
        for t in bag:
            n = _norm(t)
            if n and n not in seen:
                seen.add(n)
                uniq.append(t)
        return q + " || " + " || ".join(uniq), detected
    return q, None
