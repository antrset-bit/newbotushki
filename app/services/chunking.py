import re
from typing import List

from app.config import CHUNK_MAX_CHARS, CHUNK_MIN_CHARS, SUBCHUNK_MAX_CHARS

CONTRACT_SECTIONS = [
    "Предмет договора","Права и обязанности сторон","Обязанности сторон","Гарантии сторон","Ответственность сторон",
    "Срок действия договора","Финансовые условия","Стоимость услуг и порядок оплаты","Порядок оплаты","Термины и определения",
    "Прочие условия","Обстоятельства непреодолимой силы","Форс-мажор","Конфиденциальность","Право на использование изображений",
    "Порядок использования результатов интеллектуальной деятельности","Заверения обстоятельства",
    "Адрес и банковские реквизиты","Реквизиты и подписи сторон","Подписи сторон"
]
POSITION_SECTIONS = [
    "Общие положения","Цели и задачи","Предмет регулирования","Термины и определения","Функции","Права и обязанности",
    "Права организации","Обязанности организации","Ответственность","Порядок взаимодействия","Организация работы",
    "Порядок внесения изменений","Заключительные положения"
]
GENERIC_SECTIONS = [
    "Введение","Общие положения","Термины и определения","Порядок","Права и обязанности","Права","Обязанности","Ответственность",
    "Срок действия","Финансовые условия","Порядок оплаты","Конфиденциальность","Прочие условия","Заключительные положения","Приложения"
]

HEAD_NUM_RE = re.compile(r"^(?:раздел|глава|section|chapter)\s+\d+[.:)]?$", re.IGNORECASE|re.MULTILINE)
HEAD_NUM2_RE = re.compile(r"^\d+(?:\.\d+)*[.)]?\s+\S+", re.MULTILINE)
HEAD_ROMAN_RE = re.compile(r"^(?:[IVXLCDM]+)[\.\)]\s+\S+", re.IGNORECASE|re.MULTILINE)

def guess_doc_type(text: str) -> str:
    head = text[:5000].lower()
    if "договор" in head:
        return "contract"
    if "положение" in head:
        return "position"
    return "generic"

def is_all_caps_cyr(line: str) -> bool:
    s = line.strip()
    if len(s) < 3 or len(s) > 120:
        return False
    letters = [ch for ch in s if ch.isalpha()]
    if not letters:
        return False
    upp = sum(1 for ch in letters if ch.upper() == ch and "а" <= ch.lower() <= "я")
    return upp >= max(3, int(0.7 * len(letters)))

def find_headings(text: str, headings: List[str]) -> List[tuple[int,str]]:
    res = []
    for h in headings:
        for m in re.finditer(rf"(?im)^\s*{re.escape(h)}\s*$", text):
            res.append((m.start(), h))
    for m in HEAD_NUM_RE.finditer(text):
        res.append((m.start(), text[m.start():m.end()]))
    for m in HEAD_NUM2_RE.finditer(text):
        res.append((m.start(), text[m.start():m.end()].strip()))
    for m in HEAD_ROMAN_RE.finditer(text):
        res.append((m.start(), text[m.start():m.end()].strip()))
    for m in re.finditer(r"(?m)^(?P<line>.+)$", text):
        line = m.group("line")
        if is_all_caps_cyr(line):
            res.append((m.start(), line.strip()))
    uniq = {off: ttl for off, ttl in res}
    return sorted(uniq.items(), key=lambda x: x[0])

def split_by_sections(text: str, headings: List[str]) -> List[tuple[str,str]]:
    marks = find_headings(text, headings)
    if not marks:
        return [("", text.strip())]
    chunks = []
    for i, (start, title) in enumerate(marks):
        end = marks[i+1][0] if i+1 < len(marks) else len(text)
        chunk = text[start:end].strip()
        lines = chunk.splitlines()
        if lines:
            first_line = lines[0].strip()
            if len(first_line) <= 180:
                title = first_line
        chunks.append((title.strip(), chunk))
    return chunks

def split_long_chunk(title: str, body: str) -> List[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    out = []
    cur = title + "\n"
    for p in paras:
        add_len = 2 + len(p) if cur.strip() != title else len(p)
        if len(cur) + add_len > SUBCHUNK_MAX_CHARS and cur.strip():
            out.append(cur.strip())
            cur = title + "\n" + p
        else:
            if cur.strip() == title:
                cur = title + "\n" + p
            else:
                cur += "\n\n" + p
    if cur.strip():
        out.append(cur.strip())
    return out

def smart_split_text(text: str) -> List[str]:
    if not text or len(text.strip()) == 0:
        return []
    text = re.sub(r"\r\n?", "\n", text)

    doc_type = guess_doc_type(text)
    base_sections = CONTRACT_SECTIONS if doc_type=="contract" else POSITION_SECTIONS if doc_type=="position" else GENERIC_SECTIONS

    section_chunks = split_by_sections(text, base_sections)

    normalized: List[str] = []
    buf = ""
    for title, chunk in section_chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        if len(chunk) < CHUNK_MIN_CHARS and buf:
            buf += "\n\n" + chunk
            continue
        if buf:
            normalized.append(buf)
        buf = chunk
    if buf:
        normalized.append(buf)

    final_chunks: List[str] = []
    for ch in normalized:
        if len(ch) <= CHUNK_MAX_CHARS:
            final_chunks.append(ch)
        else:
            lines = ch.splitlines()
            title = lines[0].strip() if lines else "Раздел"
            body = "\n".join(lines[1:]).strip() if len(lines) > 1 else ch
            parts = split_long_chunk(title, body)
            for p in parts:
                if len(p) > CHUNK_MAX_CHARS:
                    sents = re.split(r"(?<=[.!?…])\s+", p)
                    tmp = ""
                    for s in sents:
                        if len(tmp) + len(s) + 1 > CHUNK_MAX_CHARS:
                            if tmp.strip():
                                final_chunks.append(tmp.strip())
                            tmp = s
                        else:
                            tmp = (tmp + " " + s).strip()
                    if tmp.strip():
                        final_chunks.append(tmp.strip())
                else:
                    final_chunks.append(p.strip())

    if not final_chunks:
        sents = re.split(r"(?<=[.!?…])\s+", text)
        tmp = ""
        for s in sents:
            if len(tmp) + len(s) + 1 > CHUNK_MAX_CHARS:
                if tmp.strip():
                    final_chunks.append(tmp.strip())
                tmp = s
            else:
                tmp = (tmp + " " + s).strip()
        if tmp.strip():
            final_chunks.append(tmp.strip())

    return [c for c in final_chunks if c and c.strip()]
