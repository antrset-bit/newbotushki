
import re
import unicodedata

CYR = r"\u0400-\u04FF"

# Базовая карта латиница/цифры -> кириллица (безопасные замены)
BASE_MAP = str.maketrans({
    # буквы, которые почти всегда совпадают по форме
    "A": "А", "a": "а",
    "B": "В",
    "E": "Е", "e": "е",
    "K": "К", "k": "к",
    "M": "М", "m": "м",
    "H": "Н",
    "O": "О", "o": "о",
    "P": "Р", "p": "р",
    "C": "С", "c": "с",
    "T": "Т", "t": "т",
    "Y": "У", "y": "у",
    "X": "Х", "x": "х",
    "V": "И", "v": "и",   # часто путают
    "W": "Ш", "w": "ш",
    "F": "Ф", "f": "ф",
    # цифры, часто встречающиеся как буквы
    "0": "О",
    "3": "З",
})

# Контекстные неоднозначные замены — только между кириллическими символами
CONTEXT_RULES = [
    (r'(?P<l>[{cyr}])N(?P<r>[{cyr}])'.format(cyr=CYR), r'\g<l>И\g<r>'),
    (r'(?P<l>[{cyr}])n(?P<r>[{cyr}])'.format(cyr=CYR), r'\g<l>п\g<r>'),
    (r'(?P<l>[{cyr}])r(?P<r>[{cyr}])'.format(cyr=CYR), r'\g<l>г\g<r>'),
    (r'(?P<l>[{cyr}])u(?P<r>[{cyr}])'.format(cyr=CYR), r'\g<l>и\g<r>'),
    (r'(?P<l>[{cyr}])l(?P<r>[{cyr}])'.format(cyr=CYR), r'\g<l>л\g<r>'),
    (r'(?P<l>[{cyr}])j(?P<r>[{cyr}])'.format(cyr=CYR), r'\g<l>й\g<r>'),
]

# Регулярные правила очистки
REGEX_RULES = [
    # унификация дефисов/тире
    (re.compile(r"[‐‑‒–—―]+"), "-"),
    # убрать мусорные знаки, часто в OCR
    (re.compile(r"[«»“”„`´•·…†‡™©®]"), ""),
    # склейка разделённых пробелами слов типа "A В Т О Н О М Н А Я" → "АВТОНОМНАЯ"
    (re.compile(r"(?:(?<=^)|(?<=\n))([A-Za-z\u0400-\u04FF])(?:\s+[A-Za-z\u0400-\u04FF]){2,}(?=$|\n)"),
     lambda m: m.group(0).replace(" ", "")),
    # последовательности равно и дефисов → один пробел
    (re.compile(r"[=]{2,}"), " "),
    (re.compile(r"-{3,}"), "-"),
    # избыточные пробелы перед знаками препинания
    (re.compile(r"\s+([,.;:!?])"), r"\1"),
    # пробел после открывающей и перед закрывающей скобкой/кавычкой
    (re.compile(r"([\(\[\{])\s+"), r"\1"),
    (re.compile(r"\s+([\)\]\}])"), r"\1"),
    # множественные пробелы и пустые строки
    (re.compile(r"[ \t]{2,}"), " "),
    (re.compile(r"\n{3,}"), "\n\n"),
]

def _normalize_unicode(text: str) -> str:
    # Приводим к NFKC и нормализуем переносы строк
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text

def _apply_base_map(text: str) -> str:
    return text.translate(BASE_MAP)

def _apply_context_rules(text: str) -> str:
    for pattern, repl in [(re.compile(p), r) for p, r in CONTEXT_RULES]:
        text = pattern.sub(repl, text)
    return text

def _apply_regex_rules(text: str) -> str:
    for pat, repl in REGEX_RULES:
        text = pat.sub(repl, text)
    return text

def _tidy_spaces(text: str) -> str:
    # нормализуем пробелы вокруг дефисов
    text = re.sub(r"\s*-\s*", " - ", text)
    # сводим множественные пробелы к одному
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()



EXTENDED_MAP = dict(BASE_MAP)
EXTENDED_MAP.update({
    "N": "И", "n": "п",
    "r": "г",
    "u": "и",
    "l": "л",
    "j": "й",
    "Z": "З", "z": "з",
})

def _map_token_if_mixed(token: str) -> str:
    """Если слово содержит кириллицу и латиницу — применяем расширенную карту к латинским символам."""
    has_cyr = any('А' <= ch <= 'я' or '\u0400' <= ch <= '\u04FF' for ch in token)
    has_lat = any('A' <= ch <= 'Z' or 'a' <= ch <= 'z' for ch in token)
    if has_cyr and has_lat:
        return token.translate(str.maketrans(EXTENDED_MAP))
    return token

def _fix_mixed_words(text: str) -> str:
    return re.sub(r"[A-Za-z\u0400-\u04FF]+", lambda m: _map_token_if_mixed(m.group(0)), text)


def postprocess(text: str) -> str:
    """
    Постобработка OCR/извлеченного текста:
    - Unicode NFKC
    - безопасная транслитерация латиницы → кириллица
    - контекстные замены (N→И, r→г и т.д. между кириллическими буквами)
    - чистка мусорных символов и нормализация дефисов/пробелов
    """
    if not text:
        return ""
    text = _normalize_unicode(text)
    text = _apply_base_map(text)
    text = _apply_context_rules(text)
    text = _apply_regex_rules(text)
    text = _tidy_spaces(text)
    return text
