
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
    "Z": "З", "z": "з", "J": "Д", "j": "д", "h": "н", "q": "д",
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




# --- Дополнительные агрессивные правки под юр-документы ---

# Частые юридические фразы и заголовки (разные «ломаные» варианты -> каноничная форма)
CANON_RULES = [
    # Договор
    (re.compile(r"(?i)\bJ[ji]оrо[vvb]ор\b"), "Договор"),
    (re.compile(r"(?i)\bJ[ji]оrо[vvb]ор[а-яA-Za-z]*"), "Договор"),
    # Исполнитель / Заказчик
    (re.compile(r"(?i)\bUсnоmm?m?r?e?nb\b"), "Исполнитель"),
    (re.compile(r"(?i)\b3аkа3[ui]n?k\b"), "Заказчик"),
    # Автономная некоммерческая организация
    (re.compile(r"(?i)АВТОНОМН[Аа][Ss]?\s+Н[еe]КОММ[еe]р[чс][еe]с[кk][аa][яy]\s+ОРГАНИЗАЦИ[Яя]"), "АВТОНОМНАЯ НЕКОММЕРЧЕСКАЯ ОРГАНИЗАЦИЯ"),
    # Приложение № 1 (и подобные)
    (re.compile(r"(?i)Прилож[еe]н[иi][еe]\s+N[еe°º]"), "Приложение № "),
    # Сторона / Стороны
    (re.compile(r"(?i)С[тt]ор[оo]н[аa]"), "Сторона"),
    (re.compile(r"(?i)С[тt]ор[оo]н[ыy]"), "Стороны"),
]

def _apply_canon_rules(text: str) -> str:
    for pat, repl in CANON_RULES:
        text = pat.sub(repl, text)
    return text

def _legal_specific_normalize(text: str) -> str:
    # № из OCR: Nе, N° , Nº, Nо → №
    text = re.sub(r"\bN[еe°ºoO]\b", "№", text)
    # r. → г.  (год/город); 2025r. → 2025 г.
    text = re.sub(r"\br\.\s*", "г. ", text)
    text = re.sub(r"(\d{4})\s*r\.", r"\1 г.", text)
    # Окончания типа "ая" и "ий" ломаются как 'aS'/'iй' — починим распространённый случай
    text = re.sub(r"([А-Яа-я])[aA][sS]\b", r"\1ая", text)  # ...аS → ...ая
    # двойные дефисы/равно после нормализации
    text = re.sub(r"\s*=\s*", " ", text)
    return text




# --- Финальная стадия: только кириллица (по запросу пользователя) ---

# Используем расширенную карту если есть, иначе базовую
try:
    _FINAL_MAP = str.maketrans(EXTENDED_MAP)  # type: ignore
except NameError:
    _FINAL_MAP = BASE_MAP  # type: ignore

_ALLOWED_PUNCT = r",\.\:\;\!\?\-—–\(\)\[\]\{\}«»\"'\/\\\%\@\#\&\*\+\=\_<>\|\^~`"  # что оставляем кроме букв/цифр и пробелов

def _force_cyrillic_only(text: str) -> str:
    # 1) максимально транслитерируем латиницу в кириллицу
    text = text.translate(_FINAL_MAP)
    # 2) удаляем оставшиеся латинские буквы целиком
    text = re.sub(r"[A-Za-z]", "", text)
    # 3) приводим множественные пробелы
    text = re.sub(r"[ \t]{2,}", " ", text)
    # 4) убираем пробелы перед знаками препинания
    text = re.sub(r"\s+([{}])".format(_ALLOWED_PUNCT), r"\1", text)
    # 5) нормализуем переносы
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


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
    text = _legal_specific_normalize(text)
    text = _apply_canon_rules(text)
    text = _tidy_spaces(text)
    text = _force_cyrillic_only(text)
    return text
