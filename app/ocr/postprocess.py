import re

# Частые латинизмы -> кириллица
LAT_TO_CYR = str.maketrans({
    "A": "А", "a": "а",
    "B": "В", "E": "Е", "e": "е",
    "K": "К", "k": "к",
    "M": "М", "m": "м",
    "H": "Н", "O": "О", "o": "о",
    "P": "Р", "p": "р",
    "C": "С", "c": "с",
    "T": "Т", "t": "т",
    "Y": "У", "y": "у",
    "X": "Х", "x": "х",
})

# Типичные OCR-ошибки для деловых текстов
REGEX_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b110\b"), "по"),
    (re.compile(r"OKa3aHH[ИЙI]" , re.IGNORECASE), "оказании"),
    (re.compile(r"\b3aka3(\d|4)u?ka\b", re.IGNORECASE), "заказчика"),
    (re.compile(r"HeqocTaTK[ui]", re.IGNORECASE), "недостатки"),
    (re.compile(r"\bMpu\b"), "при"),
]

def normalize_hyphens(text: str) -> str:
    # Склейка переносов внутри слов: «про- ект» -> «проект»
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Удаляем лишние двойные пробелы и странные невидимые символы
    text = re.sub(r"[\u00A0\u200B\u2060]", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text

def translit_fix(text: str) -> str:
    return text.translate(LAT_TO_CYR)

def apply_regex_rules(text: str) -> str:
    for pat, repl in REGEX_RULES:
        text = pat.sub(repl, text)
    return text

def postprocess(text: str) -> str:
    text = normalize_hyphens(text)
    text = translit_fix(text)
    text = apply_regex_rules(text)
    return text
