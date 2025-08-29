# -*- coding: utf-8 -*-
from typing import List, Dict

DEFAULT_HINTS = {
    "вознаграждение": ["Вознаграждение", "Оплата", "Стоимость услуг", "Финансовые условия", "Порядок оплаты"],
}

def empty_answer_message(query: str, metas: List[Dict]) -> str:
    available_sections = sorted(set([m.get("section_canonical") or m.get("section_title_raw") for m in metas if m]))
    hints = []
    if "вознаграждение" in available_sections:
        hints = DEFAULT_HINTS["вознаграждение"]
    hint_text = ""
    if hints:
        hint_text = "\nПроверьте наличие разделов: " + ", ".join(f"«{h}»" for h in hints) + "."
    avail_text = ""
    if available_sections:
        avail_text = "\nНайденные в документе разделы: " + ", ".join([str(x) for x in available_sections if x]) + "."
    return (
        "Не удалось извлечь ответ из документов по текущему запросу."
        + hint_text
        + avail_text
        + "\nПопробуйте переформулировать вопрос, например: «Каков размер вознаграждения и сроки оплаты?»"
    )
