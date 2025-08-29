# app/services/llm_clean.py
import time
from typing import List, Iterable

from google import genai
from google.genai import types

from app.config import GEMINI_API_KEY, TEXT_MODEL_NAME, MAX_OUTPUT_TOKENS

_CLIENT = genai.Client(api_key=GEMINI_API_KEY)

_SYS_PROMPT = (
    "Ты — корректный редактор русского текста после OCR. "
    "Приведи текст к нормальному виду: убери латиницу и заменяй латинские «похожие» буквы на кириллицу, "
    "исправь типичные OCR-искажения (склейки/разрывы слов, регистр, странные символы), "
    "восстанови юридические термины по контексту. "
    "НЕ добавляй новые факты и НЕ меняй смысл; сохраняй структуру абзацев. "
    "Ответь только очищенным текстом без комментариев."
)

def _chunks_by_chars(text: str, max_chars: int) -> Iterable[str]:
    paras = text.replace("\r\n", "\n").split("\n\n")
    buf = []
    cur = 0
    for p in paras:
        p = p.strip()
        if not p:
            if buf:
                yield "\n\n".join(buf)
                buf, cur = [], 0
            continue
        need = len(p) + (2 if cur > 0 else 0)
        if cur + need > max_chars and buf:
            yield "\n\n".join(buf)
            buf, cur = [], 0
        buf.append(p); cur += need
    if buf:
        yield "\n\n".join(buf)

def clean_with_gemini(text: str, max_chars: int = 6000, temperature: float = 0.1) -> str:
    if not text or not text.strip():
        return text
    out_parts: List[str] = []
    for part in _chunks_by_chars(text, max_chars=max_chars):
        contents = [
            _SYS_PROMPT,
            "\n=== ТЕКСТ ДЛЯ ОЧИСТКИ ===\n" + part
        ]
        base = 1.0
        for attempt in range(1, 5):
            try:
                resp = _CLIENT.models.generate_content(
                    model=TEXT_MODEL_NAME,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=min(MAX_OUTPUT_TOKENS, 4096),
                    ),
                )
                if getattr(resp, "text", None):
                    out_parts.append(resp.text.strip())
                    break
                chunk_text = ""
                for c in getattr(resp, "candidates", []) or []:
                    if getattr(c, "content", None) and getattr(c.content, "parts", None):
                        for p in c.content.parts:
                            chunk_text += getattr(p, "text", "")
                if chunk_text.strip():
                    out_parts.append(chunk_text.strip())
                    break
                out_parts.append(part)
                break
            except Exception as e:
                msg = repr(e)
                if any(code in msg for code in ("429","502","503","504","UNAVAILABLE","ResourceExhausted")) and attempt < 4:
                    time.sleep(base * (2 ** (attempt - 1)))
                    continue
                out_parts.append(part)
                break
    return "\n\n".join(out_parts).strip()
