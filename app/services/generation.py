from app.services.generation_fallbacks import empty_answer_message
from typing import List
from google import genai
from google.genai import types
from app.config import GEMINI_API_KEY, TEXT_MODEL_NAME, MAX_OUTPUT_TOKENS, RETRIEVAL_K

client = genai.Client(api_key=GEMINI_API_KEY)

def generate_answer_with_gemini(user_query: str, retrieved_chunks: List[str]) -> str:
    context = "\n\n".join(retrieved_chunks[:RETRIEVAL_K]) if retrieved_chunks else "(контекст не найден)"
    prompt = (
        "Вы — юридический помощник. Дай развёрнутый, практичный ответ.\n\n"
        "ИСПОЛЬЗУЙ ТОЛЬКО факты из Контекста ниже. Если чего-то нет в Контексте — прямо скажи, не выдумывай.\n\n"
        "Структура ответа:\n"
        "1) Краткий итог в 2–4 строках.\n"
        "2) Подробный разбор по пунктам, указывая ИМЯ документа, где содержатся положения.\n"
        "3) Цитаты из документа.\n"
        "4) Чёткие шаги.\n\n"
        f"КОНТЕКСТ:\n{context}\n\n"
        f"ЗАПРОС:\n{user_query}"
    )
    try:
        resp = client.models.generate_content(
            model=TEXT_MODEL_NAME,
            contents=[prompt],
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            ),
        )
        if getattr(resp, "text", None):
            return resp.text.strip()
        pf = getattr(resp, "prompt_feedback", None)
        if pf and getattr(pf, "block_reason", None):
            return f"⚠️ Запрос отклонён модерацией: {pf.block_reason}."
        cands = getattr(resp, "candidates", []) or []
        for c in cands:
            if getattr(c, "content", None) and getattr(c.content, "parts", None):
                parts_text = "".join(getattr(p, "text", "") for p in c.content.parts)
                if parts_text.strip():
                    return parts_text.strip()
        return "⚠️ Не удалось извлечь ответ из документов."
    except Exception as e:
        msg = repr(e)
        if any(x in msg for x in ["429", "503", "502", "504", "UNAVAILABLE", "ResourceExhausted"]):
            return "⚠️ Перегрузка модели. Повторите позже."
        if "401" in msg or "403" in msg or "PermissionDenied" in msg:
            return "⚠️ Неверный ключ или нет доступа."
        return f"⚠️ Ошибка: {msg}"

def generate_direct_ai_answer(user_query: str) -> str:
    system = (
        "Ты — внимательный и полезный ассистент. Отвечай чётко, по делу. "
        "Если вопрос юридический и у пользователя нет документов, давай общий совет и предупреждай о необходимости проверки юристом."
    )
    prompt = f"СИСТЕМА:\n{system}\n\nЗАПРОС ПОЛЬЗОВАТЕЛЯ:\n{user_query}"
    try:
        resp = client.models.generate_content(
            model=TEXT_MODEL_NAME,
            contents=[prompt],
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            ),
        )
        if getattr(resp, "text", None):
            return resp.text.strip()
        return "⚠️ Не удалось извлечь ответ из документов."
    except Exception as e:
        msg = repr(e)
        if any(x in msg for x in ["429", "503", "502", "504", "UNAVAILABLE", "ResourceExhausted"]):
            return "⚠️ Перегрузка модели. Повторите позже."
        if "401" in msg or "403" in msg or "PermissionDenied" in msg:
            return "⚠️ Неверный ключ или нет доступа."
        return f"⚠️ Ошибка: {msg}"
