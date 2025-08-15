import time, random, logging
import numpy as np
from google import genai
from google.genai import types
from app.config import GEMINI_API_KEY, EMBEDDING_MODEL

logger = logging.getLogger("semantic-bot")
client = genai.Client(api_key=GEMINI_API_KEY)

def get_embedding(text: str) -> np.ndarray:
    max_attempts = 5
    base = 1.0
    for attempt in range(1, max_attempts + 1):
        try:
            resp = client.models.embed_content(model=EMBEDDING_MODEL, contents=text)
            vec = resp.embeddings[0].values
            arr = np.array(vec, dtype="float32")
            norm = np.linalg.norm(arr) or 1.0
            return (arr / norm).astype('float32')
        except Exception as e:
            msg = repr(e)
            transient = any(code in msg for code in ["503", "UNAVAILABLE", "429", "502", "504"])
            if attempt < max_attempts and transient:
                sleep_s = base * (2 ** (attempt - 1)) + random.uniform(0, 0.15)
                logger.warning("Эмбеддинг временно недоступен (%s). Повтор #%d через %.1f c", msg, attempt, sleep_s)
                time.sleep(sleep_s)
                continue
            logger.exception("Ошибка эмбеддинга (после %d попыток): %s", attempt, msg)
            raise
