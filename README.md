# Semantic Telegram Bot (FastAPI + PTB + Gemini + FAISS)

Готовый пакет для деплоя на Render.

## Быстрый старт локально

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # и заполните переменные
export $(grep -v '^#' .env | xargs)  # для Linux/macOS
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Деплой на Render

1. Залейте репозиторий на GitHub.
2. В Render создайте **Web Service** из этого репо. Render автоматически прочитает `render.yaml` и `Dockerfile`.
3. В Variables добавьте секреты: `TELEGRAM_BOT_TOKEN`, `GEMINI_API_KEY`, `PUBLIC_BASE_URL`, `ADMIN_USER_IDS`, при необходимости TM_*.
4. Убедитесь, что `RUN_MODE=webhook` и `PUBLIC_BASE_URL` указывает на публичный URL сервиса.
5. После старта проверьте `/` (должно вернуть `{"ok": true}`).

### Включение Telegram webhook

Сервис сам проставит webhook на `PUBLIC_BASE_URL/telegram/<TELEGRAM_BOT_TOKEN>` при старте.

### Папки данных

Файлы, индекс, метаданные — в `/opt/render/project/src/data/*`. Эти пути настраиваются через переменные окружения.



## Новый режим: 🗂️ Работа с документами

В этом режиме можно загрузить **DOC/DOCX/PDF** и работать с ними в контексте текущего диалога (без записи в общий индекс).

Доступные команды:
- `/doc_summary` — краткое резюме последнего загруженного документа.
- `/doc_check` — проверка договора на ошибки/риски, рекомендации по правкам.
- `/doc_compare` — сравнение двух последних файлов (покажу различия по тексту).
- Любой текстовый вопрос в этом режиме — RAG-ответ по загруженным документам.

Если ответ получается длинным — бот разобьёт его на несколько сообщений автоматически.
