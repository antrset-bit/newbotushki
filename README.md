# Semantic Search Telegram Bot (Gemini + RAG)

Этот бот позволяет пользователю:

- Загружать документы (PDF, DOCX, TXT);
- Автоматически применять OCR (для PDF-сканов);
- Индексировать документы с помощью эмбеддингов (Gemini API);
- Искать по базе и возвращать релевантные фрагменты по пользовательскому запросу.

## 📦 Установка

```bash
pip install -r requirements.txt
```

## 🛠 Переменные окружения

Создайте `.env` или задайте через систему:

```env
TELEGRAM_BOT_TOKEN=ваш_токен
GEMINI_API_KEY=ваш_API_ключ
EMBEDDING_MODEL=models/embedding-001  # по умолчанию
```

## 🚀 Запуск

```bash
python semantic_bot.py
```

## 📁 Структура

- `semantic_bot.py` — основной скрипт
- `documents/` — сюда загружаются файлы для индексации
- `index.faiss` — сохраняемый индекс FAISS (создается автоматически)
