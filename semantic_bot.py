import os
import numpy as np
import faiss
import asyncio
import google.generativeai as genai
from telegram import Update, Document
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
from pdf2image import convert_from_path
from PIL import Image
from docx import Document as DocxDocument
import pytesseract
import fitz  # PyMuPDF

# --- Конфигурация переменных ---
DOC_FOLDER = "documents"
FAISS_INDEX = "index.faiss"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/embedding-001")

# --- Настройка Gemini ---
genai.configure(api_key=GEMINI_API_KEY)

def get_supported_text_model():
    try:
        models = genai.list_models()
        for m in models:
            if "generateContent" in m.supported_generation_methods:
                return m.name
    except Exception as e:
        print("❌ Ошибка при получении списка моделей:", e)
    return "models/gemini-pro"  # fallback

TEXT_MODEL_NAME = get_supported_text_model()
TEXT_MODEL = genai.GenerativeModel(TEXT_MODEL_NAME)

# --- Глобальные переменные памяти ---
TEXTS = []
EMBEDDINGS = []

# --- Извлечение текста ---
def extract_text_from_pdf(file_path):
    try:
        doc = fitz.open(file_path)
        return "\n".join([page.get_text() for page in doc])
    except Exception:
        images = convert_from_path(file_path)
        return "\n".join([pytesseract.image_to_string(img) for img in images])

def extract_text_from_docx(file_path):
    doc = DocxDocument(file_path)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_text_from_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def split_text(text, max_tokens=300):
    paragraphs = text.split("\n")
    chunks, current = [], ""
    for p in paragraphs:
        if len(current + p) > max_tokens * 4:
            chunks.append(current.strip())
            current = p
        else:
            current += "\n" + p
    if current:
        chunks.append(current.strip())
    return chunks

# --- Векторизация и индексация ---
def get_embedding(text):
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type="RETRIEVAL_DOCUMENT"
    )
    return np.array(result["embedding"])

def index_documents():
    global TEXTS, EMBEDDINGS
    TEXTS, EMBEDDINGS = [], []
    for fname in os.listdir(DOC_FOLDER):
        path = os.path.join(DOC_FOLDER, fname)
        if fname.endswith(".pdf"):
            text = extract_text_from_pdf(path)
        elif fname.endswith(".docx"):
            text = extract_text_from_docx(path)
        elif fname.endswith(".txt"):
            text = extract_text_from_txt(path)
        else:
            continue
        for chunk in split_text(text):
            emb = get_embedding(chunk)
            EMBEDDINGS.append(emb)
            TEXTS.append(chunk)
    if EMBEDDINGS:
        dim = len(EMBEDDINGS[0])
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(EMBEDDINGS).astype("float32"))
        faiss.write_index(index, FAISS_INDEX)

# --- Генерация ответа через Gemini ---
def generate_answer_with_gemini(user_query: str, retrieved_chunks: list[str]) -> str:
    context = "\n\n".join(retrieved_chunks[:2])  # Ограничим до 2 блоков
    prompt = f"""
Вы юридический помощник. Используйте приведённый ниже контекст из документов, чтобы ответить на запрос пользователя.

Контекст:
\"\"\"
{context}
\"\"\"

Запрос пользователя:
{user_query}

Ответ (четко, кратко, по делу):
"""
    try:
        response = TEXT_MODEL.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print("⚠️ Ошибка генерации Gemini:", e)
        return "⚠️ Квота превышена или временная ошибка. Попробуйте позже."

# --- Telegram handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Загрузите .pdf, .docx или .txt файл, затем задайте вопрос.")

async def handle_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    document: Document = update.message.document
    fname = document.file_name
    ext = fname.lower().split(".")[-1]
    if ext not in ("pdf", "docx", "txt"):
        await update.message.reply_text("Поддерживаются только PDF, DOCX, TXT.")
        return
    os.makedirs(DOC_FOLDER, exist_ok=True)
    file_path = os.path.join(DOC_FOLDER, fname)
    new_file = await context.bot.get_file(document.file_id)
    await new_file.download_to_drive(file_path)
    await update.message.reply_text("Файл загружен. Индексация началась...")
    index_documents()
    await update.message.reply_text("Индексация завершена. Теперь можно задавать вопросы.")

async def handle_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not os.path.exists(FAISS_INDEX):
        await update.message.reply_text("Нет проиндексированных документов. Пожалуйста, загрузите файл.")
        return
    user_query = update.message.text
    q_emb = get_embedding(user_query).astype("float32")
    index = faiss.read_index(FAISS_INDEX)
    D, I = index.search(np.array([q_emb]), k=3)
    retrieved = [TEXTS[i] for i in I[0]]
    answer = generate_answer_with_gemini(user_query, retrieved)
    await update.message.reply_text(answer[:4096])

# --- Основной запуск ---
def main():
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_file))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_query))
    print("Бот запущен.")
    app.run_polling()

if __name__ == "__main__":
    main()
