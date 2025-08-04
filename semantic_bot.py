import os
import numpy as np
import faiss
import google.generativeai as genai
from telegram import Update, Document
from telegram.ext import Application, MessageHandler, filters, CommandHandler, ContextTypes
from pdf2image import convert_from_path
from dotenv import load_dotenv
load_dotenv()
from PIL import Image
from docx import Document as DocxDocument
import pytesseract
import fitz  # PyMuPDF

EMBEDDINGS = []
TEXTS = []
FAISS_INDEX = "index.faiss"
DOC_FOLDER = "documents"

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/embedding-001")

def get_embedding(text):
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type="RETRIEVAL_DOCUMENT"
    )
    return np.array(result["embedding"])

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
            chunks.append(current)
            current = p
        else:
            current += "\n" + p
    if current:
        chunks.append(current)
    return chunks

def index_documents():
    global EMBEDDINGS, TEXTS
    EMBEDDINGS, TEXTS = [], []
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

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Отправь текст запроса или загрузите PDF, DOCX, TXT файл.")

async def handle_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not os.path.exists(FAISS_INDEX):
        await update.message.reply_text("База ещё не проиндексирована.")
        return
    query = update.message.text
    if not query:
        return
    q_emb = get_embedding(query).astype("float32")
    index = faiss.read_index(FAISS_INDEX)
    D, I = index.search(np.array([q_emb]), k=3)
    results = [TEXTS[i] for i in I[0]]
    reply = "\n\n---\n\n".join(results)
    await update.message.reply_text(reply[:4096] or "Ничего не найдено.")

async def handle_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    file = update.message.document
    fname = file.file_name
    ext = fname.lower().split(".")[-1]
    if ext not in ("pdf", "docx", "txt"):
        await update.message.reply_text("Поддерживаются только .pdf, .docx, .txt")
        return
    file_path = os.path.join(DOC_FOLDER, fname)
    new_file = await context.bot.get_file(file.file_id)
    await new_file.download_to_drive(file_path)
    await update.message.reply_text("Файл загружен. Индексация...")
    index_documents()
    await update.message.reply_text("Файл проиндексирован. Теперь вы можете отправлять запросы.")

def main():
    TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_file))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_query))
    print("Бот запущен.")
    app.run_polling()

if __name__ == "__main__":
    main()
