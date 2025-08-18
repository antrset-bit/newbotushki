import os, json, hashlib
from pathlib import Path
from typing import List
import pickle

from app.config import MANIFEST_FILE, TEXTS_FILE

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def load_manifest() -> dict:
    if os.path.exists(MANIFEST_FILE):
        with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_manifest(m: dict):
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)

def load_texts() -> List[str]:
    if os.path.exists(TEXTS_FILE):
        with open(TEXTS_FILE, "rb") as f:
            return pickle.load(f)
    return []

def save_texts(texts: List[str]):
    with open(TEXTS_FILE, "wb") as f:
        pickle.dump(texts, f)
