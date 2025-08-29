from __future__ import annotations
from pathlib import Path
from typing import Iterator
import fitz  # PyMuPDF
from PIL import Image

def pdf_to_images(path: Path, dpi: int = 300) -> Iterator[Image.Image]:
    zoom = dpi / 72.0
    mtx = fitz.Matrix(zoom, zoom)
    with fitz.open(path) as doc:
        for page in doc:
            pix = page.get_pixmap(matrix=mtx, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            yield img
