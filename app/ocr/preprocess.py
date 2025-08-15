from PIL import Image, ImageFilter, ImageOps
import numpy as np

def to_grayscale(img: Image.Image) -> Image.Image:
    return img.convert("L")

def unsharp(img: Image.Image, radius: float = 1.2, percent: int = 150, threshold: int = 3) -> Image.Image:
    # Небольшая резкость помогает OCR
    return img.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))

def adaptive_binarize(img: Image.Image, block_size: int = 35, C: int = 10) -> Image.Image:
    # Простая псевдо-адаптивная бинаризация через PIL+numpy
    arr = np.array(img)
    # Выравнивание гистограммы улучшает контраст
    arr = ImageOps.equalize(Image.fromarray(arr))
    arr = np.array(arr)

    # Скользящее среднее: приближение адаптивного порога
    try:
        from scipy.ndimage import uniform_filter  # если scipy нет, будет исключение
        mean = uniform_filter(arr.astype(np.float32), size=block_size)
        bin_img = (arr > (mean - C)).astype(np.uint8) * 255
        return Image.fromarray(bin_img)
    except Exception:
        # Фоллбэк: простая пороговая обработка + автоконтраст
        img2 = ImageOps.autocontrast(Image.fromarray(arr))
        return img2.point(lambda x: 0 if x < 180 else 255)

def cleanup_borders(img: Image.Image, border: int = 2) -> Image.Image:
    return ImageOps.expand(ImageOps.crop(img), border=border, fill=255)

def preprocess(img: Image.Image) -> Image.Image:
    # Минимальный стабильный пайплайн без тяжелых зависимостей
    g = to_grayscale(img)
    g = unsharp(g)
    g = adaptive_binarize(g)
    g = cleanup_borders(g)
    return g
