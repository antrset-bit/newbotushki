import re, asyncio, io, logging, requests, csv
from telegram import InputFile
from typing import List, Callable

from app.config import TM_ENABLE, TM_SHEET_ID, TM_SHEET_GID, TM_SHEET_NAME, TM_SHEET_CSV_URL, TM_DEBUG
logger = logging.getLogger("semantic-bot")

def _html_escape(text: str) -> str:
    return (text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
                .replace("'", "&#039;"))

URL_RE = re.compile(r'(https?://[^\s<>")]+)', re.IGNORECASE)

def _extract_urls(cell: str) -> list[str]:
    txt = str(cell or "")
    tokens = re.split(r'[,\s]+', txt.strip())
    urls = []
    for t in tokens:
        if t.startswith("http://") or t.startswith("https://"):
            urls.append(t)
        else:
            urls.extend(URL_RE.findall(t))
    seen, out = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u); out.append(u)
    return out

def _normalize_image_url(url: str) -> str:
    u = url.strip()
    import re as _re
    m = _re.search(r'drive\.google\.com/file/d/([^/]+)/', u)
    if m:
        return f"https://drive.google.com/uc?export=download&id={m.group(1)}"
    m = _re.search(r'(?:[?&]id=)([A-Za-z0-9_-]+)', u)
    if m and "drive.google.com" in u:
        return f"https://drive.google.com/uc?export=download&id={m.group(1)}"
    return u

def _is_probable_image_url(url: str) -> bool:
    u = url.lower()
    if any(u.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".gif", ".webp")):
        return True
    if "googleusercontent.com" in u or "uc?export=download" in u or "=download" in u:
        return True
    return False

def _format_date(value: str) -> str:
    from datetime import datetime
    for fmt in ("%d.%m.%Y", "%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(value.strip(), fmt).strftime("%d/%m/%Y")
        except Exception:
            pass
    return value.strip()

TM_LABELS = ['№', 'Номер заявки', 'Номер регистрации', '', 'Описание', 'Статус', 'Срок действия', 'Комментарии', '', 'Ссылка']

def tm_format_row(row: list[str], labels: list[str] = TM_LABELS) -> tuple[str, list[str]]:
    formatted_lines = []
    image_urls: list[str] = []

    for idx, val in enumerate(row):
        label = labels[idx] if idx < len(labels) else ""
        if idx == 3:
            continue

        cell = str(val or "").strip()
        if not cell:
            continue

        urls = _extract_urls(cell)
        for u in urls:
            nu = _normalize_image_url(u)
            if _is_probable_image_url(nu):
                image_urls.append(nu)

        only_links = urls and (cell == " ".join(urls) or cell == ",".join(urls))
        if only_links:
            continue

        if re.match(r"^\d{1,4}[-./]\d{1,2}[-./]\d{1,4}$", cell):
            cell = _format_date(cell)

        if label:
            formatted_lines.append(f"<b>{_html_escape(label)}:</b> {_html_escape(cell)}")
        else:
            formatted_lines.append(_html_escape(cell))

    seen, uniq_images = set(), []
    for u in image_urls:
        if u not in seen:
            seen.add(u); uniq_images.append(u)

    text = "\n".join(formatted_lines).strip()
    return text, uniq_images

async def _tm_fetch_rows_csv(sheet_id: str, gid: str, sheet_name: str, override_url: str = "") -> list[list[str]]:
    urls = []
    if override_url:
        urls.append(override_url)
    if sheet_id:
        urls.append(f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}")
        urls.append(f"https://docs.google.com/spreadsheets/d/{sheet_id}/pub?gid={gid}&single=true&output=csv")
        if sheet_name:
            from urllib.parse import quote
            urls.append(f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={quote(sheet_name)}")
    last_err = None
    for url in urls:
        try:
            resp = await asyncio.to_thread(requests.get, url, timeout=30)
            status = resp.status_code
            ctype = resp.headers.get("Content-Type", "")
            content = resp.content.decode("utf-8", errors="replace")
            if status == 200 and ("," in content or ";" in content or "\n" in content):
                reader = csv.reader(io.StringIO(content))
                rows = [row for row in reader]
                if rows and any(any(cell.strip() for cell in row) for row in rows):
                    return rows
            last_err = f"Bad content from {url} (status={status}, type={ctype})"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e} at {url}"
            continue
    raise RuntimeError(last_err or "CSV fetch failed")

async def tm_load_data() -> list[list[str]]:
    if not TM_ENABLE:
        return []
    try:
        return await _tm_fetch_rows_csv(TM_SHEET_ID, TM_SHEET_GID, TM_SHEET_NAME, TM_SHEET_CSV_URL)
    except Exception as e:
        logger.error("TM: не удалось загрузить данные листа: %s", repr(e))
        if TM_DEBUG:
            raise
        return []

def _row_matches_registered(row: list[str]) -> bool:
    col = (row[5] if len(row) > 5 else "") or ""
    return "регистрация" in col.lower()

def _row_matches_expertise(row: list[str]) -> bool:
    col = (row[5] if len(row) > 5 else "") or ""
    return "экспертиза" in col.lower()

def _row_matches_keywords(row: list[str], keywords: list[str]) -> bool:
    low = [str(c or "").lower() for c in row]
    for kw in keywords:
        kw = kw.strip().lower()
        if kw and any(kw in cell for cell in low):
            return True
    return False

async def _tm_send_image_safely(chat_id: int, url: str, context) -> bool:
    try:
        await context.bot.send_photo(chat_id, photo=url)
        return True
    except Exception as e:
        logger.warning("TM: send_photo by URL failed for %s: %r", url, e)
    try:
        r = requests.get(url, timeout=25)
        ct = (r.headers.get("Content-Type") or "").lower()
        if r.status_code == 200 and (r.content and ("image/" in ct or len(r.content) > 0)):
            buf = io.BytesIO(r.content)
            filename = "image"
            ul = url.lower()
            if ".png" in ul: filename += ".png"
            elif ".jpg" in ul or ".jpeg" in ul: filename += ".jpg"
            elif ".webp" in ul: filename += ".webp"
            elif ".gif" in ul: filename += ".gif"
            else:
                if "image/png" in ct: filename += ".png"
                elif "image/jpeg" in ct: filename += ".jpg"
                elif "image/webp" in ct: filename += ".webp"
                elif "image/gif" in ct: filename += ".gif"
                else: filename += ".bin"
            await context.bot.send_photo(chat_id, photo=InputFile(buf, filename=filename))
            return True
    except Exception as e:
        logger.warning("TM: fallback download failed for %s: %r", url, e)
    return False

async def tm_process_search(chat_id: int, condition_cb, context):
    try:
        data = await tm_load_data()
    except Exception as e:
        msg = "Данные листа недоступны или пусты."
        if TM_DEBUG:
            msg += f"\n\nДиагностика: {e!r}\nПроверьте публикацию таблицы и доступность CSV."
        await context.bot.send_message(chat_id, msg)
        return

    if not data or not any(data):
        note = "Данные листа недоступны или пусты."
        if TM_DEBUG:
            note += f"\nПроверьте: Publish to web включён, правильный GID, лист не пустой."
        await context.bot.send_message(chat_id, note)
        return

    rows = data[1:] if len(data) > 1 else []
    matched_idx = [i for i, r in enumerate(rows, start=2) if condition_cb(r)]
    if not matched_idx:
        await context.bot.send_message(chat_id, "Данные не найдены.")
        return

    for i in matched_idx:
        row = data[i-1]
        text, images = tm_format_row(row)
        if text:
            await context.bot.send_message(chat_id, text, parse_mode="HTML", disable_web_page_preview=False)
        for url in images:
            await _tm_send_image_safely(chat_id, url, context)

# Expose helpers
ROW_MATCH_REGISTERED = _row_matches_registered
ROW_MATCH_EXPERTISE = _row_matches_expertise
ROW_MATCH_KW = _row_matches_keywords
