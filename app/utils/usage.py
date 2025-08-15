import os, json, time
from app.config import USAGE_FILE, ADMIN_USER_IDS

def _today_str() -> str:
    return time.strftime("%Y-%m-%d", time.localtime())

def _load_usage() -> dict:
    if os.path.exists(USAGE_FILE):
        try:
            with open(USAGE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def _save_usage(d: dict):
    try:
        with open(USAGE_FILE, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def is_admin(user_id: int) -> bool:
    return user_id in ADMIN_USER_IDS

def get_usage(user_id: int) -> int:
    data = _load_usage()
    today = _today_str()
    return int(data.get(str(user_id), {}).get(today, 0))

def inc_usage(user_id: int) -> int:
    data = _load_usage()
    today = _today_str()
    u = data.setdefault(str(user_id), {})
    u[today] = int(u.get(today, 0)) + 1
    _save_usage(data)
    return u[today]
