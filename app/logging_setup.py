import logging

def setup_logging():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("semantic-bot")
    for noisy in ("httpx", "google_genai", "google_genai.models"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logger
