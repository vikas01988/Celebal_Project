import json
import os
import time
from config import LOG_FILE
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
def log_event(**kwargs):
    entry = {"timestamp": time.time(), **kwargs}
    with open(LOG_FILE, "a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(entry, default=str) + "\n")
