import json
import os
import time

LOGS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
SESSION_LOG_FILE = os.path.join(LOGS_DIR, "session.jsonl")

def log_detailed_session(event_type: str, data: dict):
    """Writes detailed raw logs to a JSONL file."""
    record = {
        "type": event_type,
        "time": time.time(),
        "data": data
    }
    with open(SESSION_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
