
import json
from pathlib import Path
from datetime import datetime

CHAT_HISTORY_DIR = Path("./data/chat_history")
CHAT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def _get_chat_file(session_id: str) -> Path:
    return CHAT_HISTORY_DIR / f"{session_id}.json"


def save_user_message(session_id: str, message: str):
    _append_message(session_id, "user", message)


def save_agent_response(session_id: str, message: str):
    _append_message(session_id, "agent", message)


def _append_message(session_id: str, role: str, content: str):
    entry = {
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    }
    file = _get_chat_file(session_id)
    history = []
    if file.exists():
        with open(file, "r", encoding="utf-8") as f:
            history = json.load(f)

    history.append(entry)
    with open(file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def get_chat_history(session_id: str):
    file = _get_chat_file(session_id)
    if not file.exists():
        return []
    with open(file, "r", encoding="utf-8") as f:
        return json.load(f)


def list_all_sessions():
    return [f.stem for f in CHAT_HISTORY_DIR.glob("*.json") if f.is_file()]
