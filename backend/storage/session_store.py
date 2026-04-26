from typing import Dict, Any, Optional
import threading

_lock = threading.Lock()
_sessions: Dict[str, Dict[str, Any]] = {}


def get_session(session_id: str) -> Dict[str, Any]:
    with _lock:
        if session_id not in _sessions:
            _sessions[session_id] = {
                "model_weights": None,
                "training_config": None,
                "last_training_history": None,
                "dataset_split": None,
                "training_running": False,
                "stop_flag": False,
            }
        return _sessions[session_id]


def update_session(session_id: str, key: str, value: Any):
    with _lock:
        if session_id not in _sessions:
            get_session(session_id)
        _sessions[session_id][key] = value


def set_stop_flag(session_id: str):
    update_session(session_id, "stop_flag", True)


def clear_stop_flag(session_id: str):
    update_session(session_id, "stop_flag", False)


def is_stop_requested(session_id: str) -> bool:
    s = get_session(session_id)
    return s.get("stop_flag", False)
