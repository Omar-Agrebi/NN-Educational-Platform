from typing import Dict, Any, List
import threading
import json, os

_lock = threading.Lock()
_progress: Dict[str, Dict[str, Any]] = {}
PERSIST_PATH = "/tmp/neural_forge_progress.json"


def _default_progress() -> Dict[str, Any]:
    return {
        "current_level": 1,
        "xp": 0,
        "unlocked_features": ["learning_rate", "epochs", "slow_mode"],
        "completed_challenges": [],
        "insights_collected": [],
        "total_runs": 0,
        "best_scores": {},
    }


def _load_persisted():
    global _progress
    if os.path.exists(PERSIST_PATH):
        try:
            with open(PERSIST_PATH, "r") as f:
                _progress = json.load(f)
        except Exception:
            _progress = {}


def _persist():
    try:
        with open(PERSIST_PATH, "w") as f:
            json.dump(_progress, f)
    except Exception:
        pass


_load_persisted()


def get_progress(session_id: str) -> Dict[str, Any]:
    with _lock:
        if session_id not in _progress:
            _progress[session_id] = _default_progress()
        return dict(_progress[session_id])


def update_progress(session_id: str, updates: Dict[str, Any]):
    with _lock:
        if session_id not in _progress:
            _progress[session_id] = _default_progress()
        _progress[session_id].update(updates)
        _persist()


def unlock_feature(session_id: str, feature: str):
    with _lock:
        if session_id not in _progress:
            _progress[session_id] = _default_progress()
        feats = _progress[session_id]["unlocked_features"]
        if feature not in feats:
            feats.append(feature)
        _persist()


def unlock_features(session_id: str, features: List[str]):
    for f in features:
        unlock_feature(session_id, f)


def award_xp(session_id: str, amount: int):
    with _lock:
        if session_id not in _progress:
            _progress[session_id] = _default_progress()
        _progress[session_id]["xp"] = _progress[session_id].get("xp", 0) + amount
        _persist()


def is_feature_unlocked(session_id: str, feature: str) -> bool:
    p = get_progress(session_id)
    return feature in p.get("unlocked_features", [])


def add_insight(session_id: str, insight: str):
    with _lock:
        if session_id not in _progress:
            _progress[session_id] = _default_progress()
        insights = _progress[session_id].get("insights_collected", [])
        if insight not in insights:
            insights.append(insight)
            _progress[session_id]["insights_collected"] = insights
        _persist()


def reset_progress(session_id: str):
    with _lock:
        _progress[session_id] = _default_progress()
        _persist()
