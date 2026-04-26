from typing import Dict, List, Any
import threading

_lock = threading.Lock()
_replays: Dict[str, Dict[str, List[Any]]] = {}


def save_snapshots(session_id: str, run_id: str, snapshots: List[Any]):
    with _lock:
        if session_id not in _replays:
            _replays[session_id] = {}
        _replays[session_id][run_id] = [s.dict() if hasattr(s, 'dict') else s for s in snapshots]


def get_replay(session_id: str) -> Dict[str, List[Any]]:
    with _lock:
        return _replays.get(session_id, {})


def get_replay_run(session_id: str, run_id: str) -> List[Any]:
    with _lock:
        return _replays.get(session_id, {}).get(run_id, [])


def clear_replay(session_id: str):
    with _lock:
        _replays.pop(session_id, None)
