from typing import Dict, List, Any, Optional
from threading import Lock


class ProgressStore:
    """
    In-memory per-session user progress tracker.
    Stores level, XP, unlocked features, and completed challenges.
    """

    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()

    def _default_progress(self) -> Dict[str, Any]:
        return {
            "current_level": 1,
            "xp": 0,
            "unlocked_features": [
                # Features available from the start
                "learning_rate",
                "epochs",
                "batch_size",
                "linear_dataset",
            ],
            "completed_challenges": [],  # list of level_ids
        }

    def _ensure_session(self, session_id: str) -> None:
        if session_id not in self._store:
            self._store[session_id] = self._default_progress()

    def get_progress(self, session_id: str) -> Dict[str, Any]:
        """Return full progress state for a session."""
        with self._lock:
            self._ensure_session(session_id)
            return dict(self._store[session_id])

    def update_progress(self, session_id: str, updates: Dict[str, Any]) -> None:
        """Merge updates into progress state."""
        with self._lock:
            self._ensure_session(session_id)
            self._store[session_id].update(updates)

    def award_xp(self, session_id: str, xp: int) -> int:
        """Add XP to session. Returns new total."""
        with self._lock:
            self._ensure_session(session_id)
            self._store[session_id]["xp"] += xp
            return self._store[session_id]["xp"]

    def unlock_feature(self, session_id: str, feature: str) -> None:
        """Add a feature to unlocked list (idempotent)."""
        with self._lock:
            self._ensure_session(session_id)
            features = self._store[session_id]["unlocked_features"]
            if feature not in features:
                features.append(feature)

    def unlock_features(self, session_id: str, features: List[str]) -> None:
        """Unlock multiple features at once."""
        for feature in features:
            self.unlock_feature(session_id, feature)

    def is_feature_unlocked(self, session_id: str, feature: str) -> bool:
        """Check if a feature is unlocked for a session."""
        with self._lock:
            self._ensure_session(session_id)
            return feature in self._store[session_id]["unlocked_features"]

    def complete_challenge(self, session_id: str, level_id: int) -> None:
        """Mark a challenge as completed (idempotent)."""
        with self._lock:
            self._ensure_session(session_id)
            completed = self._store[session_id]["completed_challenges"]
            if level_id not in completed:
                completed.append(level_id)

    def advance_level(self, session_id: str, new_level: int) -> None:
        """Set current level if new_level is higher than current."""
        with self._lock:
            self._ensure_session(session_id)
            current = self._store[session_id]["current_level"]
            if new_level > current:
                self._store[session_id]["current_level"] = new_level

    def reset_progress(self, session_id: str) -> None:
        """Reset progress to default (dev/testing use)."""
        with self._lock:
            self._store[session_id] = self._default_progress()

    def delete_session(self, session_id: str) -> None:
        with self._lock:
            self._store.pop(session_id, None)


# Global singleton
progress_store = ProgressStore()
