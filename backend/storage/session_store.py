import uuid
from typing import Dict, Any, Optional
from threading import Lock


class SessionStore:
    """
    In-memory session store keyed by session_id.
    Stores current model weights, training config, last training history,
    and current dataset split per session.
    Thread-safe via Lock.
    """

    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()

    def create_session(self, session_id: Optional[str] = None) -> str:
        """Create a new session. Returns session_id."""
        if session_id is None:
            session_id = str(uuid.uuid4())
        with self._lock:
            self._store[session_id] = {
                "weights": None,
                "training_config": None,
                "last_history": None,
                "dataset_split": None,
                "is_training": False,
                "current_epoch": 0,
                "total_epochs": 0,
                "current_loss": None,
                "current_acc": None,
            }
        return session_id

    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve session data."""
        with self._lock:
            return self._store.get(session_id)

    def exists(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._store

    def ensure_exists(self, session_id: str) -> None:
        """Create session if it doesn't exist."""
        if not self.exists(session_id):
            self.create_session(session_id)

    def set_weights(self, session_id: str, weights: Dict[str, Any]) -> None:
        self.ensure_exists(session_id)
        with self._lock:
            self._store[session_id]["weights"] = weights

    def get_weights(self, session_id: str) -> Optional[Dict[str, Any]]:
        session = self.get(session_id)
        return session["weights"] if session else None

    def set_training_config(self, session_id: str, config: Dict[str, Any]) -> None:
        self.ensure_exists(session_id)
        with self._lock:
            self._store[session_id]["training_config"] = config

    def get_training_config(self, session_id: str) -> Optional[Dict[str, Any]]:
        session = self.get(session_id)
        return session["training_config"] if session else None

    def set_last_history(self, session_id: str, history: Any) -> None:
        self.ensure_exists(session_id)
        with self._lock:
            self._store[session_id]["last_history"] = history

    def get_last_history(self, session_id: str) -> Optional[Any]:
        session = self.get(session_id)
        return session["last_history"] if session else None

    def set_dataset_split(self, session_id: str, split: Dict[str, Any]) -> None:
        self.ensure_exists(session_id)
        with self._lock:
            self._store[session_id]["dataset_split"] = split

    def get_dataset_split(self, session_id: str) -> Optional[Dict[str, Any]]:
        session = self.get(session_id)
        return session["dataset_split"] if session else None

    def set_training_status(
        self,
        session_id: str,
        is_training: bool,
        current_epoch: int = 0,
        total_epochs: int = 0,
        current_loss: Optional[float] = None,
        current_acc: Optional[float] = None,
    ) -> None:
        self.ensure_exists(session_id)
        with self._lock:
            s = self._store[session_id]
            s["is_training"] = is_training
            s["current_epoch"] = current_epoch
            s["total_epochs"] = total_epochs
            s["current_loss"] = current_loss
            s["current_acc"] = current_acc

    def get_training_status(self, session_id: str) -> Dict[str, Any]:
        session = self.get(session_id)
        if not session:
            return {
                "is_training": False,
                "current_epoch": 0,
                "total_epochs": 0,
                "current_loss": None,
                "current_acc": None,
            }
        return {
            "is_training": session["is_training"],
            "current_epoch": session["current_epoch"],
            "total_epochs": session["total_epochs"],
            "current_loss": session["current_loss"],
            "current_acc": session["current_acc"],
        }

    def delete_session(self, session_id: str) -> None:
        with self._lock:
            self._store.pop(session_id, None)

    def list_sessions(self):
        with self._lock:
            return list(self._store.keys())


# Global singleton
session_store = SessionStore()
