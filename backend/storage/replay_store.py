from typing import Dict, List, Any, Optional
from threading import Lock


class ReplayStore:
    """
    Stores per-session, per-run epoch snapshots for replay playback.
    Structure: { session_id: { run_id: [snapshot, ...] } }

    Each snapshot:
        {
            epoch: int,
            weights: dict,
            train_loss: float,
            test_loss: float,
            train_acc: float,
            test_acc: float,
            boundary_data: dict | None,
        }
    """

    def __init__(self):
        self._store: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        self._lock = Lock()

    def _ensure_session(self, session_id: str) -> None:
        if session_id not in self._store:
            self._store[session_id] = {}

    def _ensure_run(self, session_id: str, run_id: str) -> None:
        self._ensure_session(session_id)
        if run_id not in self._store[session_id]:
            self._store[session_id][run_id] = []

    def save_snapshot(
        self,
        session_id: str,
        run_id: str,
        snapshot: Dict[str, Any],
    ) -> None:
        """Append a single epoch snapshot to a run."""
        with self._lock:
            self._ensure_run(session_id, run_id)
            self._store[session_id][run_id].append(snapshot)

    def save_all_snapshots(
        self,
        session_id: str,
        run_id: str,
        snapshots: List[Dict[str, Any]],
    ) -> None:
        """Bulk-save all snapshots for a completed run."""
        with self._lock:
            self._ensure_session(session_id)
            self._store[session_id][run_id] = snapshots

    def get_replay(self, session_id: str) -> Dict[str, List[Dict[str, Any]]]:
        """Return all runs (and their snapshots) for a session."""
        with self._lock:
            return dict(self._store.get(session_id, {}))

    def get_run_replay(
        self,
        session_id: str,
        run_id: str,
    ) -> Optional[List[Dict[str, Any]]]:
        """Return snapshots for a specific run, or None if not found."""
        with self._lock:
            session = self._store.get(session_id, {})
            return session.get(run_id)

    def list_runs(self, session_id: str) -> List[str]:
        """List all run_ids for a session."""
        with self._lock:
            return list(self._store.get(session_id, {}).keys())

    def clear_replay(self, session_id: str) -> None:
        """Remove all replay data for a session."""
        with self._lock:
            self._store.pop(session_id, None)

    def clear_run(self, session_id: str, run_id: str) -> None:
        """Remove replay data for a specific run."""
        with self._lock:
            session = self._store.get(session_id, {})
            session.pop(run_id, None)

    def snapshot_count(self, session_id: str, run_id: str) -> int:
        """Number of snapshots stored for a run."""
        with self._lock:
            session = self._store.get(session_id, {})
            return len(session.get(run_id, []))


# Global singleton
replay_store = ReplayStore()
