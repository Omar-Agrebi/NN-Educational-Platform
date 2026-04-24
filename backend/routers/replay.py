from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any

from storage.replay_store import replay_store

router = APIRouter(prefix="/replay", tags=["replay"])


@router.get("/{session_id}")
async def get_all_replays(session_id: str) -> Dict[str, Any]:
    """Return all saved snapshots for every run in a session."""
    data = replay_store.get_replay(session_id)
    if not data:
        raise HTTPException(
            status_code=404,
            detail=f"No replay data found for session '{session_id}'."
        )
    return {
        "session_id": session_id,
        "runs": {
            run_id: {
                "run_id": run_id,
                "snapshot_count": len(snapshots),
                "snapshots": [
                    {
                        "epoch": s["epoch"],
                        "train_loss": s["train_loss"],
                        "test_loss": s["test_loss"],
                        "train_acc": s["train_acc"],
                        "test_acc": s["test_acc"],
                        # Weights omitted from list view (use per-run endpoint for full data)
                    }
                    for s in snapshots
                ],
            }
            for run_id, snapshots in data.items()
        },
    }


@router.get("/{session_id}/{run_id}")
async def get_run_replay(session_id: str, run_id: str) -> Dict[str, Any]:
    """Return all epoch snapshots for a specific run, including weights."""
    snapshots = replay_store.get_run_replay(session_id, run_id)
    if snapshots is None:
        raise HTTPException(
            status_code=404,
            detail=f"No replay data found for run '{run_id}' in session '{session_id}'."
        )
    return {
        "session_id": session_id,
        "run_id": run_id,
        "snapshot_count": len(snapshots),
        "snapshots": snapshots,  # Full snapshots including weights
    }


@router.delete("/{session_id}")
async def clear_replay(session_id: str):
    """Clear all replay data for a session."""
    replay_store.clear_replay(session_id)
    return {"message": f"Replay data cleared for session '{session_id}'"}
