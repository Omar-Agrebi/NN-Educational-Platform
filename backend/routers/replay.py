from fastapi import APIRouter, HTTPException
from storage.replay_store import get_replay, get_replay_run, clear_replay

router = APIRouter(prefix="/replay", tags=["replay"])


@router.get("/{session_id}")
def get_all_replays(session_id: str):
    return get_replay(session_id)


@router.get("/{session_id}/{run_id}")
def get_run_replay(session_id: str, run_id: str):
    data = get_replay_run(session_id, run_id)
    if not data:
        raise HTTPException(status_code=404, detail="Replay not found")
    return data


@router.delete("/{session_id}")
def delete_replay(session_id: str):
    clear_replay(session_id)
    return {"message": "Replay cleared"}
