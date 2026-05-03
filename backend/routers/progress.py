from fastapi import APIRouter, HTTPException
from models.schemas import ChallengeSubmitRequest, ProgressRequest
from models.responses import ProgressResponse
from storage.progress_store import get_progress, update_progress, unlock_features, award_xp, reset_progress
from storage.session_store import get_session
from progression.level_manager import get_all_levels
from progression.challenge_validator import validate

router = APIRouter(prefix="/progress", tags=["progress"])


@router.get("/{session_id}")
def get_progress_endpoint(session_id: str):
    prog = get_progress(session_id)
    levels = get_all_levels()
    return ProgressResponse(
        current_level=prog["current_level"],
        xp=prog["xp"],
        unlocked_features=prog["unlocked_features"],
        level_configs=levels,
        completed_challenges=prog["completed_challenges"],
        insights_collected=prog.get("insights_collected", []),
        total_runs=prog.get("total_runs", 0),
        best_scores=prog.get("best_scores", {}),
    )


@router.post("/validate-challenge")
def validate_challenge(req: ChallengeSubmitRequest):
    session = get_session(req.session_id)
    last_result = session.get("last_result")
    if not last_result:
        raise HTTPException(status_code=400, detail="No training result found for this session")

    result = validate(req.level_id, last_result)

    if result.passed:
        prog = get_progress(req.session_id)
        completed = prog.get("completed_challenges", [])
        if req.level_id not in completed:
            completed.append(req.level_id)
            award_xp(req.session_id, result.xp_earned)
            unlock_features(req.session_id, result.unlocked_features)
            new_level = max(prog["current_level"], req.level_id + 1)
            update_progress(req.session_id, {
                "completed_challenges": completed,
                "current_level": new_level,
            })

    return result


@router.post("/reset")
def reset(req: ProgressRequest):
    reset_progress(req.session_id)
    return {"message": "Progress reset"}
