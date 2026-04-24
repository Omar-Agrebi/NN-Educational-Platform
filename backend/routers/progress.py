from fastapi import APIRouter, HTTPException
from typing import List

from models.schemas import ChallengeSubmitRequest, ProgressRequest
from models.responses import ProgressResponse, LevelConfig as LevelConfigResponse, ChallengeResult
from storage.progress_store import progress_store
from storage.session_store import session_store
from storage.replay_store import replay_store
from progression.level_manager import get_all_levels, get_level
from progression.challenge_validator import validate

router = APIRouter(prefix="/progress", tags=["progress"])


def _build_level_config_response(level) -> LevelConfigResponse:
    return LevelConfigResponse(
        level_id=level.level_id,
        name=level.name,
        description=level.description,
        dataset=level.dataset,
        locked_params=level.locked_params,
        unlocked_params=level.unlocked_params,
        default_config=level.default_config,
        xp_reward=level.xp_reward,
        challenge_description=level.challenge.description,
        unlock_reward=level.unlock_reward,
    )


@router.get("/{session_id}", response_model=ProgressResponse)
async def get_progress(session_id: str):
    """Return full progress state for a session."""
    prog = progress_store.get_progress(session_id)
    all_levels = get_all_levels()
    level_configs = [_build_level_config_response(lvl) for lvl in all_levels]

    return ProgressResponse(
        current_level=prog["current_level"],
        xp=prog["xp"],
        unlocked_features=prog["unlocked_features"],
        level_configs=level_configs,
        completed_challenges=prog["completed_challenges"],
    )


@router.post("/validate-challenge", response_model=ChallengeResult)
async def validate_challenge(request: ChallengeSubmitRequest):
    """
    Submit a completed run for challenge validation.
    Checks metrics against the level's challenge definition.
    Awards XP and unlocks features if passed.
    """
    session_id = request.session_id
    level_id = request.level_id
    run_id = request.run_id

    # Load last training history for this session
    history = session_store.get_last_history(session_id)
    if history is None:
        raise HTTPException(
            status_code=404,
            detail="No training history found. Run /train first."
        )

    # Load level config
    try:
        level = get_level(level_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Build training result dict for validator
    from analysis.metrics import (
        accuracy as acc_fn, f1_score as f1_fn,
        precision as prec_fn, recall as rec_fn
    )
    from storage.session_store import session_store as ss
    from core.neural_net import NeuralNetwork
    import numpy as np

    split = ss.get_dataset_split(session_id)
    if split is None:
        raise HTTPException(status_code=404, detail="No dataset split found for session.")

    X_train = np.array(split["X_train"])
    X_test = np.array(split["X_test"])
    y_train = np.array(split["y_train"])
    y_test = np.array(split["y_test"])

    weights = ss.get_weights(session_id)
    if weights is None:
        raise HTTPException(status_code=404, detail="No model weights found for session.")

    cfg = weights["config"]
    model = NeuralNetwork(
        input_size=cfg["input_size"],
        hidden_layers=cfg["hidden_layers"],
        neurons_per_layer=cfg["neurons_per_layer"],
        output_size=cfg["output_size"],
        activation=cfg["activation"],
    )
    model.set_weights(weights)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_acc = acc_fn(y_train, y_pred_train)
    test_acc = acc_fn(y_test, y_pred_test)
    f1 = f1_fn(y_test, y_pred_test)

    training_result = {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "f1_score": f1,
        "train_test_gap": abs(train_acc - test_acc),
        "accuracy_gain": 0.0,  # handled by experiment configs for LVL 3
    }

    # Validate
    result = validate(level.challenge, training_result, level_id)

    # Apply rewards
    if result.is_gate_passed:
        progress_store.complete_challenge(session_id, level_id)
        progress_store.award_xp(session_id, result.xp_earned)
        progress_store.unlock_features(session_id, result.unlocked_features)
        progress_store.advance_level(session_id, level_id + 1)
    elif result.xp_earned > 0:
        progress_store.award_xp(session_id, result.xp_earned)

    return ChallengeResult(
        passed=result.passed,
        is_gate_passed=result.is_gate_passed,
        score=result.score,
        xp_earned=result.xp_earned,
        message=result.message,
        unlocked_features=result.unlocked_features,
    )


@router.post("/reset")
async def reset_progress(request: ProgressRequest):
    """Reset progress to default state. Dev/testing use only."""
    progress_store.reset_progress(request.session_id)
    replay_store.clear_replay(request.session_id)
    return {"message": f"Progress reset for session {request.session_id}"}
