"""
Shared training pipeline used by all level pages.
Handles API call, state updates, challenge validation, insight triggers.
"""
import streamlit as st
import utils.state_manager as state_manager
from utils.api_client import safe_train, safe_validate


def run_training(session_id: str, dataset: str, config: dict, level_id: int,
                 auto_validate: bool = True) -> dict | None:
    """
    Execute a training run and persist result to session state.
    Returns the TrainingResult dict or None on failure.
    """
    model_cfg = {
        "hidden_layers": config.get("hidden_layers", 0),
        "neurons_per_layer": config.get("neurons_per_layer", 8),
        "activation": config.get("activation", "relu"),
        "regularization": config.get("regularization", "none"),
        "reg_lambda": config.get("reg_lambda", 0.001),
    }
    train_cfg = {
        "learning_rate": config.get("learning_rate", 0.01),
        "epochs": config.get("epochs", 100),
        "batch_size": config.get("batch_size", 32),
        "slow_mode": config.get("slow_mode", False),
    }

    state_manager.set("training_running", True)
    with st.spinner("⚙ Training in progress…"):
        result = safe_train(session_id, dataset, model_cfg, train_cfg)
    state_manager.set("training_running", False)

    if result is None:
        return None

    state_manager.set("last_training_result", result)
    run_id = result.get("run_id", "")
    state_manager.set("last_run_id", run_id)

    # ── Insight triggers ──────────────────────────────────────────────
    failure = result.get("failure_report", {})
    flags = failure.get("flags", [])

    if "overfitting" in flags:
        state_manager.add_insight("overfit_seen", "The Generalization Gap")

    metrics = result.get("final_metrics", {})
    acc = metrics.get("test_accuracy", 0)
    f1 = metrics.get("f1_score", 0)

    if dataset == "xor" and acc < 0.6:
        state_manager.add_insight("xor_fail", "The XOR Problem")

    if dataset == "noisy" and "regularization" in flags.__class__.__name__:
        pass  # handled elsewhere

    if auto_validate and run_id:
        ch_result = safe_validate(session_id, level_id, run_id)
        if ch_result:
            result["challenge_result"] = ch_result
            if ch_result.get("is_gate_passed"):
                # Sync newly unlocked features from API
                state_manager.sync_progress_from_api()
                st.balloons()

    return result


def extract_metrics(result: dict) -> dict:
    return result.get("final_metrics", {}) if result else {}


def extract_history(result: dict) -> dict:
    return result.get("history", {}) if result else {}


def extract_boundary(result: dict) -> dict | None:
    return result.get("boundary_data") if result else None


def extract_failure(result: dict) -> dict:
    return result.get("failure_report", {}) if result else {}


def extract_explanation_cards(result: dict) -> list:
    return result.get("explanation_cards", []) if result else []


def extract_challenge_result(result: dict) -> dict | None:
    return result.get("challenge_result") if result else None
