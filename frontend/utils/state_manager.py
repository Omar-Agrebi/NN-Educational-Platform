import uuid
import streamlit as st
from typing import Any, Optional


_DEFAULTS = {
    "session_id": None,
    "current_level": 1,
    "xp": 0,
    "unlocked_features": ["learning_rate", "epochs", "batch_size", "linear_dataset"],
    "completed_challenges": [],
    "last_training_result": None,
    "training_running": False,
    "replay_data": None,
    "selected_epoch": 1,
    "model_a_result": None,
    "model_b_result": None,
    "insights": [],
    "onboarding_done": False,
    "dismissed_explainers": [],
    "last_run_id": None,
    "level_configs": [],
    "dna_history": [],
}


def init_state():
    for key, default in _DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default
    if st.session_state["session_id"] is None:
        st.session_state["session_id"] = str(uuid.uuid4())


def get(key: str, default: Any = None) -> Any:
    return st.session_state.get(key, default)


def set(key: str, value: Any):
    st.session_state[key] = value


def get_session_id() -> str:
    if "session_id" not in st.session_state or st.session_state["session_id"] is None:
        st.session_state["session_id"] = str(uuid.uuid4())
    return st.session_state["session_id"]


def is_feature_unlocked(feature_name: str) -> bool:
    features = st.session_state.get("unlocked_features", [])
    return feature_name in features


def is_level_unlocked(level_id: int) -> bool:
    if level_id == 1:
        return True
    completed = st.session_state.get("completed_challenges", [])
    return (level_id - 1) in completed


def is_level_completed(level_id: int) -> bool:
    completed = st.session_state.get("completed_challenges", [])
    return level_id in completed


def add_insight(insight_key: str, label: str):
    insights = st.session_state.get("insights", [])
    if insight_key not in [i["key"] for i in insights]:
        insights.append({"key": insight_key, "label": label})
        st.session_state["insights"] = insights
        st.toast(f"🪙 INSIGHT UNLOCKED: {label}", icon="🪙")


def sync_progress_from_api():
    """Pull latest progress from backend and update session state."""
    try:
        from utils.api_client import get_progress
        session_id = get_session_id()
        prog = get_progress(session_id)
        if prog:
            st.session_state["current_level"] = prog.get("current_level", 1)
            st.session_state["xp"] = prog.get("xp", 0)
            st.session_state["unlocked_features"] = prog.get("unlocked_features", _DEFAULTS["unlocked_features"])
            st.session_state["completed_challenges"] = prog.get("completed_challenges", [])
            st.session_state["level_configs"] = prog.get("level_configs", [])
    except Exception:
        pass  # offline — use local state


def get_level_config(level_id: int) -> Optional[dict]:
    configs = st.session_state.get("level_configs", [])
    for cfg in configs:
        if cfg.get("level_id") == level_id:
            return cfg
    return None
