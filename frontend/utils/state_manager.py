import streamlit as st
import uuid
from utils.api_client import get_progress, APIError


def init_state():
    defaults = {
        "session_id": str(uuid.uuid4())[:8],
        "current_level": 1,
        "xp": 0,
        "unlocked_features": ["learning_rate", "epochs", "slow_mode"],
        "last_training_result": None,
        "training_running": False,
        "replay_data": None,
        "selected_epoch": 0,
        "model_a_result": None,
        "model_b_result": None,
        "insights": [],
        "onboarding_done": False,
        "completed_challenges": [],
        "best_scores": {},
        "total_runs": 0,
        "dismissed_explainers": [],
        "sandbox_result": None,
        "lr_finder_result": None,
        "last_run_id": None,
        "baseline_result": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def get(key: str, default=None):
    return st.session_state.get(key, default)


def set(key: str, value):
    st.session_state[key] = value


def get_session_id() -> str:
    return st.session_state.get("session_id", "unknown")


def is_feature_unlocked(feature: str) -> bool:
    feats = st.session_state.get("unlocked_features", [])
    return feature in feats


def is_level_unlocked(level_id: int) -> bool:
    if level_id == 1:
        return True
    completed = st.session_state.get("completed_challenges", [])
    return (level_id - 1) in completed


def is_challenge_completed(level_id: int) -> bool:
    return level_id in st.session_state.get("completed_challenges", [])


def add_insight(insight: str):
    insights = st.session_state.get("insights", [])
    if insight not in insights:
        insights.append(insight)
        st.session_state["insights"] = insights
        return True
    return False


def sync_progress_from_api():
    try:
        sid = get_session_id()
        prog = get_progress(sid)
        st.session_state["current_level"] = prog.get("current_level", 1)
        st.session_state["xp"] = prog.get("xp", 0)
        st.session_state["unlocked_features"] = prog.get("unlocked_features", ["learning_rate", "epochs", "slow_mode"])
        st.session_state["completed_challenges"] = prog.get("completed_challenges", [])
        st.session_state["insights"] = prog.get("insights_collected", [])
        st.session_state["total_runs"] = prog.get("total_runs", 0)
        st.session_state["best_scores"] = prog.get("best_scores", {})
    except APIError:
        pass
