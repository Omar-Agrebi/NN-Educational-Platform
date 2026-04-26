"""Shared helpers used by all level pages."""
import streamlit as st
import time
from utils import state_manager as sm
from utils.api_client import train_model, get_experiments, run_experiment, get_replay, APIError
from utils.formatters import level_color, fmt_pct, fmt_loss
from styles.theme import inject_theme, xp_bar, level_header, card_title, metric_box
from components.training_panel import render_training_panel
from components.loss_curve import render_loss_curve
from components.boundary_viz import render_boundary
from components.architecture_viz import render_architecture
from components.failure_banner import render_failure_banners
from components.challenge_card import render_challenge_card
from components.micro_explainer import check_and_show_insights
from components.weight_inspector import render_weight_inspector
from components.parameter_dna import render_parameter_dna
from components.replay_player import render_replay_player
from components.confusion_matrix import render_confusion_matrix
from components.lr_finder import render_lr_finder
from utils.api_client import get_progress as _get_progress_api

def get_level(level_id: int) -> dict:
    """Get level config from the backend API progress endpoint."""
    try:
        prog = _get_progress_api("__level_query__")
        level_configs = prog.get("level_configs", [])
        for lc in level_configs:
            if lc.get("level_id") == level_id:
                return lc
    except Exception:
        pass
    # Fallback minimal config
    return {
        "level_id": level_id,
        "name": f"Level {level_id}",
        "dataset": "linear",
        "default_config": {"learning_rate": 0.01, "epochs": 100, "hidden_layers": 0},
        "challenge": {"type": "test_accuracy", "target_metric": "test_acc", "target_value": 0.85,
                      "description": "Complete the challenge"},
        "xp_reward": 100,
        "tips": [],
        "color": "#00f5ff",
    }


def setup_page(level_id: int):
    """Call at top of every level page."""
    inject_theme()
    sm.init_state()
    sm.sync_progress_from_api()
    if not sm.is_level_unlocked(level_id):
        xp_bar(sm.get("current_level", 1), sm.get("xp", 0), sm.get_session_id(), sm.get("insights", []))
        st.warning(f"🔒 Complete Level {level_id - 1} to unlock this level.")
        if st.button("← Back to World Map"):
            st.switch_page("app.py")
        st.stop()
    return get_level(level_id)


def render_sidebar(level_id: int):
    color = level_color(level_id)
    with st.sidebar:
        st.markdown(
            f"<div style='font-family:var(--display);font-size:12px;color:{color};"
            f"letter-spacing:2px;margin-bottom:12px'>LEVEL 0{level_id}</div>",
            unsafe_allow_html=True,
        )
        if st.button("← World Map", use_container_width=True):
            st.switch_page("app.py")
        lvl_cfg = get_level(level_id)
        for tip in lvl_cfg.get("tips", [])[:2]:
            st.caption(tip)
        st.divider()
        st.markdown(f"**Session:** `{sm.get_session_id()[:8]}`")
        st.markdown(f"**XP:** {sm.get('xp', 0)}")
        st.markdown(f"**Runs:** {sm.get('total_runs', 0)}")


def do_training(dataset: str, model_config: dict, train_config: dict) -> dict:
    """Call backend, store result, return it."""
    with st.spinner("⚙ Forging…"):
        try:
            result = train_model(
                session_id=sm.get_session_id(),
                dataset=dataset,
                model_config=model_config,
                train_config=train_config,
            )
            sm.set("last_training_result", result)
            sm.set("last_run_id", result.get("run_id"))
            sm.sync_progress_from_api()
            check_and_show_insights(result)
            return result
        except APIError as e:
            st.error(f"Training failed: {e}")
            return None


def render_metrics_row(result: dict):
    if not result:
        return
    h = result.get("history", {})
    m = result.get("final_metrics", {})
    train_accs = h.get("train_acc", [])
    test_accs  = h.get("test_acc", [])
    train_loss = h.get("train_loss", [])
    test_loss  = h.get("test_loss", [])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_box("Train Acc", fmt_pct(train_accs[-1] if train_accs else None), "var(--cyan)")
    with c2:
        metric_box("Test Acc",  fmt_pct(test_accs[-1]  if test_accs  else None), "var(--orange)")
    with c3:
        metric_box("Train Loss", fmt_loss(train_loss[-1] if train_loss else None), "var(--cyan)")
    with c4:
        metric_box("Test Loss",  fmt_loss(test_loss[-1]  if test_loss  else None), "var(--orange)")

    gap = abs((train_accs[-1] if train_accs else 0) - (test_accs[-1] if test_accs else 0))
    f1  = m.get("f1_score", 0)
    c5, c6 = st.columns(2)
    with c5:
        gap_color = "var(--green)" if gap < 0.05 else "var(--red)" if gap > 0.15 else "var(--orange)"
        metric_box("Gen Gap", fmt_pct(gap), gap_color, "train vs test accuracy")
    with c6:
        f1_color = "var(--green)" if f1 > 0.8 else "var(--orange)" if f1 > 0.5 else "var(--red)"
        metric_box("F1 Score", f"{f1:.3f}", f1_color)
