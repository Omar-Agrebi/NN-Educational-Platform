import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.level_helpers import *
from utils.formatters import level_color
from components.comparison_panel import render_comparison_panel
from utils.api_client import run_experiment, APIError

LEVEL_ID = 4
DATASET  = "noisy"

level_cfg = setup_page(LEVEL_ID)
color = level_color(LEVEL_ID)
xp_bar(sm.get("current_level",1), sm.get("xp",0), sm.get_session_id(), sm.get("insights",[]))
render_sidebar(LEVEL_ID)

level_header(LEVEL_ID, "THE OVERFIT ARENA",
             "Watch your model memorize noise. Then regularize it into submission.", color)

tab_train, tab_exp, tab_replay = st.tabs(["⚡ Training", "🔬 Reg Experiment", "⏪ Replay"])

with tab_train:
    col_ctrl, col_center, col_right = st.columns([1, 1.6, 1.2])

    with col_ctrl:
        model_config, train_config = render_training_panel(LEVEL_ID, level_cfg["default_config"])
        render_architecture(model_config, key="arch_l4")
        render_parameter_dna(model_config, train_config, color)
        st.markdown("<br>", unsafe_allow_html=True)
        train_clicked = st.button("▶ TRAIN MODEL", type="primary", use_container_width=True, key="train_l4")

    result = sm.get("last_training_result")

    with col_center:
        # Loss curve is PRIMARY here — this level is about the gap
        render_loss_curve(
            result.get("history", {}) if result else {},
            title="Train vs Test Loss — Watch the Gap",
            show_accuracy=True,
            height=220,
            key="loss_l4_main",
        )

        if result:
            h = result.get("history", {})
            ta = h.get("train_acc", [])
            va = h.get("test_acc", [])
            if ta and va:
                gap = abs(ta[-1] - va[-1])
                gap_color = "var(--green)" if gap < 0.05 else "var(--red)" if gap > 0.15 else "var(--orange)"
                st.markdown(
                    f"<div style='text-align:center;padding:12px;background:rgba(0,0,0,0.3);"
                    f"border:1px solid {gap_color};border-radius:8px;margin-top:10px'>"
                    f"<div style='font-family:var(--mono);font-size:9px;color:var(--muted)'>TRAIN / TEST GAP</div>"
                    f"<div style='font-family:var(--display);font-size:32px;font-weight:900;color:{gap_color}'>"
                    f"{gap:.1%}</div>"
                    f"<div style='font-size:10px;color:var(--muted)'>Target: &lt;5%</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        card_title("Decision Boundary")
        boundary = result.get("boundary_data") if result else None
        render_boundary(boundary, title="", height=240, key="bd_l4")

    with col_right:
        if result:
            render_metrics_row(result)
        render_failure_banners(
            result.get("failure_report", {}) if result else {},
            result.get("explanation_cards", []) if result else [],
        )
        render_challenge_card(level_cfg, result)
        if result and result.get("confusion_matrix"):
            render_confusion_matrix(result["confusion_matrix"])

    if train_clicked:
        result = do_training(DATASET, model_config, train_config)
        if result:
            st.rerun()

    if result and result.get("weight_snapshots"):
        with st.expander("🔬 Weight Inspector", expanded=False):
            render_weight_inspector(result.get("weight_snapshots", []), model_config)

with tab_exp:
    st.markdown(
        "<div style='font-size:13px;color:var(--muted);margin-bottom:16px;line-height:1.6'>"
        "No regularization vs L2 on noisy data. Same architecture, same everything. "
        "Only regularization differs.</div>",
        unsafe_allow_html=True,
    )
    if st.button("▶ Run Regularization Experiment", type="primary", key="run_reg_exp"):
        with st.spinner("Training with and without L2…"):
            try:
                exp_result = run_experiment("regularization_experiment", sm.get_session_id())
                sm.set("experiment_result_l4", exp_result)
            except APIError as e:
                st.error(f"Experiment failed: {e}")

    exp = sm.get("experiment_result_l4")
    if exp:
        st.markdown(
            f"<div class='success-banner' style='margin-bottom:16px'>"
            f"<strong style='color:var(--green)'>Lesson:</strong> {exp.get('lesson','')}</div>",
            unsafe_allow_html=True,
        )
        render_comparison_panel(
            exp.get("model_a_result"), exp.get("model_b_result"),
            label_a="No Regularization", label_b="L2 (λ=0.01)",
            archetype_a="The Regularization Denier", archetype_b="The Disciplined Generalizer",
        )

with tab_replay:
    try:
        from utils.api_client import get_replay
        render_replay_player(get_replay(sm.get_session_id()))
    except Exception:
        st.info("Run training first to enable replay.")
