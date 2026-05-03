import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.level_helpers import *
from utils.formatters import level_color
from components.comparison_panel import render_comparison_panel
from utils.api_client import run_experiment, APIError

LEVEL_ID = 3
DATASET  = "xor"

level_cfg = setup_page(LEVEL_ID)
color = level_color(LEVEL_ID)
xp_bar(sm.get("current_level",1), sm.get("xp",0), sm.get_session_id(), sm.get("insights",[]))
render_sidebar(LEVEL_ID)

level_header(LEVEL_ID, "DEPTH TRIALS",
             "Does adding more layers always help? Prove it with numbers.", color)

tab_train, tab_exp, tab_replay = st.tabs(["⚡ Training", "🔬 Depth Experiment", "⏪ Replay"])

with tab_train:
    col_ctrl, col_center, col_right = st.columns([1, 1.6, 1.2])

    with col_ctrl:
        model_config, train_config = render_training_panel(LEVEL_ID, level_cfg["default_config"])
        render_architecture(model_config, key="arch_l3")
        render_parameter_dna(model_config, train_config, color)
        st.markdown("<br>", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            set_baseline = st.button("📌 Set Baseline", use_container_width=True, key="baseline_l3")
        with col_b:
            train_clicked = st.button("▶ TRAIN", type="primary", use_container_width=True, key="train_l3")

    result   = sm.get("last_training_result")
    baseline = sm.get("baseline_result")

    with col_center:
        card_title("Decision Boundary")
        boundary = result.get("boundary_data") if result else None
        render_boundary(boundary, title="", height=260, key="bd_l3")

        # Accuracy delta display
        if result and baseline:
            base_accs = baseline.get("history", {}).get("test_acc", [])
            curr_accs = result.get("history", {}).get("test_acc", [])
            if base_accs and curr_accs:
                gain = curr_accs[-1] - base_accs[-1]
                gain_color = "var(--green)" if gain >= 0.10 else "var(--orange)" if gain > 0 else "var(--red)"
                st.markdown(
                    f"<div style='background:rgba(0,0,0,0.3);border:1px solid {gain_color};"
                    f"border-radius:8px;padding:12px;text-align:center;margin-top:10px'>"
                    f"<div style='font-family:var(--mono);font-size:9px;color:var(--muted)'>ACCURACY GAIN vs BASELINE</div>"
                    f"<div style='font-family:var(--display);font-size:28px;font-weight:900;color:{gain_color}'>"
                    f"{'+' if gain >= 0 else ''}{gain:.1%}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    with col_right:
        render_loss_curve(result.get("history", {}) if result else {}, key="loss_l3")
        if result:
            render_metrics_row(result)
        render_failure_banners(
            result.get("failure_report", {}) if result else {},
            result.get("explanation_cards", []) if result else [],
        )
        render_challenge_card(level_cfg, result)

    if set_baseline:
        result_tmp = do_training(DATASET, model_config, train_config)
        if result_tmp:
            sm.set("baseline_result", result_tmp)
            st.success("Baseline set! Now add more layers and retrain to compare.")
            st.rerun()

    if train_clicked:
        result = do_training(DATASET, model_config, train_config)
        if result:
            st.rerun()

with tab_exp:
    st.markdown(
        "<div style='font-size:13px;color:var(--muted);margin-bottom:16px;line-height:1.6'>"
        "Controlled experiment: 1 layer vs 3 layers. Same dataset, same LR, same epochs. "
        "Only depth changes.</div>",
        unsafe_allow_html=True,
    )
    if st.button("▶ Run Depth Experiment", type="primary", key="run_depth_exp"):
        with st.spinner("Training both depths…"):
            try:
                exp_result = run_experiment("depth_experiment", sm.get_session_id())
                sm.set("experiment_result_l3", exp_result)
            except APIError as e:
                st.error(f"Experiment failed: {e}")

    exp = sm.get("experiment_result_l3")
    if exp:
        st.markdown(
            f"<div class='success-banner' style='margin-bottom:16px'>"
            f"<strong style='color:var(--green)'>Lesson:</strong> {exp.get('lesson','')}</div>",
            unsafe_allow_html=True,
        )
        render_comparison_panel(
            exp.get("model_a_result"), exp.get("model_b_result"),
            label_a="Shallow (1 layer)", label_b="Deep (3 layers)",
            archetype_a="The Shallow Thinker", archetype_b="The Deep Reasoner",
        )

with tab_replay:
    try:
        from utils.api_client import get_replay
        render_replay_player(get_replay(sm.get_session_id()))
    except Exception:
        st.info("Run training first to enable replay.")
