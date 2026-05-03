import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.level_helpers import *
from utils.formatters import level_color
from components.comparison_panel import render_comparison_panel
from utils.api_client import run_experiment, APIError

LEVEL_ID = 2
DATASET  = "xor"

level_cfg = setup_page(LEVEL_ID)
color = level_color(LEVEL_ID)
xp_bar(sm.get("current_level",1), sm.get("xp",0), sm.get_session_id(), sm.get("insights",[]))
render_sidebar(LEVEL_ID)

level_header(LEVEL_ID, "THE XOR CRUCIBLE",
             "A perceptron draws one straight line. XOR needs a curve. Watch it fail, then fix it.", color)

tab_train, tab_exp, tab_replay = st.tabs(["⚡ Training", "🔬 Experiment Mode", "⏪ Replay"])

with tab_train:
    col_ctrl, col_center, col_right = st.columns([1, 1.6, 1.2])

    with col_ctrl:
        model_config, train_config = render_training_panel(LEVEL_ID, level_cfg["default_config"])
        st.markdown("<br>", unsafe_allow_html=True)
        render_architecture(model_config, key="arch_l2")
        render_parameter_dna(model_config, train_config, color)
        st.markdown("<br>", unsafe_allow_html=True)
        train_clicked = st.button("▶ TRAIN MODEL", type="primary", use_container_width=True, key="train_l2")

    result = sm.get("last_training_result")

    with col_center:
        card_title("Decision Boundary")
        boundary = result.get("boundary_data") if result else None
        render_boundary(boundary, title="", height=320, key="bd_l2")

        # XOR failure explainer — show when perceptron on XOR
        if result:
            hl = model_config.get("hidden_layers", 0)
            flags = result.get("failure_report", {}).get("flags", [])
            if hl == 0 and "underfitting" in flags:
                st.markdown("""
<div class="failure-banner" style="margin-top:10px">
  <div style="font-size:16px">⚠</div>
  <div>
    <div style="font-size:13px;font-weight:700;color:var(--red);margin-bottom:3px">Perceptron Cannot Solve XOR</div>
    <div style="font-size:11px;color:var(--muted);line-height:1.5">
      A single neuron can only draw a straight line. XOR requires a curved decision boundary — 
      mathematically impossible without hidden layers.
    </div>
    <div style="font-size:11px;color:var(--cyan);font-style:italic;margin-top:4px">
      → Add at least 1 hidden layer to break linearity
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    with col_right:
        render_loss_curve(result.get("history", {}) if result else {}, key="loss_l2")
        if result:
            render_metrics_row(result)
        render_failure_banners(
            result.get("failure_report", {}) if result else {},
            result.get("explanation_cards", []) if result else [],
        )
        render_challenge_card(level_cfg, result)

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
        "This experiment runs two models on XOR — one with 0 hidden layers, one with 2. "
        "Everything else is identical. Only architecture changes.</div>",
        unsafe_allow_html=True,
    )
    if st.button("▶ Run Perceptron vs MLP Experiment", type="primary", key="run_exp_l2"):
        with st.spinner("Running both models…"):
            try:
                exp_result = run_experiment("perceptron_vs_mlp_xor", sm.get_session_id())
                sm.set("experiment_result_l2", exp_result)
            except APIError as e:
                st.error(f"Experiment failed: {e}")

    exp = sm.get("experiment_result_l2")
    if exp:
        st.markdown(
            f"<div class='success-banner' style='margin-bottom:16px'>"
            f"<strong style='color:var(--green)'>Lesson:</strong> {exp.get('lesson','')}</div>",
            unsafe_allow_html=True,
        )
        render_comparison_panel(
            exp.get("model_a_result"), exp.get("model_b_result"),
            label_a="Perceptron (0 layers)", label_b="MLP (2 layers)",
            archetype_a="The Overconfident Perceptron", archetype_b="The Enlightened MLP",
        )

with tab_replay:
    try:
        from utils.api_client import get_replay
        render_replay_player(get_replay(sm.get_session_id()))
    except Exception:
        st.info("Run training first to enable replay.")
