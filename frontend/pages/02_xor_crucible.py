import streamlit as st
import utils.state_manager as state_manager
from styles.theme import inject_theme
from components.xp_bar import render_xp_bar
from components.training_panel import render_training_panel, render_dna_chart
from components.loss_curve import render_loss_curve
from components.boundary_viz import render_boundary
from components.architecture_viz import render_architecture
from components.failure_banner import render_failure_banners
from components.challenge_card import render_challenge_card
from components.micro_explainer import render_micro_explainer
from components.comparison_panel import render_comparison_panel
from utils.training_helper import (
    run_training, extract_metrics, extract_history,
    extract_boundary, extract_failure, extract_explanation_cards, extract_challenge_result
)
from utils.formatters import fmt_pct, fmt_loss, level_color
from utils.api_client import safe_run_experiment

inject_theme()
state_manager.init_state()
state_manager.sync_progress_from_api()

LEVEL_ID = 2
COLOR = level_color(LEVEL_ID)
session_id = state_manager.get_session_id()
unlocked = state_manager.get("unlocked_features", [])
xp = state_manager.get("xp", 0)

render_xp_bar(xp, state_manager.get("current_level", 1), unlocked, session_id)

# ── Level gate ────────────────────────────────────────────────────────────────
if not state_manager.is_level_unlocked(LEVEL_ID):
    st.warning("🔒 Complete Level 1 — The Perceptron Forge to unlock this level.")
    st.stop()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="page-header">
  <span class="level-badge" style="color:{COLOR};border-color:{COLOR};">LVL 2</span>
  <span style="font-size:1.3rem;font-weight:900;color:{COLOR};margin-left:12px;
        letter-spacing:0.08em;">THE XOR CRUCIBLE</span>
  <div style="color:#4a5568;font-size:0.85rem;margin-top:4px;">
    The perceptron cannot solve XOR. Add hidden layers and break the barrier.
  </div>
</div>
""", unsafe_allow_html=True)

LEVEL_CONFIG = {
    "level_id": 2,
    "name": "THE XOR CRUCIBLE",
    "challenge_description": "Achieve test accuracy > 90% on the XOR dataset.",
    "xp_reward": 200,
    "unlock_reward": ["layer_depth", "noisy_dataset"],
    "default_config": {
        "learning_rate": 0.01, "epochs": 100,
        "hidden_layers": 0, "neurons_per_layer": 8,
        "activation": "relu", "regularization": "none",
        "reg_lambda": 0.0, "batch_size": 32,
    },
}

# ── XOR pattern preview ───────────────────────────────────────────────────────
st.markdown("""
<div class="forge-card" style="padding:12px 16px;margin-bottom:8px;">
  <span class="section-title">THE XOR PATTERN</span>
  <div style="display:flex;gap:24px;margin-top:6px;font-family:monospace;font-size:0.85rem;">
    <span>(−,−) → <span style="color:#ff3b5c;">●  0</span></span>
    <span>(+,+) → <span style="color:#ff3b5c;">●  0</span></span>
    <span>(−,+) → <span style="color:#00f5ff;">◆  1</span></span>
    <span>(+,−) → <span style="color:#00f5ff;">◆  1</span></span>
  </div>
  <div style="color:#4a5568;font-size:0.78rem;margin-top:6px;">
    No single straight line can separate these classes. A perceptron will plateau at ~50% accuracy.
  </div>
</div>
""", unsafe_allow_html=True)

tab_train, tab_exp = st.tabs(["🔥 Train", "🔬 Experiment Mode"])

with tab_train:
    col_ctrl, col_boundary, col_metrics = st.columns([1, 2, 1.5])

    with col_ctrl:
        config = render_training_panel(LEVEL_CONFIG, unlocked, LEVEL_ID)
        render_architecture(config, key="arch_l2")

        st.markdown('<div class="train-btn">', unsafe_allow_html=True)
        train_clicked = st.button("▶ TRAIN", use_container_width=True, key="train_l2")
        st.markdown('</div>', unsafe_allow_html=True)

        result = state_manager.get("last_training_result")
        if result:
            render_dna_chart(config, LEVEL_ID)

        def _submit_l2():
            r = state_manager.get("last_training_result")
            run_id = state_manager.get("last_run_id", "")
            if r and run_id:
                from utils.api_client import safe_validate
                ch = safe_validate(session_id, LEVEL_ID, run_id)
                if ch:
                    r["challenge_result"] = ch
                    state_manager.set("last_training_result", r)
                    state_manager.sync_progress_from_api()
                    st.rerun()

        render_challenge_card(LEVEL_CONFIG, extract_challenge_result(result), on_submit=_submit_l2)

    if train_clicked:
        result = run_training(session_id, "xor", config, LEVEL_ID)
        # Insight: first XOR failure
        metrics = extract_metrics(result) if result else {}
        if metrics.get("test_accuracy", 1.0) < 0.6 and config.get("hidden_layers", 0) == 0:
            state_manager.add_insight("xor_fail", "The XOR Problem")
        st.rerun()

    result = state_manager.get("last_training_result")
    failure = extract_failure(result)
    cards = extract_explanation_cards(result)
    metrics = extract_metrics(result)

    with col_boundary:
        st.markdown('<div class="section-title">DECISION BOUNDARY</div>', unsafe_allow_html=True)
        render_boundary(extract_boundary(result), key="bd_l2")

        # Failure explainer: perceptron straight line
        is_perceptron_fail = (
            config.get("hidden_layers", 0) == 0
            and metrics.get("test_accuracy", 1.0) < 0.65
            and bool(result)
        )
        render_micro_explainer(
            {"title": "⚠ Perceptron Cannot Solve XOR",
             "body": "A single neuron draws one straight line. XOR requires TWO lines — one for each pair of clusters. No matter how long you train, a perceptron will plateau near 50% on XOR.",
             "action_hint": "Add at least 1 hidden layer to unlock non-linear boundaries."},
            trigger_condition=is_perceptron_fail,
            explainer_key="xor_perceptron_fail",
        )

    with col_metrics:
        st.markdown('<div class="section-title">TRAINING CURVES</div>', unsafe_allow_html=True)
        render_loss_curve(extract_history(result), key="loss_l2")

        if metrics:
            c1, c2 = st.columns(2)
            c1.metric("Train Acc", fmt_pct(metrics.get("train_accuracy", 0)))
            c2.metric("Test Acc", fmt_pct(metrics.get("test_accuracy", 0)))
            c1.metric("Train Loss", fmt_loss(metrics.get("train_loss", 0)))
            c2.metric("Test Loss", fmt_loss(metrics.get("test_loss", 0)))

        render_failure_banners(failure, cards)

with tab_exp:
    st.markdown("""
    <div class="forge-card">
      <div style="color:#8b5cf6;font-weight:800;margin-bottom:6px;">
        EXPERIMENT: Perceptron vs MLP on XOR
      </div>
      <div style="color:#e2e8f0;font-size:0.86rem;line-height:1.6;">
        This experiment runs two models simultaneously: a perceptron (0 hidden layers) and
        a 2-layer MLP. Both use identical training settings. Watch the perceptron fail and the
        MLP succeed — this is the core lesson of this level.
      </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("▶ RUN EXPERIMENT", key="run_exp_l2", use_container_width=True):
        with st.spinner("Running both models…"):
            exp_result = safe_run_experiment("PERCEPTRON_VS_MLP_XOR", session_id)
            if exp_result:
                state_manager.set("exp_result_l2", exp_result)
                st.rerun()

    exp = state_manager.get("exp_result_l2")
    if exp:
        render_comparison_panel(
            exp.get("model_a_result", {}),
            exp.get("model_b_result", {}),
            label_a="Perceptron (0 layers)",
            label_b="MLP (2 layers)",
        )
        st.markdown(f"""
        <div class="forge-card" style="border-color:#8b5cf6;margin-top:12px;">
          <div style="color:#8b5cf6;font-weight:700;margin-bottom:6px;">💡 What did this teach me?</div>
          <div style="color:#e2e8f0;font-size:0.86rem;line-height:1.6;">
            {exp.get("comparison_summary", "")}
          </div>
        </div>
        """, unsafe_allow_html=True)
