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
from utils.training_helper import (
    run_training, extract_metrics, extract_history,
    extract_boundary, extract_failure, extract_explanation_cards, extract_challenge_result
)
from utils.formatters import fmt_pct, fmt_loss, level_color

inject_theme()
state_manager.init_state()
state_manager.sync_progress_from_api()

LEVEL_ID = 1
COLOR = level_color(LEVEL_ID)
session_id = state_manager.get_session_id()
unlocked = state_manager.get("unlocked_features", [])
xp = state_manager.get("xp", 0)

render_xp_bar(xp, state_manager.get("current_level", 1), unlocked, session_id)

# ── Level gate ────────────────────────────────────────────────────────────────
# Level 1 is always unlocked

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="page-header">
  <span class="level-badge" style="color:{COLOR};border-color:{COLOR};">LVL 1</span>
  <span style="font-size:1.3rem;font-weight:900;color:{COLOR};margin-left:12px;
        letter-spacing:0.08em;">THE PERCEPTRON FORGE</span>
  <div style="color:#4a5568;font-size:0.85rem;margin-top:4px;">
    Can a single neuron separate the world?
  </div>
</div>
""", unsafe_allow_html=True)

LEVEL_CONFIG = {
    "level_id": 1,
    "name": "THE PERCEPTRON FORGE",
    "challenge_description": "Achieve train accuracy > 85% on the linear dataset.",
    "xp_reward": 100,
    "unlock_reward": ["hidden_layers", "activation", "xor_dataset"],
    "default_config": {
        "learning_rate": 0.001, "epochs": 50,
        "hidden_layers": 0, "neurons_per_layer": 8,
        "activation": "sigmoid", "regularization": "none",
        "reg_lambda": 0.0, "batch_size": 32,
    },
}

# ── Layout ────────────────────────────────────────────────────────────────────
col_ctrl, col_boundary, col_metrics = st.columns([1, 2, 1.5])

with col_ctrl:
    config = render_training_panel(LEVEL_CONFIG, unlocked, LEVEL_ID)
    render_architecture(config, key="arch_l1")

    st.markdown('<div class="train-btn">', unsafe_allow_html=True)
    train_clicked = st.button("▶ TRAIN", use_container_width=True, key="train_l1")
    st.markdown('</div>', unsafe_allow_html=True)

    result = state_manager.get("last_training_result")
    if result:
        render_dna_chart(config, LEVEL_ID)

    def _submit():
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

    render_challenge_card(LEVEL_CONFIG, extract_challenge_result(result), on_submit=_submit)

# ── Training trigger ──────────────────────────────────────────────────────────
if train_clicked:
    result = run_training(session_id, "linear", config, LEVEL_ID)
    st.rerun()

result = state_manager.get("last_training_result")

with col_boundary:
    st.markdown('<div class="section-title">DECISION BOUNDARY</div>', unsafe_allow_html=True)
    render_boundary(extract_boundary(result), key="bd_l1")

with col_metrics:
    st.markdown('<div class="section-title">TRAINING CURVES</div>', unsafe_allow_html=True)
    render_loss_curve(extract_history(result), key="loss_l1")

    metrics = extract_metrics(result)
    if metrics:
        c1, c2 = st.columns(2)
        c1.metric("Train Acc", fmt_pct(metrics.get("train_accuracy", 0)))
        c2.metric("Test Acc", fmt_pct(metrics.get("test_accuracy", 0)))
        c1.metric("Train Loss", fmt_loss(metrics.get("train_loss", 0)))
        c2.metric("Test Loss", fmt_loss(metrics.get("test_loss", 0)))

    failure = extract_failure(result)
    cards = extract_explanation_cards(result)
    render_failure_banners(failure, cards)

    render_micro_explainer(
        {"title": "How the Perceptron Works",
         "body": "A perceptron computes a weighted sum of inputs and applies a sigmoid. It can only draw a single straight decision boundary — perfect for linearly separable data.",
         "action_hint": "Try increasing the learning rate if accuracy is too low."},
        trigger_condition=bool(result),
        explainer_key="perceptron_intro",
    )

# ── Experiment tab ────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("🔬 What makes this problem easy?", expanded=False):
    st.markdown("""
    <div class="forge-card">
      <div style="color:#00f5ff;font-weight:700;margin-bottom:8px;">The Linear Dataset</div>
      <div style="color:#e2e8f0;font-size:0.88rem;line-height:1.7;">
        The linear dataset contains two Gaussian clusters arranged so that a single straight line
        cleanly separates them. A perceptron — just an input layer connected directly to an output —
        can find this line. This is the only problem a perceptron can solve.
        Try the XOR dataset at Level 2 to see it fail spectacularly.
      </div>
    </div>
    """, unsafe_allow_html=True)
