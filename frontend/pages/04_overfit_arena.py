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

LEVEL_ID = 4
COLOR = level_color(LEVEL_ID)
session_id = state_manager.get_session_id()
unlocked = state_manager.get("unlocked_features", [])
xp = state_manager.get("xp", 0)

render_xp_bar(xp, state_manager.get("current_level", 1), unlocked, session_id)

if not state_manager.is_level_unlocked(LEVEL_ID):
    st.warning("🔒 Complete Level 3 — Depth Trials to unlock this level.")
    st.stop()

st.markdown(f"""
<div class="page-header">
  <span class="level-badge" style="color:{COLOR};border-color:{COLOR};">LVL 4</span>
  <span style="font-size:1.3rem;font-weight:900;color:{COLOR};margin-left:12px;
        letter-spacing:0.08em;">THE OVERFIT ARENA</span>
  <div style="color:#4a5568;font-size:0.85rem;margin-top:4px;">
    Watch your model memorize noise in real time. Then tame it.
  </div>
</div>
""", unsafe_allow_html=True)

LEVEL_CONFIG = {
    "level_id": 4,
    "name": "THE OVERFIT ARENA",
    "challenge_description": "Keep the train/test accuracy gap below 5%.",
    "xp_reward": 300,
    "unlock_reward": ["imbalanced_dataset", "full_architecture"],
    "default_config": {
        "learning_rate": 0.01, "epochs": 200,
        "hidden_layers": 2, "neurons_per_layer": 32,
        "activation": "relu", "regularization": "none",
        "reg_lambda": 0.0, "batch_size": 32,
    },
}

tab_train, tab_exp = st.tabs(["🔥 Train", "🔬 Regularization Experiment"])

with tab_train:
    # Loss curve is PRIMARY — use wider center column
    col_ctrl, col_loss, col_boundary = st.columns([1, 2.5, 1.5])

    with col_ctrl:
        config = render_training_panel(LEVEL_CONFIG, unlocked, LEVEL_ID)
        render_architecture(config, key="arch_l4")

        st.markdown('<div class="train-btn">', unsafe_allow_html=True)
        train_clicked = st.button("▶ TRAIN", use_container_width=True, key="train_l4")
        st.markdown('</div>', unsafe_allow_html=True)

        result = state_manager.get("last_training_result_l4")
        if result:
            render_dna_chart(config, LEVEL_ID)

        def _submit_l4():
            r = state_manager.get("last_training_result_l4")
            run_id = state_manager.get("last_run_id_l4", "")
            if r and run_id:
                from utils.api_client import safe_validate
                ch = safe_validate(session_id, LEVEL_ID, run_id)
                if ch:
                    r["challenge_result"] = ch
                    state_manager.set("last_training_result_l4", r)
                    state_manager.sync_progress_from_api()
                    st.rerun()

        ch_result = extract_challenge_result(result) if result else None
        render_challenge_card(LEVEL_CONFIG, ch_result, on_submit=_submit_l4)

    if train_clicked:
        result = run_training(session_id, "noisy", config, LEVEL_ID, auto_validate=False)
        if result:
            state_manager.set("last_training_result_l4", result)
            state_manager.set("last_run_id_l4", result.get("run_id", ""))
            # Insight: first overfitting detection
            flags = result.get("failure_report", {}).get("flags", [])
            if "overfitting" in flags:
                state_manager.add_insight("overfit_seen", "The Generalization Gap")
            # Insight: first regularization applied
            if config.get("regularization", "none") != "none":
                state_manager.add_insight("reg_works", "Regularization Works")
        st.rerun()

    result = state_manager.get("last_training_result_l4")
    metrics = extract_metrics(result)
    failure = extract_failure(result)
    cards = extract_explanation_cards(result)

    train_acc = metrics.get("train_accuracy", 0)
    test_acc = metrics.get("test_accuracy", 0)
    gap = abs(train_acc - test_acc)
    is_overfitting = gap > 0.05 and bool(result)

    with col_loss:
        st.markdown('<div class="section-title">TRAINING CURVES — WATCH THE GAP</div>',
                    unsafe_allow_html=True)

        # Overfitting diagnosis overlay
        if is_overfitting:
            st.markdown("""
            <div class="diagnosis-overlay">
              <span style="color:#ff3b5c;font-weight:700;font-size:0.78rem;letter-spacing:0.1em;">
                ⚡ OVERFITTING DETECTED — ORANGE REGION IS THE GENERALIZATION GAP
              </span>
            </div>
            """, unsafe_allow_html=True)

        render_loss_curve(extract_history(result), key="loss_l4", show_accuracy=True)

        if metrics:
            c1, c2, c3 = st.columns(3)
            c1.metric("Train Acc", fmt_pct(train_acc))
            c2.metric("Test Acc", fmt_pct(test_acc))
            gap_color = "inverse" if gap > 0.05 else "normal"
            c3.metric("Gap", f"{gap*100:.1f}%", delta=f"Need <5%", delta_color=gap_color)

        render_failure_banners(failure, cards)

        render_micro_explainer(
            {"title": "🔥 What is Overfitting?",
             "body": "Your model has memorized the training noise instead of learning the true pattern. The orange gap between curves reveals how poorly it generalizes to unseen data.",
             "action_hint": "Enable L2 regularization and set λ=0.01 to penalize large weights."},
            trigger_condition=is_overfitting,
            explainer_key="overfit_explain",
        )

    with col_boundary:
        st.markdown('<div class="section-title">DECISION BOUNDARY</div>', unsafe_allow_html=True)
        render_boundary(extract_boundary(result), key="bd_l4")

        if is_overfitting:
            st.markdown("""
            <div class="failure-banner" style="font-size:0.78rem;">
              🔥 Boundary is fitting noise — too jagged for real patterns.
            </div>
            """, unsafe_allow_html=True)

with tab_exp:
    st.markdown("""
    <div class="forge-card">
      <div style="color:#ff3b5c;font-weight:800;margin-bottom:6px;">
        EXPERIMENT: No Regularization vs L2
      </div>
      <div style="color:#e2e8f0;font-size:0.86rem;line-height:1.6;">
        Two identical 2-layer networks on the noisy dataset. One with no regularization,
        one with L2 (λ=0.01). Watch the generalization gap shrink.
      </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("▶ RUN REGULARIZATION EXPERIMENT", key="run_reg_exp", use_container_width=True):
        with st.spinner("Running regularization experiment…"):
            exp = safe_run_experiment("REGULARIZATION_EXPERIMENT", session_id)
            if exp:
                state_manager.set("exp_result_l4", exp)
                st.rerun()

    exp = state_manager.get("exp_result_l4")
    if exp:
        render_comparison_panel(
            exp.get("model_a_result", {}),
            exp.get("model_b_result", {}),
            label_a="No Regularization",
            label_b="L2 (λ=0.01)",
        )
        st.markdown(f"""
        <div class="forge-card" style="border-color:#ff3b5c;margin-top:12px;">
          <div style="color:#ff3b5c;font-weight:700;margin-bottom:6px;">💡 What did this teach me?</div>
          <div style="color:#e2e8f0;font-size:0.86rem;line-height:1.6;">
            {exp.get("comparison_summary", "")}
          </div>
        </div>
        """, unsafe_allow_html=True)
