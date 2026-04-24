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
from components.comparison_panel import render_comparison_panel
from utils.training_helper import (
    run_training, extract_metrics, extract_history,
    extract_boundary, extract_failure, extract_explanation_cards, extract_challenge_result
)
from utils.formatters import fmt_pct, fmt_loss, level_color
from utils.api_client import safe_run_experiment, safe_train

inject_theme()
state_manager.init_state()
state_manager.sync_progress_from_api()

LEVEL_ID = 3
COLOR = level_color(LEVEL_ID)
session_id = state_manager.get_session_id()
unlocked = state_manager.get("unlocked_features", [])
xp = state_manager.get("xp", 0)

render_xp_bar(xp, state_manager.get("current_level", 1), unlocked, session_id)

if not state_manager.is_level_unlocked(LEVEL_ID):
    st.warning("🔒 Complete Level 2 — The XOR Crucible to unlock this level.")
    st.stop()

st.markdown(f"""
<div class="page-header">
  <span class="level-badge" style="color:{COLOR};border-color:{COLOR};">LVL 3</span>
  <span style="font-size:1.3rem;font-weight:900;color:{COLOR};margin-left:12px;
        letter-spacing:0.08em;">DEPTH TRIALS</span>
  <div style="color:#4a5568;font-size:0.85rem;margin-top:4px;">
    How deep is deep enough? Prove a >10% accuracy gain over the 1-layer baseline.
  </div>
</div>
""", unsafe_allow_html=True)

LEVEL_CONFIG = {
    "level_id": 3,
    "name": "DEPTH TRIALS",
    "challenge_description": "Improve test accuracy by more than 10% vs the 1-layer baseline.",
    "xp_reward": 200,
    "unlock_reward": ["regularization", "noisy_dataset"],
    "default_config": {
        "learning_rate": 0.01, "epochs": 100,
        "hidden_layers": 1, "neurons_per_layer": 8,
        "activation": "relu", "regularization": "none",
        "reg_lambda": 0.0, "batch_size": 32,
    },
}

# ── Auto-run baseline on first visit ─────────────────────────────────────────
if state_manager.get("baseline_result_l3") is None:
    with st.spinner("Running 1-layer baseline…"):
        baseline = safe_train(session_id, "xor", {
            "hidden_layers": 1, "neurons_per_layer": 8,
            "activation": "relu", "regularization": "none", "reg_lambda": 0.0,
        }, {
            "learning_rate": 0.01, "epochs": 100, "batch_size": 32, "slow_mode": False,
        })
        if baseline:
            state_manager.set("baseline_result_l3", baseline)

baseline_result = state_manager.get("baseline_result_l3")
baseline_acc = baseline_result.get("final_metrics", {}).get("test_accuracy", 0) if baseline_result else 0

if baseline_acc:
    st.markdown(f"""
    <div class="forge-card" style="padding:10px 16px;margin-bottom:8px;">
      <span class="section-title">BASELINE (1 HIDDEN LAYER)</span>
      <span style="font-family:monospace;color:#ff6b2b;font-size:1.1rem;margin-left:12px;">
        Test Acc: {fmt_pct(baseline_acc)}
      </span>
      <span style="color:#4a5568;font-size:0.78rem;margin-left:12px;">
        You need to beat this by >10% with more layers.
      </span>
    </div>
    """, unsafe_allow_html=True)

tab_train, tab_exp = st.tabs(["🔥 Train", "🔬 Depth Experiment"])

with tab_train:
    col_ctrl, col_left_bd, col_right_panel = st.columns([1, 2, 2])

    with col_ctrl:
        config = render_training_panel(LEVEL_CONFIG, unlocked, LEVEL_ID)
        render_architecture(config, key="arch_l3")

        st.markdown('<div class="train-btn">', unsafe_allow_html=True)
        train_clicked = st.button("▶ TRAIN", use_container_width=True, key="train_l3")
        st.markdown('</div>', unsafe_allow_html=True)

        result = state_manager.get("last_training_result_l3")
        if result:
            render_dna_chart(config, LEVEL_ID)

        def _submit_l3():
            r = state_manager.get("last_training_result_l3")
            run_id = state_manager.get("last_run_id_l3", "")
            if r and run_id:
                from utils.api_client import safe_validate
                # Inject accuracy_gain into result for validator
                current_acc = r.get("final_metrics", {}).get("test_accuracy", 0)
                gain = current_acc - baseline_acc
                r["accuracy_gain"] = gain
                ch = safe_validate(session_id, LEVEL_ID, run_id)
                if ch:
                    r["challenge_result"] = ch
                    state_manager.set("last_training_result_l3", r)
                    state_manager.sync_progress_from_api()
                    st.rerun()

        ch_result = extract_challenge_result(result) if result else None
        render_challenge_card(LEVEL_CONFIG, ch_result, on_submit=_submit_l3)

    if train_clicked:
        result = run_training(session_id, "xor", config, LEVEL_ID, auto_validate=False)
        if result:
            state_manager.set("last_training_result_l3", result)
            state_manager.set("last_run_id_l3", result.get("run_id", ""))
        st.rerun()

    result = state_manager.get("last_training_result_l3")
    current_acc = extract_metrics(result).get("test_accuracy", 0) if result else 0
    gain = current_acc - baseline_acc

    with col_left_bd:
        # Side-by-side boundaries: baseline left, current right
        st.markdown('<div class="section-title">BASELINE BOUNDARY (1 layer)</div>', unsafe_allow_html=True)
        render_boundary(
            baseline_result.get("boundary_data") if baseline_result else None,
            title="Baseline (1 Layer)", key="bd_baseline_l3"
        )

        if gain:
            color = "#00ff88" if gain > 0.10 else "#ff6b2b"
            st.markdown(f"""
            <div class="forge-card" style="text-align:center;padding:10px;">
              <div style="color:#4a5568;font-size:0.72rem;">ACCURACY GAIN</div>
              <div style="font-family:monospace;color:{color};font-size:1.5rem;font-weight:800;">
                {'+' if gain > 0 else ''}{gain*100:.1f}%
              </div>
              <div style="color:#4a5568;font-size:0.72rem;">Need >10% to pass</div>
            </div>
            """, unsafe_allow_html=True)

    with col_right_panel:
        st.markdown('<div class="section-title">YOUR BOUNDARY</div>', unsafe_allow_html=True)
        render_boundary(extract_boundary(result), title="Your Model", key="bd_l3")
        render_loss_curve(extract_history(result), key="loss_l3", show_accuracy=False)

        metrics = extract_metrics(result)
        if metrics:
            c1, c2 = st.columns(2)
            c1.metric("Test Acc", fmt_pct(current_acc))
            c2.metric("Gain", f"{'+' if gain>=0 else ''}{gain*100:.1f}%",
                      delta_color="normal" if gain > 0 else "inverse")

        render_failure_banners(extract_failure(result), extract_explanation_cards(result))

with tab_exp:
    st.markdown("""
    <div class="forge-card">
      <div style="color:#ff6b2b;font-weight:800;margin-bottom:6px;">
        EXPERIMENT: Shallow vs Deep
      </div>
      <div style="color:#e2e8f0;font-size:0.86rem;line-height:1.6;">
        A 1-layer network vs a 3-layer network on XOR — identical learning rate, epochs, batch size.
        Only depth changes. Watch what that single difference does to the decision boundary.
      </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("▶ RUN DEPTH EXPERIMENT", key="run_depth_exp", use_container_width=True):
        with st.spinner("Running depth experiment…"):
            exp = safe_run_experiment("DEPTH_EXPERIMENT", session_id)
            if exp:
                state_manager.set("exp_result_l3", exp)
                st.rerun()

    exp = state_manager.get("exp_result_l3")
    if exp:
        render_comparison_panel(
            exp.get("model_a_result", {}),
            exp.get("model_b_result", {}),
            label_a="Shallow (1 layer)",
            label_b="Deep (3 layers)",
        )
        st.markdown(f"""
        <div class="forge-card" style="border-color:#ff6b2b;margin-top:12px;">
          <div style="color:#ff6b2b;font-weight:700;margin-bottom:6px;">💡 What did this teach me?</div>
          <div style="color:#e2e8f0;font-size:0.86rem;line-height:1.6;">
            {exp.get("comparison_summary", "")}
          </div>
        </div>
        """, unsafe_allow_html=True)
