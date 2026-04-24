import streamlit as st
import plotly.graph_objects as go
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

LEVEL_ID = 5
COLOR = level_color(LEVEL_ID)
session_id = state_manager.get_session_id()
unlocked = state_manager.get("unlocked_features", [])
xp = state_manager.get("xp", 0)

render_xp_bar(xp, state_manager.get("current_level", 1), unlocked, session_id)

if not state_manager.is_level_unlocked(LEVEL_ID):
    st.warning("🔒 Complete Level 4 — The Overfit Arena to unlock this level.")
    st.stop()

st.markdown(f"""
<div class="page-header">
  <span class="level-badge" style="color:{COLOR};border-color:{COLOR};">LVL 5</span>
  <span style="font-size:1.3rem;font-weight:900;color:{COLOR};margin-left:12px;
        letter-spacing:0.08em;">THE ARCHITECT CHAMBER</span>
  <div style="color:#4a5568;font-size:0.85rem;margin-top:4px;">
    Accuracy lies on imbalanced data. Build for F1 score > 0.88.
  </div>
</div>
""", unsafe_allow_html=True)

# Dataset warning banner
st.markdown("""
<div class="forge-card" style="border-color:#00ff88;padding:10px 16px;margin-bottom:8px;">
  <span style="color:#00ff88;font-weight:700;">⚖ IMBALANCED DATASET</span>
  <span style="color:#4a5568;font-size:0.82rem;margin-left:10px;">
    85% class 0 · 15% class 1 — predicting all-zeros gets 85% accuracy but 0% F1.
  </span>
</div>
""", unsafe_allow_html=True)

LEVEL_CONFIG = {
    "level_id": 5,
    "name": "THE ARCHITECT CHAMBER",
    "challenge_description": "Achieve F1 score > 0.88 on the imbalanced dataset.",
    "xp_reward": 400,
    "unlock_reward": ["duel_mode"],
    "default_config": {
        "learning_rate": 0.01, "epochs": 100,
        "hidden_layers": 1, "neurons_per_layer": 16,
        "activation": "relu", "regularization": "none",
        "reg_lambda": 0.0, "batch_size": 32,
    },
}

col_ctrl, col_center, col_right = st.columns([1, 2, 1.5])

with col_ctrl:
    config = render_training_panel(LEVEL_CONFIG, unlocked, LEVEL_ID)
    render_architecture(config, key="arch_l5")

    st.markdown('<div class="train-btn">', unsafe_allow_html=True)
    train_clicked = st.button("▶ TRAIN", use_container_width=True, key="train_l5")
    st.markdown('</div>', unsafe_allow_html=True)

    result = state_manager.get("last_training_result_l5")
    if result:
        render_dna_chart(config, LEVEL_ID)

    def _submit_l5():
        r = state_manager.get("last_training_result_l5")
        run_id = state_manager.get("last_run_id_l5", "")
        if r and run_id:
            from utils.api_client import safe_validate
            ch = safe_validate(session_id, LEVEL_ID, run_id)
            if ch:
                r["challenge_result"] = ch
                state_manager.set("last_training_result_l5", r)
                state_manager.sync_progress_from_api()
                st.rerun()

    ch_result = extract_challenge_result(result) if result else None
    render_challenge_card(LEVEL_CONFIG, ch_result, on_submit=_submit_l5)

if train_clicked:
    result = run_training(session_id, "imbalanced", config, LEVEL_ID, auto_validate=False)
    if result:
        state_manager.set("last_training_result_l5", result)
        state_manager.set("last_run_id_l5", result.get("run_id", ""))
    st.rerun()

result = state_manager.get("last_training_result_l5")
metrics = extract_metrics(result)
failure = extract_failure(result)
cards = extract_explanation_cards(result)

acc = metrics.get("test_accuracy", 0)
f1 = metrics.get("f1_score", 0)
prec = metrics.get("precision", 0)
rec = metrics.get("recall", 0)
cm = metrics.get("confusion_matrix", [[0, 0], [0, 0]])

acc_f1_mismatch = acc > 0.80 and f1 < 0.40 and bool(result)

with col_center:
    # PRIMARY: show BOTH accuracy and F1 prominently
    st.markdown('<div class="section-title">KEY METRICS — F1 IS THE REAL TARGET</div>',
                unsafe_allow_html=True)

    if metrics:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", fmt_pct(acc))
        f1_color = "normal" if f1 > 0.88 else "inverse"
        c2.metric("F1 Score ⭐", fmt_pct(f1))
        c3.metric("Precision", fmt_pct(prec))
        c4.metric("Recall", fmt_pct(rec))

    render_micro_explainer(
        {"title": "Accuracy is Misleading Here",
         "body": "If your model predicts everything as class 0, it gets 85% accuracy but catches zero minority samples. F1 score punishes this: it balances precision and recall so minority performance matters.",
         "action_hint": "Try increasing regularization or neurons to help the minority class."},
        trigger_condition=acc_f1_mismatch,
        explainer_key="acc_f1_mismatch",
    )

    render_loss_curve(extract_history(result), key="loss_l5", show_accuracy=True)

    # Confusion matrix heatmap
    if cm and cm != [[0, 0], [0, 0]]:
        st.markdown('<div class="section-title" style="margin-top:8px;">CONFUSION MATRIX</div>',
                    unsafe_allow_html=True)
        fig_cm = go.Figure(go.Heatmap(
            z=cm,
            x=["Pred 0", "Pred 1"],
            y=["True 0", "True 1"],
            colorscale=[[0, "#111827"], [1, "#8b5cf6"]],
            showscale=False,
            text=[[str(v) for v in row] for row in cm],
            texttemplate="%{text}",
            textfont=dict(size=18, color="#e2e8f0"),
        ))
        fig_cm.update_layout(
            plot_bgcolor="#0a0f1e", paper_bgcolor="#111827",
            margin=dict(l=40, r=20, t=20, b=40), height=200,
            font=dict(color="#e2e8f0"),
            xaxis=dict(side="bottom"),
        )
        st.plotly_chart(fig_cm, use_container_width=True, key="cm_l5")

with col_right:
    st.markdown('<div class="section-title">DECISION BOUNDARY</div>', unsafe_allow_html=True)
    render_boundary(extract_boundary(result), key="bd_l5")
    render_failure_banners(failure, cards)

    # F1 target progress
    if f1:
        target = 0.88
        pct = min(f1 / target * 100, 100)
        color = "#00ff88" if f1 > target else "#ff6b2b"
        st.markdown(f"""
        <div class="forge-card" style="padding:12px;">
          <div class="section-title">F1 PROGRESS TO TARGET</div>
          <div class="xp-bar-track">
            <div style="width:{pct:.0f}%;height:100%;background:{color};
                 border-radius:999px;transition:width 0.8s ease;"></div>
          </div>
          <div style="font-family:monospace;color:{color};text-align:right;margin-top:4px;">
            {fmt_pct(f1)} / 88.0% target
          </div>
        </div>
        """, unsafe_allow_html=True)
