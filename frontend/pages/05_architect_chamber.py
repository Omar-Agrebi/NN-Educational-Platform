import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.level_helpers import *
from utils.formatters import level_color

LEVEL_ID = 5
DATASET  = "imbalanced"

level_cfg = setup_page(LEVEL_ID)
color = level_color(LEVEL_ID)
xp_bar(sm.get("current_level",1), sm.get("xp",0), sm.get_session_id(), sm.get("insights",[]))
render_sidebar(LEVEL_ID)

level_header(LEVEL_ID, "THE ARCHITECT CHAMBER",
             "85% accuracy. 0% recall. Accuracy is lying to you — use F1.", color)

tab_train, tab_lr, tab_replay = st.tabs(["⚡ Training", "🔍 LR Finder", "⏪ Replay"])

with tab_train:
    col_ctrl, col_center, col_right = st.columns([1, 1.6, 1.2])

    with col_ctrl:
        model_config, train_config = render_training_panel(LEVEL_ID, level_cfg["default_config"])
        render_architecture(model_config, key="arch_l5")
        render_parameter_dna(model_config, train_config, color)
        st.markdown("<br>", unsafe_allow_html=True)
        train_clicked = st.button("▶ TRAIN MODEL", type="primary", use_container_width=True, key="train_l5")

    result = sm.get("last_training_result")

    with col_center:
        card_title("Decision Boundary")
        boundary = result.get("boundary_data") if result else None
        render_boundary(boundary, title="", height=280, key="bd_l5")

        # Accuracy vs F1 lesson block
        if result:
            m = result.get("final_metrics", {})
            acc = m.get("accuracy", 0)
            f1  = m.get("f1_score", 0)
            if acc > 0.80 and f1 < 0.5:
                st.markdown("""
<div class="failure-banner" style="margin-top:10px">
  <div style="font-size:16px">⚠</div>
  <div>
    <div style="font-size:13px;font-weight:700;color:var(--red);margin-bottom:3px">Accuracy Is Deceiving You</div>
    <div style="font-size:11px;color:var(--muted);line-height:1.5">
      High accuracy on imbalanced data often means the model predicts only the majority class. 
      F1 Score reveals the truth — it requires both precision and recall to be high.
    </div>
    <div style="font-size:11px;color:var(--cyan);font-style:italic;margin-top:4px">
      → Check the confusion matrix — look at recall for Class 1
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    with col_right:
        if result:
            # F1 is primary metric here — show prominently
            m = result.get("final_metrics", {})
            f1  = m.get("f1_score", 0)
            acc = m.get("accuracy", 0)
            f1_color = "var(--green)" if f1 > 0.80 else "var(--orange)" if f1 > 0.5 else "var(--red)"

            st.markdown(f"""
<div style="background:var(--card2);border:2px solid {f1_color};border-radius:10px;
  padding:16px;text-align:center;margin-bottom:12px">
  <div style="font-family:var(--mono);font-size:9px;color:var(--muted);margin-bottom:4px">PRIMARY METRIC</div>
  <div style="font-family:var(--display);font-size:36px;font-weight:900;color:{f1_color}">{f1:.3f}</div>
  <div style="font-family:var(--mono);font-size:10px;color:var(--muted)">F1 SCORE · Target: &gt;0.88</div>
</div>
<div style="font-family:var(--mono);font-size:11px;color:var(--muted);
  background:var(--card);border-radius:6px;padding:8px 12px;margin-bottom:12px">
  Accuracy: <span style="color:var(--cyan)">{acc:.1%}</span>
  <span style="color:var(--muted);margin:0 6px">·</span>
  F1: <span style="color:{f1_color}">{f1:.3f}</span>
</div>
""", unsafe_allow_html=True)

        render_loss_curve(result.get("history", {}) if result else {}, key="loss_l5", show_accuracy=True, height=180)
        render_failure_banners(
            result.get("failure_report", {}) if result else {},
            result.get("explanation_cards", []) if result else [],
        )
        render_challenge_card(level_cfg, result)

    if train_clicked:
        result = do_training(DATASET, model_config, train_config)
        if result:
            st.rerun()

    # Confusion matrix below
    if result and result.get("confusion_matrix"):
        st.markdown("<hr style='border-color:var(--border);margin:16px 0'>", unsafe_allow_html=True)
        col_cm, col_wi = st.columns(2)
        with col_cm:
            render_confusion_matrix(result["confusion_matrix"])
        with col_wi:
            if result.get("weight_snapshots"):
                render_weight_inspector(result.get("weight_snapshots", []), model_config)

with tab_lr:
    render_lr_finder(DATASET, model_config if 'model_config' in dir() else {
        "hidden_layers":1,"neurons_per_layer":[8],"activations_per_layer":["relu"],
        "activation":"relu","regularization":"none","reg_lambda":0.0,
    })

with tab_replay:
    try:
        from utils.api_client import get_replay
        render_replay_player(get_replay(sm.get_session_id()))
    except Exception:
        st.info("Run training first to enable replay.")
