import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.level_helpers import *
from utils.formatters import level_color
from components.comparison_panel import render_comparison_panel
from utils.api_client import train_model, APIError


LEVEL_ID = 6

level_cfg = setup_page(LEVEL_ID)
color = level_color(LEVEL_ID)
xp_bar(sm.get("current_level",1), sm.get("xp",0), sm.get_session_id(), sm.get("insights",[]))
render_sidebar(LEVEL_ID)

st.markdown("""
<div style="text-align:center;padding:24px 0 16px">
  <div style="font-family:var(--display);font-size:36px;font-weight:900;
    color:var(--orange);letter-spacing:6px;text-shadow:0 0 30px rgba(255,107,43,0.5)">
    THE DUEL
  </div>
  <div style="font-size:13px;color:var(--muted);margin-top:6px">
    No hints. Two models. One dataset. Prove your model is superior on two metrics.
  </div>
</div>
""", unsafe_allow_html=True)

# Dataset picker
dataset_options = ["xor", "linear", "noisy", "imbalanced"]
dataset_labels  = {
    "xor": "XOR (hard, non-linear)",
    "linear": "Linear (easy baseline)",
    "noisy": "Noisy (overfitting trap)",
    "imbalanced": "Imbalanced (metric trap)",
}
chosen_dataset = st.selectbox(
    "Choose your arena (dataset)",
    options=dataset_options,
    format_func=lambda x: dataset_labels.get(x, x),
    key="duel_dataset",
)

st.markdown("<hr style='border-color:var(--border);margin:16px 0'>", unsafe_allow_html=True)

# Two-column duel layout
col_you, col_vs, col_opp = st.columns([5, 1, 5])

with col_you:
    st.markdown(
        "<div style='font-family:var(--display);font-size:13px;font-weight:700;"
        "color:var(--cyan);letter-spacing:3px;margin-bottom:12px'>YOUR MODEL</div>",
        unsafe_allow_html=True,
    )
    your_model_cfg, your_train_cfg = render_training_panel(LEVEL_ID, {
        "learning_rate": 0.01, "epochs": 150, "hidden_layers": 2,
        "neurons_per_layer": [16, 16], "activations_per_layer": ["relu", "relu"],
    })
    render_architecture(your_model_cfg, key="arch_you")

with col_vs:
    st.markdown(
        "<div style='font-family:var(--display);font-size:24px;font-weight:900;"
        "color:var(--orange);text-align:center;padding-top:60px;"
        "text-shadow:0 0 20px rgba(255,107,43,0.6)'>VS</div>",
        unsafe_allow_html=True,
    )

with col_opp:
    st.markdown(
        "<div style='font-family:var(--display);font-size:13px;font-weight:700;"
        "color:var(--red);letter-spacing:3px;margin-bottom:12px;text-align:right'>OPPONENT</div>",
        unsafe_allow_html=True,
    )
    st.markdown("""
<div class="forge-card" style="border-color:rgba(255,59,92,0.3)">
  <div style="font-family:var(--mono);font-size:10px;color:var(--red);letter-spacing:1px;margin-bottom:8px">
    THE OVERCONFIDENT PERCEPTRON
  </div>
  <div style="font-size:12px;color:var(--muted);line-height:1.6;margin-bottom:12px">
    Locked configuration. This opponent is deliberately misconfigured — 
    high LR, no hidden layers. Can you beat it?
  </div>
  <div style="font-family:var(--mono);font-size:11px;color:var(--muted)">
    LR: <span style="color:var(--red)">0.1</span><br>
    Epochs: <span style="color:var(--muted)">50</span><br>
    Hidden layers: <span style="color:var(--red)">0</span><br>
    Regularization: <span style="color:var(--muted)">none</span>
  </div>
</div>
""", unsafe_allow_html=True)
    opponent_model_cfg = {
        "hidden_layers": 0, "neurons_per_layer": [], "activations_per_layer": [],
        "activation": "sigmoid", "regularization": "none", "reg_lambda": 0.0, "dropout_rate": 0.0,
    }
    opponent_train_cfg = {
        "learning_rate": 0.1, "epochs": 50, "batch_size": 32,
        "slow_mode": False, "early_stopping": False, "patience": 10,
        "momentum": 0.9, "gradient_clip": 5.0,
    }

st.markdown("<hr style='border-color:var(--border);margin:20px 0'>", unsafe_allow_html=True)

duel_clicked = st.button(
    "⚔ START THE DUEL",
    type="primary",
    use_container_width=True,
    key="duel_start",
)

if duel_clicked:
    col_p1, col_p2 = st.columns(2)
    your_result = None
    opp_result  = None

    with col_p1:
        with st.spinner("Training your model…"):
            try:
                your_result = train_model(
                    session_id=sm.get_session_id() + "_you",
                    dataset=chosen_dataset,
                    model_config=your_model_cfg,
                    train_config=your_train_cfg,
                )
                sm.set("duel_your_result", your_result)
            except APIError as e:
                st.error(f"Your model failed: {e}")

    with col_p2:
        with st.spinner("Training opponent…"):
            try:
                opp_result = train_model(
                    session_id=sm.get_session_id() + "_opp",
                    dataset=chosen_dataset,
                    model_config=opponent_model_cfg,
                    train_config=opponent_train_cfg,
                )
                sm.set("duel_opp_result", opp_result)
            except APIError as e:
                st.error(f"Opponent model failed: {e}")

    if your_result and opp_result:
        sm.sync_progress_from_api()
        st.rerun()

# Results
your_result = sm.get("duel_your_result")
opp_result  = sm.get("duel_opp_result")

if your_result and opp_result:
    my = your_result.get("final_metrics", {})
    op = opp_result.get("final_metrics", {})
    acc_win = my.get("accuracy", 0) > op.get("accuracy", 0)
    f1_win  = my.get("f1_score", 0) > op.get("f1_score", 0)
    wins    = int(acc_win) + int(f1_win)

    if wins >= 2:
        st.markdown("""
<div style="text-align:center;padding:20px;background:rgba(0,255,136,0.08);
  border:2px solid var(--green);border-radius:12px;margin-bottom:20px;
  animation:level-up-flash 0.5s ease">
  <div style="font-family:var(--display);font-size:32px;font-weight:900;
    color:var(--green);letter-spacing:4px">VICTORY</div>
  <div style="font-size:14px;color:var(--text);margin-top:8px">
    You outperformed the opponent on accuracy AND F1. 🏆 MASTER BADGE earned.
  </div>
</div>
""", unsafe_allow_html=True)
        if sm.add_insight("duel_won"):
            st.toast("🏆 MASTER BADGE: You Are The Architect", icon="🏆")
    else:
        st.markdown(f"""
<div style="text-align:center;padding:20px;background:rgba(255,59,92,0.08);
  border:2px solid var(--red);border-radius:12px;margin-bottom:20px">
  <div style="font-family:var(--display);font-size:32px;font-weight:900;
    color:var(--red);letter-spacing:4px">DEFEATED</div>
  <div style="font-size:14px;color:var(--text);margin-top:8px">
    Won {wins}/2 required metrics. Refine your architecture and try again.
  </div>
</div>
""", unsafe_allow_html=True)

    # HP bars
    your_acc = my.get("accuracy", 0)
    opp_acc  = op.get("accuracy", 0)
    your_f1  = my.get("f1_score", 0)
    opp_f1   = op.get("f1_score", 0)

    c1, c2, c3 = st.columns([5, 1, 5])
    with c1:
        st.markdown(f"""
<div style="font-family:var(--mono);font-size:9px;color:var(--muted);margin-bottom:3px">ACCURACY</div>
<div class="hp-track"><div class="hp-fill-you" style="width:{int(your_acc*100)}%"></div></div>
<div style="font-family:var(--mono);font-size:11px;color:var(--cyan);margin-bottom:8px">{your_acc:.3f}</div>
<div style="font-family:var(--mono);font-size:9px;color:var(--muted);margin-bottom:3px">F1 SCORE</div>
<div class="hp-track"><div class="hp-fill-you" style="width:{int(your_f1*100)}%"></div></div>
<div style="font-family:var(--mono);font-size:11px;color:var(--cyan)">{your_f1:.3f}</div>
""", unsafe_allow_html=True)
    with c2:
        st.markdown("<div style='text-align:center;font-family:var(--display);font-size:16px;color:var(--orange);padding-top:20px'>VS</div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
<div style="font-family:var(--mono);font-size:9px;color:var(--muted);margin-bottom:3px;text-align:right">ACCURACY</div>
<div class="hp-track"><div class="hp-fill-opp" style="width:{int(opp_acc*100)}%"></div></div>
<div style="font-family:var(--mono);font-size:11px;color:var(--red);margin-bottom:8px;text-align:right">{opp_acc:.3f}</div>
<div style="font-family:var(--mono);font-size:9px;color:var(--muted);margin-bottom:3px;text-align:right">F1 SCORE</div>
<div class="hp-track"><div class="hp-fill-opp" style="width:{int(opp_f1*100)}%"></div></div>
<div style="font-family:var(--mono);font-size:11px;color:var(--red);text-align:right">{opp_f1:.3f}</div>
""", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:var(--border);margin:20px 0'>", unsafe_allow_html=True)

    render_comparison_panel(
        your_result, opp_result,
        label_a="YOUR MODEL", label_b="OPPONENT",
        archetype_a="The Architect", archetype_b="The Overconfident Perceptron",
    )

    # Submit for challenge
    run_id = your_result.get("run_id", "")
    your_result["duel_wins"] = wins
    if run_id:
        from utils.api_client import validate_challenge
        if st.button("⚡ Submit Duel Result", type="primary", key="submit_duel"):
            try:
                cres = validate_challenge(sm.get_session_id(), LEVEL_ID, run_id)
                if cres.get("passed"):
                    st.success(cres.get("message", "Duel won!"))
                    sm.sync_progress_from_api()
                    st.balloons()
                else:
                    st.warning(cres.get("message", "Not yet."))
            except APIError as e:
                st.error(str(e))
