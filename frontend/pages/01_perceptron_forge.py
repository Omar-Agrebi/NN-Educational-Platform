import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.level_helpers import *
from utils.formatters import level_color

LEVEL_ID = 1
DATASET  = "linear"

level_cfg = setup_page(LEVEL_ID)
color = level_color(LEVEL_ID)
xp_bar(sm.get("current_level",1), sm.get("xp",0), sm.get_session_id(), sm.get("insights",[]))
render_sidebar(LEVEL_ID)

level_header(LEVEL_ID, "THE PERCEPTRON FORGE",
             "Can a single neuron separate the world? Train your first perceptron.", color)

tab_train, tab_experiment, tab_replay = st.tabs(["⚡ Training", "🔬 LR Finder", "⏪ Replay"])

with tab_train:
    col_ctrl, col_center, col_right = st.columns([1, 1.6, 1.2])

    with col_ctrl:
        model_config, train_config = render_training_panel(LEVEL_ID, level_cfg["default_config"])
        st.markdown("<br>", unsafe_allow_html=True)
        render_architecture(model_config, key="arch_l1")
        render_parameter_dna(model_config, train_config, color)
        st.markdown("<br>", unsafe_allow_html=True)
        train_clicked = st.button("▶ TRAIN MODEL", type="primary", use_container_width=True, key="train_l1")

    result = sm.get("last_training_result")

    with col_center:
        card_title("Decision Boundary")
        boundary = result.get("boundary_data") if result else None
        render_boundary(boundary, title="", height=320, key="bd_l1")

    with col_right:
        render_loss_curve(result.get("history", {}) if result else {}, key="loss_l1")
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

    # Weight inspector below
    if result and result.get("weight_snapshots"):
        with st.expander("🔬 Weight Inspector", expanded=False):
            render_weight_inspector(result.get("weight_snapshots", []), model_config)

with tab_experiment:
    render_lr_finder(DATASET, model_config if 'model_config' in dir() else {"hidden_layers":0,"neurons_per_layer":[],"activations_per_layer":[],"activation":"sigmoid","regularization":"none","reg_lambda":0.0})

with tab_replay:
    try:
        from utils.api_client import get_replay
        replay_data = get_replay(sm.get_session_id())
        render_replay_player(replay_data)
    except Exception as e:
        st.info("Run training first to enable replay.")
