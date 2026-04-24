import streamlit as st
import utils.state_manager as state_manager
from styles.theme import inject_theme
from components.xp_bar import render_xp_bar
from components.training_panel import render_training_panel, render_dna_chart
from components.architecture_viz import render_architecture
from components.comparison_panel import render_comparison_panel
from components.replay_player import render_replay_player
from utils.training_helper import extract_metrics
from utils.formatters import fmt_pct, level_color
from utils.api_client import safe_train, safe_validate, get_replay

inject_theme()
state_manager.init_state()
state_manager.sync_progress_from_api()

LEVEL_ID = 6
COLOR = level_color(LEVEL_ID)
session_id = state_manager.get_session_id()
unlocked = state_manager.get("unlocked_features", [])
xp = state_manager.get("xp", 0)

render_xp_bar(xp, state_manager.get("current_level", 1), unlocked, session_id)

if not state_manager.is_level_unlocked(LEVEL_ID):
    st.warning("🔒 Complete Level 5 — The Architect Chamber to unlock The Duel.")
    st.stop()

# ── Dramatic header ───────────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center;padding:20px 0 12px;">
  <div style="font-size:2.5rem;font-weight:900;letter-spacing:0.2em;color:#f59e0b;
       text-shadow:0 0 24px rgba(245,158,11,0.6);">
    ⚔ THE DUEL
  </div>
  <div style="color:#4a5568;font-size:0.82rem;letter-spacing:0.2em;margin-top:4px;">
    FINAL FORGE — DEFEAT THE OPPONENT ON ACCURACY AND F1
  </div>
</div>
""", unsafe_allow_html=True)

LEVEL_CONFIG = {
    "level_id": 6,
    "name": "THE DUEL",
    "challenge_description": "Beat the opponent model on both accuracy AND F1 score.",
    "xp_reward": 500,
    "unlock_reward": ["master_badge", "full_forge", "replay_mode"],
    "default_config": {
        "learning_rate": 0.01, "epochs": 100,
        "hidden_layers": 2, "neurons_per_layer": 16,
        "activation": "relu", "regularization": "l2",
        "reg_lambda": 0.001, "batch_size": 32,
    },
}

OPPONENT_CONFIG = {
    "model_cfg": {
        "hidden_layers": 0, "neurons_per_layer": 2,
        "activation": "sigmoid", "regularization": "none", "reg_lambda": 0.0,
    },
    "train_cfg": {
        "learning_rate": 10.0, "epochs": 5, "batch_size": 256, "slow_mode": False,
    },
}

tab_duel, tab_replay = st.tabs(["⚔ DUEL", "⏪ REPLAY"])

with tab_duel:
    # Dataset selector
    dataset = st.selectbox(
        "Choose your battlefield",
        ["xor", "noisy", "imbalanced", "linear"],
        format_func=lambda d: {"xor": "⊕ XOR", "noisy": "🌪️ Noisy",
                               "imbalanced": "⚖ Imbalanced", "linear": "📏 Linear"}[d],
        key="duel_dataset",
    )

    col_player, col_vs, col_opponent = st.columns([5, 1, 5])

    with col_player:
        st.markdown("""
        <div style="text-align:center;margin-bottom:10px;">
          <span class="level-badge" style="color:#00f5ff;border-color:#00f5ff;font-size:1rem;">
            🧑 YOUR MODEL
          </span>
        </div>
        """, unsafe_allow_html=True)
        player_config = render_training_panel(LEVEL_CONFIG, unlocked, LEVEL_ID)
        render_architecture(player_config, key="arch_player")

    with col_vs:
        st.markdown("""
        <div style="display:flex;flex-direction:column;align-items:center;
             justify-content:center;height:100%;padding-top:60px;">
          <div class="vs-text">VS</div>
        </div>
        """, unsafe_allow_html=True)

    with col_opponent:
        st.markdown("""
        <div style="text-align:center;margin-bottom:10px;">
          <span class="level-badge" style="color:#ff6b2b;border-color:#ff6b2b;font-size:1rem;">
            🤖 OPPONENT
          </span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="locked-card" style="opacity:0.7;">
          <div class="section-title">OPPONENT CONFIG (LOCKED)</div>
          <div style="font-family:monospace;font-size:0.78rem;color:#4a5568;line-height:2;">
            Hidden Layers: 0<br>
            Learning Rate: 10.0 (too high)<br>
            Epochs: 5 (too few)<br>
            Batch Size: 256 (too large)<br>
            <span style="color:#ff3b5c;">⚠ Intentionally misconfigured</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        opp_result = state_manager.get("duel_opponent_result")
        if opp_result:
            opp_metrics = extract_metrics(opp_result)
            st.markdown(f"""
            <div class="forge-card" style="padding:10px;border-color:#ff6b2b;">
              <div class="section-title">OPPONENT RESULTS</div>
              <div style="font-family:monospace;color:#ff6b2b;">
                Acc: {fmt_pct(opp_metrics.get("test_accuracy",0))}
                &nbsp;|&nbsp;
                F1: {fmt_pct(opp_metrics.get("f1_score",0))}
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ── BATTLE button ─────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    battle_col = st.columns([1, 2, 1])[1]
    with battle_col:
        battle_clicked = st.button(
            "⚔ INITIATE DUEL", use_container_width=True, key="duel_battle_btn"
        )

    if battle_clicked:
        # Train player model
        with st.spinner("⚙ Your model is forging…"):
            player_result = safe_train(
                session_id, dataset,
                {
                    "hidden_layers": player_config.get("hidden_layers", 2),
                    "neurons_per_layer": player_config.get("neurons_per_layer", 16),
                    "activation": player_config.get("activation", "relu"),
                    "regularization": player_config.get("regularization", "none"),
                    "reg_lambda": player_config.get("reg_lambda", 0.001),
                },
                {
                    "learning_rate": player_config.get("learning_rate", 0.01),
                    "epochs": player_config.get("epochs", 100),
                    "batch_size": player_config.get("batch_size", 32),
                    "slow_mode": False,
                },
            )

        # Train opponent model
        with st.spinner("🤖 Opponent model training…"):
            opp_result = safe_train(
                session_id + "_opponent", dataset,
                OPPONENT_CONFIG["model_cfg"],
                OPPONENT_CONFIG["train_cfg"],
            )

        if player_result:
            state_manager.set("duel_player_result", player_result)
        if opp_result:
            state_manager.set("duel_opponent_result", opp_result)

        # Check win condition
        if player_result and opp_result:
            p_acc = player_result.get("final_metrics", {}).get("test_accuracy", 0)
            p_f1  = player_result.get("final_metrics", {}).get("f1_score", 0)
            o_acc = opp_result.get("final_metrics", {}).get("test_accuracy", 0)
            o_f1  = opp_result.get("final_metrics", {}).get("f1_score", 0)

            wins_both = p_acc > o_acc and p_f1 > o_f1
            if wins_both:
                state_manager.add_insight("duel_win", "You Are The Architect")
                # Validate challenge
                run_id = player_result.get("run_id", "")
                if run_id:
                    ch = safe_validate(session_id, LEVEL_ID, run_id)
                    if ch:
                        player_result["challenge_result"] = ch
                        state_manager.set("duel_player_result", player_result)
                        state_manager.sync_progress_from_api()
                        if ch.get("is_gate_passed"):
                            st.balloons()

        st.rerun()

    # ── Results ───────────────────────────────────────────────────────────────
    player_result = state_manager.get("duel_player_result")
    opp_result = state_manager.get("duel_opponent_result")

    if player_result and opp_result:
        st.markdown("---")
        render_comparison_panel(
            player_result, opp_result,
            label_a="Your Model", label_b="Opponent",
            is_duel=True,
        )

        # Challenge card
        ch_result = player_result.get("challenge_result")
        if ch_result:
            if ch_result.get("is_gate_passed"):
                st.markdown("""
                <div style="text-align:center;padding:24px;background:rgba(0,255,136,0.07);
                     border:2px solid #00ff88;border-radius:16px;animation:level-up-flash 0.8s ease-out;
                     margin-top:16px;">
                  <div style="font-size:2rem;font-weight:900;color:#f59e0b;letter-spacing:0.1em;">
                    🏆 MASTER BADGE EARNED
                  </div>
                  <div style="color:#00ff88;margin-top:8px;font-weight:700;">
                    You have mastered the Neural Forge.
                  </div>
                  <div style="color:#4a5568;font-size:0.82rem;margin-top:4px;">
                    Replay mode unlocked — review every epoch of your journey.
                  </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                p_acc = player_result.get("final_metrics", {}).get("test_accuracy", 0)
                p_f1  = player_result.get("final_metrics", {}).get("f1_score", 0)
                o_acc = opp_result.get("final_metrics", {}).get("test_accuracy", 0)
                o_f1  = opp_result.get("final_metrics", {}).get("f1_score", 0)
                hint = []
                if p_acc <= o_acc:
                    hint.append("Improve accuracy: add hidden layers or train longer.")
                if p_f1 <= o_f1:
                    hint.append("Improve F1: tune regularization and increase neurons.")
                st.markdown(f"""
                <div class="failure-banner" style="margin-top:16px;">
                  <b>⚠ Not yet victorious.</b> You need to win on BOTH metrics.<br>
                  <span style="color:#00f5ff;font-size:0.82rem;">
                    {'  '.join(hint)}
                  </span>
                </div>
                """, unsafe_allow_html=True)

with tab_replay:
    if not state_manager.is_feature_unlocked("replay_mode"):
        st.markdown("""
        <div class="locked-card" style="text-align:center;padding:40px;">
          <div style="font-size:2rem;">🔒</div>
          <div style="color:#4a5568;margin-top:8px;">Win The Duel to unlock Replay Mode.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="section-title">TRAINING REPLAY</div>', unsafe_allow_html=True)
        try:
            replay_raw = get_replay(session_id)
            if replay_raw:
                render_replay_player(replay_raw)
            else:
                st.info("No replay data found. Complete a training run first.")
        except Exception as e:
            st.error(f"Could not load replay: {e}")
