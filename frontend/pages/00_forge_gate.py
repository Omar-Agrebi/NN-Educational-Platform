import streamlit as st
import utils.state_manager as state_manager
from styles.theme import inject_theme
from components.xp_bar import render_xp_bar
from utils.formatters import level_color, dataset_emoji

inject_theme()
state_manager.init_state()
state_manager.sync_progress_from_api()

session_id = state_manager.get_session_id()
xp = state_manager.get("xp", 0)
current_level = state_manager.get("current_level", 1)
unlocked = state_manager.get("unlocked_features", [])
completed = state_manager.get("completed_challenges", [])

render_xp_bar(xp, current_level, unlocked, session_id)

# ── Onboarding intro ──────────────────────────────────────────────────────────
if not state_manager.get("onboarding_done", False):
    st.markdown("""
    <div style="text-align:center;padding:60px 20px 40px 20px;">
      <div style="font-size:3.5rem;font-weight:900;letter-spacing:0.2em;
           color:#00f5ff;animation:glitch 3s ease-in-out infinite;margin-bottom:16px;">
        ⚙ NEURAL FORGE
      </div>
      <div style="max-width:600px;margin:0 auto;font-size:1.05rem;color:#8892a4;
           line-height:1.8;margin-bottom:32px;">
        You don't <em>read</em> about neural networks here.<br>
        You <strong style="color:#ff6b2b;">break</strong> them.
        <strong style="color:#8b5cf6;">Fix</strong> them.
        <strong style="color:#00ff88;">Master</strong> them.
      </div>

      <!-- Animated hex grid background -->
      <div style="position:relative;margin-bottom:32px;">
        <div style="font-size:4rem;animation:hex-drift 4s ease-in-out infinite;opacity:0.15;
             position:absolute;top:-20px;left:10%;pointer-events:none;">⬡</div>
        <div style="font-size:2rem;animation:hex-drift 6s ease-in-out infinite 1s;opacity:0.1;
             position:absolute;top:10px;right:15%;pointer-events:none;">⬡</div>
        <div style="font-size:6rem;animation:hex-drift 5s ease-in-out infinite 0.5s;opacity:0.06;
             position:absolute;top:-30px;left:40%;pointer-events:none;">⬡</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col = st.columns([1, 2, 1])[1]
    with col:
        if st.button("BEGIN FORGING  ▶", use_container_width=True, key="begin_btn"):
            state_manager.set("onboarding_done", True)
            st.rerun()
    st.stop()

# ── World Map ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:24px 0 12px 0;">
  <div style="font-size:2rem;font-weight:900;letter-spacing:0.2em;color:#00f5ff;">
    ⚙ NEURAL FORGE
  </div>
  <div style="color:#4a5568;font-size:0.8rem;letter-spacing:0.2em;margin-top:4px;">
    WORLD MAP — SELECT YOUR LEVEL
  </div>
</div>
""", unsafe_allow_html=True)

LEVELS = [
    {
        "id": 1, "name": "THE PERCEPTRON FORGE", "dataset": "linear",
        "challenge": "Achieve train accuracy > 85%",
        "xp": 100, "page": "pages/01_perceptron_forge.py",
        "desc": "Master the simplest neural network.",
    },
    {
        "id": 2, "name": "THE XOR CRUCIBLE", "dataset": "xor",
        "challenge": "Achieve test accuracy > 90%",
        "xp": 200, "page": "pages/02_xor_crucible.py",
        "desc": "Break the perceptron. Build your first MLP.",
    },
    {
        "id": 3, "name": "DEPTH TRIALS", "dataset": "xor",
        "challenge": "Gain >10% accuracy over baseline",
        "xp": 200, "page": "pages/03_depth_trials.py",
        "desc": "Discover how depth changes everything.",
    },
    {
        "id": 4, "name": "THE OVERFIT ARENA", "dataset": "noisy",
        "challenge": "Keep train/test gap < 5%",
        "xp": 300, "page": "pages/04_overfit_arena.py",
        "desc": "Watch your model memorize noise. Then stop it.",
    },
    {
        "id": 5, "name": "THE ARCHITECT CHAMBER", "dataset": "imbalanced",
        "challenge": "Achieve F1 score > 0.88",
        "xp": 400, "page": "pages/05_architect_chamber.py",
        "desc": "Accuracy lies. Build for F1.",
    },
    {
        "id": 6, "name": "THE DUEL", "dataset": "player_choice",
        "challenge": "Beat the opponent on accuracy AND F1",
        "xp": 500, "page": "pages/06_duel_mode.py",
        "desc": "Face the intentionally broken opponent. Win.",
    },
]

cols_row1 = st.columns(3)
cols_row2 = st.columns(3)
all_cols = cols_row1 + cols_row2

for i, lvl in enumerate(LEVELS):
    lid = lvl["id"]
    color = level_color(lid)
    is_unlocked = state_manager.is_level_unlocked(lid)
    is_done = lid in completed

    with all_cols[i]:
        if is_unlocked:
            glow = f"box-shadow:0 0 14px {color}44;border:1px solid {color};" if not is_done else "border:1px solid #00ff88;box-shadow:0 0 10px rgba(0,255,136,0.2);"
            border_color = "#00ff88" if is_done else color
            done_badge = '<span style="color:#00ff88;font-weight:800;">✓ COMPLETE</span>' if is_done else ""

            st.markdown(f"""
            <div class="forge-card" style="{glow}border-radius:12px;min-height:180px;
                 animation:unlock-pulse 3s ease-in-out infinite;">
              <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                <span class="level-badge" style="color:{border_color};border-color:{border_color};">
                  LVL {lid}
                </span>
                <span style="color:#4a5568;font-size:1.2rem;">{dataset_emoji(lvl['dataset'])}</span>
              </div>
              <div style="font-weight:800;color:{border_color};margin:8px 0 4px;
                   font-size:0.82rem;letter-spacing:0.05em;">
                {lvl['name']}
              </div>
              <div style="color:#8892a4;font-size:0.78rem;margin-bottom:8px;">{lvl['desc']}</div>
              <div style="color:#4a5568;font-size:0.72rem;margin-bottom:4px;">
                📋 {lvl['challenge']}
              </div>
              <div style="color:{color};font-family:monospace;font-size:0.72rem;margin-bottom:8px;">
                ⚡ {lvl['xp']} XP
              </div>
              {done_badge}
            </div>
            """, unsafe_allow_html=True)

            if st.button(f"▶ ENTER", key=f"enter_lvl_{lid}", use_container_width=True):
                st.switch_page(lvl["page"])
        else:
            st.markdown(f"""
            <div class="locked-card" style="min-height:180px;">
              <span class="level-badge" style="color:#4a5568;border-color:#4a5568;">
                LVL {lid}
              </span>
              <div style="font-weight:800;color:#4a5568;margin:8px 0 4px;
                   font-size:0.82rem;letter-spacing:0.05em;">
                {lvl['name']}
              </div>
              <div style="color:#2d3748;font-size:0.78rem;margin-bottom:8px;">{lvl['desc']}</div>
              <div style="color:#2d3748;font-size:0.72rem;">📋 {lvl['challenge']}</div>
              <div style="color:#2d3748;font-family:monospace;font-size:0.72rem;margin-top:4px;">
                ⚡ {lvl['xp']} XP
              </div>
            </div>
            """, unsafe_allow_html=True)
            st.button("🔒 LOCKED", key=f"locked_{lid}", disabled=True, use_container_width=True)

# ── Codex ─────────────────────────────────────────────────────────────────────
insights = state_manager.get("insights", [])
if insights:
    st.markdown("---")
    st.markdown('<div class="section-title">🪙 INSIGHTS CODEX</div>', unsafe_allow_html=True)
    cols = st.columns(min(len(insights), 4))
    for i, ins in enumerate(insights):
        with cols[i % 4]:
            st.markdown(f'<span class="insight-tag">{ins["label"]}</span>', unsafe_allow_html=True)

# ── Dev reset ─────────────────────────────────────────────────────────────────
with st.expander("⚙ Dev Tools", expanded=False):
    if st.button("Reset All Progress", key="dev_reset"):
        from utils.api_client import reset_progress
        reset_progress(session_id)
        state_manager.sync_progress_from_api()
        state_manager.set("onboarding_done", False)
        st.rerun()
