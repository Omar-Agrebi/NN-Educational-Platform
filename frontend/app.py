import streamlit as st

st.set_page_config(
    page_title="NEURAL FORGE",
    page_icon="⚙",
    layout="wide",
    initial_sidebar_state="collapsed",
)

import utils.state_manager as state_manager
from styles.theme import inject_theme

inject_theme()
state_manager.init_state()
state_manager.sync_progress_from_api()

# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:12px 0;border-bottom:1px solid #1f2d48;margin-bottom:12px;">
      <span style="font-weight:900;color:#00f5ff;letter-spacing:0.15em;font-size:0.9rem;">
        ⚙ NEURAL FORGE
      </span>
    </div>
    """, unsafe_allow_html=True)

    st.page_link("pages/00_forge_gate.py",   label="🗺 Forge Gate",         icon=None)
    st.page_link("pages/01_perceptron_forge.py", label="LVL 1 — Perceptron Forge")
    st.page_link("pages/02_xor_crucible.py", label="LVL 2 — XOR Crucible")
    st.page_link("pages/03_depth_trials.py", label="LVL 3 — Depth Trials")
    st.page_link("pages/04_overfit_arena.py",label="LVL 4 — Overfit Arena")
    st.page_link("pages/05_architect_chamber.py", label="LVL 5 — Architect Chamber")
    st.page_link("pages/06_duel_mode.py",    label="LVL 6 — The Duel")

    if state_manager.is_feature_unlocked("replay_mode"):
        st.markdown("---")
        st.page_link("pages/06_duel_mode.py", label="⏪ REPLAY MODE")

    # Insights codex
    insights = state_manager.get("insights", [])
    if insights:
        st.markdown("---")
        st.markdown('<div class="section-title">🪙 INSIGHTS CODEX</div>', unsafe_allow_html=True)
        for ins in insights:
            st.markdown(f'<span class="insight-tag">{ins["label"]}</span>', unsafe_allow_html=True)

# ── Redirect to Forge Gate ────────────────────────────────────────────────────
st.switch_page("pages/00_forge_gate.py")
