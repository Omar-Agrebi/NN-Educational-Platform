import streamlit as st

st.set_page_config(
    page_title="NEURAL FORGE",
    page_icon="⚙",
    layout="wide",
    initial_sidebar_state="expanded",
)

from styles.theme import inject_theme, xp_bar, level_header
from utils import state_manager as sm
from utils.api_client import health_check, APIError
from utils.formatters import level_color

inject_theme()
sm.init_state()
sm.sync_progress_from_api()

# ── XP BAR (always visible, below Streamlit header) ────────
xp_bar(
    level_id=sm.get("current_level", 1),
    xp=sm.get("xp", 0),
    session_id=sm.get_session_id(),
    insights=sm.get("insights", []),
)

# ── Onboarding ──────────────────────────────────────────────
if not sm.get("onboarding_done"):
    st.markdown("""
<div style="text-align:center;padding:60px 20px 40px">
  <div style="font-family:var(--display);font-size:48px;font-weight:900;
    color:var(--cyan);letter-spacing:6px;text-shadow:0 0 40px rgba(0,245,255,0.4);
    margin-bottom:16px">
    NEURAL<span style="color:var(--orange)">FORGE</span>
  </div>
  <div style="font-size:16px;color:var(--muted);max-width:520px;margin:0 auto 32px;
    line-height:1.7;font-weight:300">
    You don't read about neural networks here.<br>
    You <strong style="color:var(--cyan)">break</strong> them.
    <strong style="color:var(--orange)">Fix</strong> them.
    <strong style="color:var(--green)">Master</strong> them.
  </div>
</div>
""", unsafe_allow_html=True)

    col = st.columns([1, 2, 1])[1]
    with col:
        if st.button("⚙ BEGIN FORGING ▶", type="primary", use_container_width=True):
            sm.set("onboarding_done", True)
            st.rerun()
    st.stop()

# ── Backend health check ────────────────────────────────────
if not health_check():
    st.error("⚠ Cannot connect to backend at http://localhost:8000 — start it first: `uvicorn main:app --reload --port 8000`")
    st.stop()

# ── World Map ───────────────────────────────────────────────
level_header(0, "THE FORGE GATE", "Choose your level. Complete challenges to unlock the next forge.", "#00f5ff")

LEVEL_META = [
    (1, "THE PERCEPTRON FORGE", "Linear 2D", "Achieve >85% test accuracy", 100, "01_perceptron_forge"),
    (2, "THE XOR CRUCIBLE",     "XOR Pattern", "Solve XOR with >90% accuracy", 200, "02_xor_crucible"),
    (3, "DEPTH TRIALS",         "XOR + Linear", "Show >10% accuracy gain by depth", 200, "03_depth_trials"),
    (4, "THE OVERFIT ARENA",    "Noisy Data",  "Reduce train/test gap to <5%", 300, "04_overfit_arena"),
    (5, "THE ARCHITECT CHAMBER","Imbalanced",  "Achieve >88% F1 Score", 400, "05_architect_chamber"),
    (6, "THE DUEL",             "Your Choice", "Beat opponent on 2 metrics", 500, "06_duel_mode"),
]

cols = st.columns(3)
for idx, (lvl, name, dataset, challenge, xp, page) in enumerate(LEVEL_META):
    col = cols[idx % 3]
    with col:
        unlocked  = sm.is_level_unlocked(lvl)
        completed = sm.is_challenge_completed(lvl)
        color     = level_color(lvl)

        status_html = (
            f"<span style='color:var(--green);font-family:var(--mono);font-size:9px'>✓ COMPLETE</span>"
            if completed else
            f"<span style='color:{color};font-family:var(--mono);font-size:9px'>▶ ACTIVE</span>"
            if unlocked else
            "<span style='color:var(--muted);font-family:var(--mono);font-size:9px'>🔒 LOCKED</span>"
        )
        card_class = "done" if completed else "unlocked" if unlocked else "locked"
        lock_icon  = "" if unlocked else "<div style='position:absolute;top:16px;right:16px;font-size:18px;opacity:0.3'>🔒</div>"

        st.markdown(f"""
<div class="nf-world-card {card_class}" style="border-color:{'rgba(0,255,136,0.35)' if completed else f'{color}55' if unlocked else 'var(--border)'}">
  {lock_icon}
  <div style="font-family:var(--mono);font-size:9px;color:var(--muted);letter-spacing:2px;margin-bottom:6px">LEVEL 0{lvl}</div>
  <div style="font-family:var(--display);font-size:13px;font-weight:700;color:#fff;margin-bottom:6px;letter-spacing:1px">{name}</div>
  <div style="font-size:11px;color:var(--muted);margin-bottom:8px">Dataset: {dataset}</div>
  <div style="font-size:11px;color:var(--text);background:var(--card2);border-radius:4px;
    padding:5px 8px;margin-bottom:12px;line-height:1.4">{challenge}</div>
  <div style="display:flex;align-items:center;justify-content:space-between">
    <span style="font-family:var(--mono);font-size:10px;color:var(--orange)">+{xp} XP</span>
    {status_html}
  </div>
</div>
""", unsafe_allow_html=True)

        if unlocked:
            if st.button(f"{'Replay' if completed else 'Enter'} Level {lvl}", key=f"enter_lvl_{lvl}", use_container_width=True):
                st.switch_page(f"pages/{page}.py")

# ── Custom Forge entry ──────────────────────────────────────────────────────
st.markdown("<hr style='border-color:var(--border);margin:24px 0 16px'>", unsafe_allow_html=True)
st.markdown(
    "<div style='font-family:var(--display);font-size:12px;color:var(--orange);"
    "letter-spacing:3px;margin-bottom:12px'>⚙ CUSTOM FORGE</div>",
    unsafe_allow_html=True,
)
col_cf1, col_cf2 = st.columns([3, 1])
with col_cf1:
    st.markdown(
        "<div style='font-size:13px;color:var(--text);line-height:1.6'>"
        "Bring your own dataset. Upload CSV or XLSX, auto-detect the problem type "
        "(binary classification, multi-class, or regression), select your model, "
        "and get real results on real data."
        "</div>",
        unsafe_allow_html=True,
    )
with col_cf2:
    if st.button("⚙ Open Custom Forge", use_container_width=True, key="open_custom"):
        st.switch_page("pages/07_custom_forge.py")

# ── Insights Codex ──────────────────────────────────────────
insights = sm.get("insights", [])
if insights:
    st.markdown("<hr style='border-color:var(--border);margin:32px 0 16px'>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-family:var(--display);font-size:12px;color:var(--orange);"
        "letter-spacing:3px;margin-bottom:12px'>🪙 INSIGHT CODEX</div>",
        unsafe_allow_html=True,
    )
    insight_labels = {
        "xor_failure_witnessed": "The XOR Problem",
        "overfit_witnessed":     "The Generalization Gap",
        "regularization_used":   "Regularization Works",
        "duel_won":              "You Are The Architect",
    }
    row = ""
    for ins in insights:
        label = insight_labels.get(ins, ins.replace("_", " ").title())
        row += (
            f"<span style='font-family:var(--mono);font-size:10px;color:var(--orange);"
            f"border:1px solid rgba(255,107,43,0.3);border-radius:4px;padding:3px 10px;"
            f"margin-right:8px;margin-bottom:6px;display:inline-block'>🪙 {label}</span>"
        )
    st.markdown(f"<div style='display:flex;flex-wrap:wrap'>{row}</div>", unsafe_allow_html=True)

# ── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='font-family:var(--display);font-size:13px;color:var(--cyan);"
        "letter-spacing:2px;margin-bottom:16px'>NEURAL FORGE</div>",
        unsafe_allow_html=True,
    )
    st.markdown(f"**Session:** `{sm.get_session_id()[:8]}`")
    st.markdown(f"**XP:** {sm.get('xp', 0)}")
    st.markdown(f"**Level:** {sm.get('current_level', 1)}")
    st.markdown(f"**Runs:** {sm.get('total_runs', 0)}")
    st.divider()
    if st.button("🔄 Reset Progress", use_container_width=True):
        from utils.api_client import reset_progress
        reset_progress(sm.get_session_id())
        sm.sync_progress_from_api()
        st.rerun()
