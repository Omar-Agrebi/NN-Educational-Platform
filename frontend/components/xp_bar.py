import streamlit as st
from utils.formatters import level_color, short_session_id

XP_PER_LEVEL = {1: 0, 2: 100, 3: 300, 4: 500, 5: 800, 6: 1200, 7: 1700}


def _xp_progress(xp: int, level_id: int) -> float:
    lo = XP_PER_LEVEL.get(level_id, 0)
    hi = XP_PER_LEVEL.get(level_id + 1, lo + 500)
    if hi == lo:
        return 1.0
    return min(max((xp - lo) / (hi - lo), 0.0), 1.0)


def render_xp_bar(current_xp: int, level_id: int, unlocked_features: list, session_id: str = ""):
    color = level_color(level_id)
    pct = _xp_progress(current_xp, level_id)
    pct_str = f"{pct * 100:.0f}%"
    short_id = short_session_id(session_id)

    feature_tags = ""
    show_features = [f for f in unlocked_features
                     if f not in ("learning_rate", "epochs", "batch_size", "linear_dataset")][:6]
    for f in show_features:
        feature_tags += f'<span class="unlock-tag">{f.replace("_"," ")}</span>'

    html = f"""
    <div style="background:#0d1424;border-bottom:1px solid #1f2d48;padding:10px 20px;
                display:flex;align-items:center;gap:20px;flex-wrap:wrap;margin-bottom:8px;">
      <span class="level-badge" style="color:{color};border-color:{color};">
        LVL {level_id}
      </span>
      <div style="flex:1;min-width:160px;">
        <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
          <span style="font-size:0.72rem;color:#4a5568;font-family:monospace;">{current_xp} XP</span>
          <span style="font-size:0.72rem;color:#4a5568;">{pct_str}</span>
        </div>
        <div class="xp-bar-track">
          <div class="xp-bar-fill" style="width:{pct_str};"></div>
        </div>
      </div>
      <div style="display:flex;gap:6px;flex-wrap:wrap;align-items:center;">
        {feature_tags}
      </div>
      <span style="font-family:monospace;font-size:0.68rem;color:#1f2d48;margin-left:auto;">
        SESSION: {short_id}
      </span>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
