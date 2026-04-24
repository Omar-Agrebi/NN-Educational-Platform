import streamlit as st
from utils.formatters import level_color, fmt_pct, fmt_xp


def render_challenge_card(level_config: dict, challenge_result: dict = None,
                           on_submit=None, show_submit: bool = True):
    level_id = level_config.get("level_id", 1)
    color = level_color(level_id)
    name = level_config.get("name", f"Level {level_id}")
    desc = level_config.get("challenge_description", "Complete the challenge.")
    xp_reward = level_config.get("xp_reward", 100)
    unlock_reward = level_config.get("unlock_reward", [])

    passed = challenge_result.get("passed", False) if challenge_result else False
    score = challenge_result.get("score", 0.0) if challenge_result else 0.0
    xp_earned = challenge_result.get("xp_earned", 0) if challenge_result else 0
    msg = challenge_result.get("message", "") if challenge_result else ""

    if passed:
        border_color = "#00ff88"
        status_html = f'<span style="color:#00ff88;font-weight:800;">✓ CHALLENGE COMPLETE</span>'
        xp_html = f'<span style="background:#00ff8822;border:1px solid #00ff88;border-radius:4px;padding:2px 8px;font-family:monospace;color:#00ff88;">{fmt_xp(xp_earned)}</span>'
        glow = "box-shadow:0 0 18px rgba(0,255,136,0.3);"
    elif challenge_result is not None:
        border_color = "#ff6b2b"
        score_pct = f"{score * 100:.0f}%"
        status_html = f'<span style="color:#ff6b2b;font-weight:800;">⚠ CHALLENGE ACTIVE — {score_pct} complete</span>'
        xp_html = f'<span style="color:#4a5568;font-family:monospace;">Up to {fmt_xp(xp_reward)}</span>'
        glow = ""
    else:
        border_color = color
        status_html = f'<span style="color:{color};font-weight:700;">◈ ACTIVE QUEST</span>'
        xp_html = f'<span style="color:{color};font-family:monospace;">{fmt_xp(xp_reward)}</span>'
        glow = ""

    unlocks_html = " ".join([
        f'<span style="background:rgba(139,92,246,0.15);border:1px solid #8b5cf655;border-radius:3px;padding:1px 6px;font-size:0.68rem;color:#8b5cf6;font-family:monospace;">{u.replace("_"," ")}</span>'
        for u in unlock_reward
    ])

    st.markdown(f"""
    <div class="quest-card" style="border-color:{border_color};{glow}">
      <div style="margin-bottom:8px;">
        <span class="section-title">CHALLENGE</span>
        <div style="color:{color};font-weight:800;font-size:0.95rem;letter-spacing:0.05em;">
          {name}
        </div>
      </div>
      <div style="color:#e2e8f0;margin:8px 0;font-size:0.88rem;">{desc}</div>
      <div style="display:flex;justify-content:space-between;align-items:center;margin-top:10px;">
        {status_html}
        {xp_html}
      </div>
      {"<div style='margin-top:8px;font-size:0.78rem;color:#4a5568;'>Unlocks: " + unlocks_html + "</div>" if unlock_reward else ""}
      {"<div style='margin-top:8px;font-size:0.78rem;color:#8b5cf6;font-style:italic;'>" + msg + "</div>" if msg else ""}
    </div>
    """, unsafe_allow_html=True)

    if show_submit and not passed and on_submit:
        if st.button("📬 SUBMIT RUN", key=f"submit_challenge_{level_id}"):
            on_submit()
