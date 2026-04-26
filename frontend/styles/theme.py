import streamlit as st


def inject_theme():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&display=swap');

/* ─── ROOT VARIABLES ──────────────────────────────────────── */
:root {
  --navy:    #0a0f1e;
  --navy2:   #0d1525;
  --card:    #111827;
  --card2:   #0f1d2f;
  --border:  #1f2d48;
  --cyan:    #00f5ff;
  --purple:  #8b5cf6;
  --orange:  #ff6b2b;
  --green:   #00ff88;
  --red:     #ff3b5c;
  --muted:   #4a5568;
  --text:    #c9d4e8;
  --mono:    'Share Tech Mono', monospace;
  --display: 'Orbitron', monospace;
  --ui:      'Rajdhani', sans-serif;
}

/* ─── GLOBAL PAGE ─────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
  background-color: var(--navy) !important;
  font-family: var(--ui) !important;
  color: var(--text) !important;
}

[data-testid="stAppViewContainer"] > .main {
  background-color: var(--navy) !important;
}

[data-testid="stHeader"] {
  background-color: rgba(10, 15, 30, 0.95) !important;
  border-bottom: 1px solid var(--border) !important;
  backdrop-filter: blur(12px) !important;
}

/* Scanline overlay */
[data-testid="stAppViewContainer"]::before {
  content: '';
  position: fixed;
  inset: 0;
  background: repeating-linear-gradient(
    0deg, transparent, transparent 2px,
    rgba(0,245,255,0.008) 2px, rgba(0,245,255,0.008) 4px
  );
  pointer-events: none;
  z-index: 9998;
}

/* ─── SIDEBAR ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background-color: var(--card) !important;
  border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] * {
  color: var(--text) !important;
}

/* ─── XP BAR — placed BELOW the Streamlit toolbar ─────────── */
/* We inject it as a sticky div INSIDE the main content area   */
.nf-xp-bar-wrapper {
  position: sticky;
  top: 0;
  z-index: 999;
  background: rgba(10,15,30,0.97);
  border-bottom: 1px solid var(--border);
  padding: 8px 20px;
  display: flex;
  align-items: center;
  gap: 16px;
  margin: -1rem -1rem 1rem -1rem;
  backdrop-filter: blur(10px);
}

.nf-logo {
  font-family: var(--display);
  font-size: 14px;
  font-weight: 900;
  color: var(--cyan);
  letter-spacing: 3px;
  text-shadow: 0 0 16px rgba(0,245,255,0.4);
  white-space: nowrap;
}

.nf-logo span { color: var(--orange); }

.nf-level-badge {
  font-family: var(--mono);
  font-size: 10px;
  color: var(--purple);
  border: 1px solid var(--purple);
  border-radius: 4px;
  padding: 3px 10px;
  letter-spacing: 2px;
  box-shadow: 0 0 8px rgba(139,92,246,0.3);
  white-space: nowrap;
}

.nf-xp-section {
  display: flex;
  align-items: center;
  gap: 10px;
  flex: 1;
}

.nf-xp-track {
  flex: 1;
  max-width: 200px;
  height: 5px;
  background: var(--border);
  border-radius: 3px;
  overflow: hidden;
}

.nf-xp-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--purple), var(--cyan));
  border-radius: 3px;
  transition: width 0.6s ease;
}

.nf-xp-label {
  font-family: var(--mono);
  font-size: 9px;
  color: var(--muted);
  white-space: nowrap;
}

.nf-session {
  font-family: var(--mono);
  font-size: 9px;
  color: var(--muted);
  margin-left: auto;
}

.nf-insight-count {
  font-family: var(--mono);
  font-size: 10px;
  color: var(--orange);
  border: 1px solid rgba(255,107,43,0.3);
  border-radius: 4px;
  padding: 2px 8px;
}

/* ─── CARDS ───────────────────────────────────────────────── */
.forge-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 20px;
  margin-bottom: 14px;
}

.forge-card-title {
  font-family: var(--mono);
  font-size: 9px;
  color: var(--muted);
  letter-spacing: 2px;
  text-transform: uppercase;
  margin-bottom: 14px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.forge-card-title::before {
  content: '';
  display: inline-block;
  width: 3px;
  height: 10px;
  background: var(--cyan);
  border-radius: 2px;
}

.locked-card {
  background: #0d1117;
  border: 1px dashed var(--border);
  border-radius: 12px;
  padding: 16px;
  opacity: 0.55;
  cursor: not-allowed;
  position: relative;
  margin-bottom: 8px;
}

.quest-card {
  background: linear-gradient(135deg, #1a1030, #0d1a2e);
  border: 1px solid var(--purple);
  border-radius: 10px;
  padding: 18px;
  box-shadow: 0 0 20px rgba(139,92,246,0.15);
  margin-bottom: 14px;
}

.tips-card {
  background: rgba(0,245,255,0.04);
  border: 1px solid rgba(0,245,255,0.2);
  border-radius: 10px;
  padding: 16px;
  margin-top: 10px;
}

/* ─── FAILURE BANNERS ─────────────────────────────────────── */
.failure-banner {
  display: flex;
  gap: 12px;
  align-items: flex-start;
  background: rgba(255,59,92,0.08);
  border-left: 3px solid var(--red);
  border-radius: 0 8px 8px 0;
  padding: 12px 14px;
  margin-bottom: 8px;
  animation: pulse-red 2s ease-in-out infinite;
}

.warn-banner {
  display: flex;
  gap: 12px;
  align-items: flex-start;
  background: rgba(255,107,43,0.08);
  border-left: 3px solid var(--orange);
  border-radius: 0 8px 8px 0;
  padding: 12px 14px;
  margin-bottom: 8px;
}

.success-banner {
  background: rgba(0,255,136,0.08);
  border: 1px solid rgba(0,255,136,0.3);
  border-radius: 8px;
  padding: 12px 16px;
  margin-bottom: 8px;
}

/* ─── BUTTONS ─────────────────────────────────────────────── */
.stButton > button {
  font-family: var(--ui) !important;
  font-weight: 700 !important;
  letter-spacing: 1px !important;
  border-radius: 8px !important;
  transition: all 0.2s !important;
}

.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, var(--cyan), #00b8cc) !important;
  color: var(--navy) !important;
  border: none !important;
  box-shadow: 0 0 20px rgba(0,245,255,0.25) !important;
}

.stButton > button[kind="primary"]:hover {
  box-shadow: 0 0 30px rgba(0,245,255,0.45) !important;
  transform: translateY(-1px) !important;
}

/* ─── WIDGETS ─────────────────────────────────────────────── */
.stSlider > div > div > div {
  background: var(--cyan) !important;
}

.stSelectbox > div > div {
  background: var(--card2) !important;
  border-color: var(--border) !important;
  color: var(--text) !important;
}

.stCheckbox > label {
  color: var(--text) !important;
}

[data-testid="stMetricValue"] {
  font-family: var(--mono) !important;
  color: var(--cyan) !important;
}

[data-testid="stMetricLabel"] {
  font-family: var(--mono) !important;
  color: var(--muted) !important;
  font-size: 10px !important;
}

/* ─── TABS ────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
  background: transparent !important;
  border-bottom: 1px solid var(--border) !important;
  gap: 0 !important;
}

.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  color: var(--muted) !important;
  font-family: var(--ui) !important;
  font-weight: 600 !important;
  border-radius: 0 !important;
  border-bottom: 2px solid transparent !important;
}

.stTabs [aria-selected="true"] {
  color: var(--cyan) !important;
  border-bottom-color: var(--cyan) !important;
}

.stTabs [data-baseweb="tab-panel"] {
  background: transparent !important;
  padding-top: 16px !important;
}

/* ─── EXPANDER ────────────────────────────────────────────── */
.streamlit-expanderHeader {
  background: var(--card2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 6px !important;
  color: var(--text) !important;
  font-family: var(--ui) !important;
}

/* ─── DIVIDER ─────────────────────────────────────────────── */
hr {
  border-color: var(--border) !important;
}

/* ─── ANIMATIONS ──────────────────────────────────────────── */
@keyframes pulse-red {
  0%, 100% { box-shadow: none; }
  50%       { box-shadow: inset 0 0 16px rgba(255,59,92,0.12); }
}

@keyframes glow-cyan {
  0%, 100% { box-shadow: 0 0 8px rgba(0,245,255,0.2); }
  50%       { box-shadow: 0 0 24px rgba(0,245,255,0.5); }
}

@keyframes level-up-flash {
  0%   { opacity: 0; transform: scale(0.8); }
  30%  { opacity: 1; transform: scale(1.15); }
  100% { opacity: 1; transform: scale(1); }
}

@keyframes unlock-pulse {
  0%, 100% { border-color: var(--purple); }
  50%       { border-color: var(--cyan); }
}

@keyframes slide-in {
  from { transform: translateX(40px); opacity: 0; }
  to   { transform: translateX(0); opacity: 1; }
}

.nf-slide-in { animation: slide-in 0.35s ease; }
.nf-glow     { animation: glow-cyan 2s ease-in-out infinite; }

/* ─── MISC HELPERS ────────────────────────────────────────── */
.nf-mono    { font-family: var(--mono) !important; }
.nf-display { font-family: var(--display) !important; }
.nf-muted   { color: var(--muted) !important; }
.nf-cyan    { color: var(--cyan) !important; }
.nf-orange  { color: var(--orange) !important; }
.nf-green   { color: var(--green) !important; }
.nf-red     { color: var(--red) !important; }
.nf-purple  { color: var(--purple) !important; }

.nf-tag {
  display: inline-block;
  font-family: var(--mono);
  font-size: 9px;
  letter-spacing: 2px;
  border-radius: 4px;
  padding: 2px 10px;
  margin-right: 6px;
}

.nf-world-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 18px;
  transition: all 0.25s;
  cursor: pointer;
  position: relative;
  overflow: hidden;
  margin-bottom: 10px;
}

.nf-world-card.unlocked {
  border-color: rgba(0,245,255,0.35);
  box-shadow: 0 0 18px rgba(0,245,255,0.07);
}

.nf-world-card.done {
  border-color: rgba(0,255,136,0.35);
  box-shadow: 0 0 14px rgba(0,255,136,0.07);
}

.nf-world-card.locked {
  opacity: 0.48;
  filter: grayscale(0.4);
  cursor: not-allowed;
}

/* HP bars for duel */
.hp-bar-wrap { margin-bottom: 6px; }
.hp-track {
  height: 10px;
  background: var(--border);
  border-radius: 5px;
  overflow: hidden;
}
.hp-fill-you  { height: 100%; background: linear-gradient(90deg, var(--cyan), #00b4cc); border-radius: 5px; transition: width 0.5s; }
.hp-fill-opp  { height: 100%; background: linear-gradient(90deg, var(--red), #cc2244); border-radius: 5px; transition: width 0.5s; }

/* Input fields */
.stTextInput > div > div > input {
  background: var(--card2) !important;
  border-color: var(--border) !important;
  color: var(--text) !important;
  font-family: var(--mono) !important;
}

/* Number inputs */
.stNumberInput > div > div > input {
  background: var(--card2) !important;
  color: var(--text) !important;
}

/* Plotly charts transparent bg */
.js-plotly-plot .plotly { background: transparent !important; }

</style>
""", unsafe_allow_html=True)


def xp_bar(level_id: int, xp: int, session_id: str, insights: list, xp_per_level: int = 1500):
    pct = min(100, int(xp / xp_per_level * 100))
    n_insights = len(insights)
    st.markdown(f"""
<div class="nf-xp-bar-wrapper">
  <div class="nf-logo">NEURAL<span>FORGE</span></div>
  <div class="nf-level-badge">LVL {level_id}</div>
  <div class="nf-xp-section">
    <span class="nf-xp-label">XP</span>
    <div class="nf-xp-track">
      <div class="nf-xp-fill" style="width:{pct}%"></div>
    </div>
    <span class="nf-xp-label" style="color:var(--text)">{xp}</span>
  </div>
  {'<div class="nf-insight-count">🪙 ' + str(n_insights) + ' Insights</div>' if n_insights else ''}
  <div class="nf-session">sess::{session_id[:6]}</div>
</div>
""", unsafe_allow_html=True)


def level_header(level_num: int, level_name: str, subtitle: str, color: str):
    st.markdown(f"""
<div style="margin-bottom:20px">
  <div style="display:inline-flex;align-items:center;gap:8px;
    font-family:var(--mono);font-size:10px;color:{color};
    border:1px solid {color}40;border-radius:4px;
    padding:3px 12px;letter-spacing:2px;margin-bottom:10px;
    background:{color}0d">
    ◈ LEVEL 0{level_num} · {level_name}
  </div>
  <div style="font-family:var(--display);font-size:26px;font-weight:900;
    color:#fff;letter-spacing:2px;line-height:1.1">
    {level_name.replace(' ', ' <span style="color:{color}">').replace('THE ', 'THE </span>') if 'THE' in level_name else f'<span style="color:{color}">{level_name}</span>'}
  </div>
  <div style="font-size:13px;color:var(--muted);margin-top:6px;font-weight:300">{subtitle}</div>
</div>
""", unsafe_allow_html=True)


def card_title(text: str, color: str = "var(--cyan)"):
    st.markdown(f"""
<div style="font-family:var(--mono);font-size:9px;color:var(--muted);
  letter-spacing:2px;text-transform:uppercase;margin-bottom:12px;
  display:flex;align-items:center;gap:8px">
  <span style="display:inline-block;width:3px;height:10px;
    background:{color};border-radius:2px"></span>
  {text}
</div>
""", unsafe_allow_html=True)


def metric_box(label: str, value: str, color: str = "var(--cyan)", sub: str = ""):
    st.markdown(f"""
<div style="background:var(--card2);border:1px solid var(--border);
  border-radius:8px;padding:12px;margin-bottom:8px">
  <div style="font-family:var(--mono);font-size:9px;color:var(--muted);
    letter-spacing:1px;text-transform:uppercase;margin-bottom:4px">{label}</div>
  <div style="font-family:var(--mono);font-size:22px;font-weight:700;
    color:{color}">{value}</div>
  {'<div style="font-size:10px;color:var(--muted);margin-top:2px">' + sub + '</div>' if sub else ''}
</div>
""", unsafe_allow_html=True)
