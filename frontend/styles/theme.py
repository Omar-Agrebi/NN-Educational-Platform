import streamlit as st

ANIMATIONS_CSS = """
@keyframes pulse-red {
    0%, 100% { box-shadow: 0 0 0px rgba(255,59,92,0); }
    50%       { box-shadow: 0 0 20px rgba(255,59,92,0.7); }
}
@keyframes pulse-cyan {
    0%, 100% { box-shadow: 0 0 4px rgba(0,245,255,0.3); }
    50%       { box-shadow: 0 0 20px rgba(0,245,255,0.7); }
}
@keyframes fill-bar {
    from { width: 0%; }
    to   { width: var(--target-width, 100%); }
}
@keyframes level-up-flash {
    0%   { opacity: 0; transform: scale(0.8); }
    30%  { opacity: 1; transform: scale(1.12); }
    100% { opacity: 1; transform: scale(1); }
}
@keyframes unlock-pulse {
    0%, 100% { border-color: #8b5cf6; }
    50%       { border-color: #00f5ff; }
}
@keyframes boundary-fade {
    from { opacity: 0; }
    to   { opacity: 1; }
}
@keyframes draw-curve {
    from { stroke-dashoffset: 1000; }
    to   { stroke-dashoffset: 0; }
}
@keyframes scan-red {
    0%   { background-position: 0% 0%; }
    100% { background-position: 0% 100%; }
}
@keyframes glitch {
    0%  { text-shadow: 2px 0 #ff3b5c, -2px 0 #00f5ff; }
    20% { text-shadow: -2px 0 #ff3b5c, 2px 0 #00f5ff; }
    40% { text-shadow: 2px 2px #8b5cf6, -2px -2px #00f5ff; }
    60% { text-shadow: -1px 0 #ff3b5c, 1px 0 #00ff88; }
    80% { text-shadow: 2px 0 #00f5ff, -2px 0 #ff3b5c; }
    100%{ text-shadow: none; }
}
@keyframes typewriter {
    from { width: 0; }
    to   { width: 100%; }
}
@keyframes blink-cursor {
    from, to { border-right-color: transparent; }
    50%      { border-right-color: #00f5ff; }
}
@keyframes hex-drift {
    0%   { transform: translateY(0px) rotate(0deg); opacity: 0.04; }
    50%  { transform: translateY(-20px) rotate(3deg); opacity: 0.08; }
    100% { transform: translateY(0px) rotate(0deg); opacity: 0.04; }
}
@keyframes vs-pulse {
    0%, 100% { color: #ff6b2b; text-shadow: 0 0 10px rgba(255,107,43,0.5); }
    50%       { color: #ff3b5c; text-shadow: 0 0 24px rgba(255,59,92,0.9); }
}
@keyframes hp-grow {
    from { width: 0%; }
    to   { width: var(--hp, 50%); }
}
@keyframes stagger-in {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
"""

MAIN_CSS = """
/* ── Reset & Base ──────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}
.stApp {
    background-color: #0a0f1e !important;
    color: #e2e8f0 !important;
}
.block-container {
    padding-top: 1rem !important;
    max-width: 1400px;
}

/* ── Sidebar ───────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #0d1424 !important;
    border-right: 1px solid #1f2d48;
}

/* ── Buttons ───────────────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #00f5ff22, #8b5cf622) !important;
    color: #00f5ff !important;
    border: 1px solid #00f5ff !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #00f5ff44, #8b5cf644) !important;
    box-shadow: 0 0 16px rgba(0,245,255,0.4) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── Train button override ─────────────────────────────────────────────── */
.train-btn > button {
    background: linear-gradient(135deg, #00f5ff, #8b5cf6) !important;
    color: #0a0f1e !important;
    font-size: 1.1rem !important;
    font-weight: 800 !important;
    padding: 0.75rem 2rem !important;
    border: none !important;
    box-shadow: 0 0 24px rgba(0,245,255,0.5) !important;
}

/* ── Sliders ───────────────────────────────────────────────────────────── */
.stSlider [data-baseweb="slider"] {
    padding: 0.2rem 0;
}
.stSlider [data-testid="stTickBar"] {
    color: #4a5568 !important;
}

/* ── Selectbox ─────────────────────────────────────────────────────────── */
.stSelectbox [data-baseweb="select"] > div {
    background-color: #111827 !important;
    border-color: #1f2d48 !important;
    color: #e2e8f0 !important;
}

/* ── Metrics ───────────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #111827;
    border: 1px solid #1f2d48;
    border-radius: 10px;
    padding: 0.75rem 1rem;
}
[data-testid="stMetricLabel"] { color: #4a5568 !important; font-size: 0.75rem; }
[data-testid="stMetricValue"] { color: #00f5ff !important; font-family: monospace; }
[data-testid="stMetricDelta"] { font-family: monospace; }

/* ── Tabs ──────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: #111827;
    border-bottom: 1px solid #1f2d48;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    color: #4a5568 !important;
    font-weight: 600;
    border-bottom: 2px solid transparent;
}
.stTabs [aria-selected="true"] {
    color: #00f5ff !important;
    border-bottom: 2px solid #00f5ff !important;
    background: transparent !important;
}

/* ── Expander ──────────────────────────────────────────────────────────── */
.streamlit-expanderHeader {
    background: #111827 !important;
    border: 1px solid #1f2d48 !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}
.streamlit-expanderContent {
    background: #0d1424 !important;
    border: 1px solid #1f2d48 !important;
    border-top: none !important;
}

/* ── Toggle ────────────────────────────────────────────────────────────── */
.stCheckbox label { color: #e2e8f0 !important; }

/* ── Custom card classes ────────────────────────────────────────────────── */
.forge-card {
    background: #111827;
    border: 1px solid #1f2d48;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 12px;
}
.locked-card {
    background: #0d1117;
    border: 1px solid #1f2d48;
    border-radius: 12px;
    padding: 20px;
    opacity: 0.55;
    filter: grayscale(0.6);
    cursor: not-allowed;
    position: relative;
    margin-bottom: 12px;
}
.locked-card::before {
    content: "🔒";
    position: absolute;
    top: 12px;
    right: 14px;
    font-size: 1.2rem;
    opacity: 0.7;
}
.quest-card {
    background: linear-gradient(135deg, #1a1030, #0d1a2e);
    border: 1px solid #8b5cf6;
    border-radius: 8px;
    padding: 20px;
    box-shadow: 0 0 12px rgba(139,92,246,0.25);
    margin-bottom: 12px;
}
.failure-banner {
    background: rgba(255,59,92,0.10);
    border-left: 3px solid #ff3b5c;
    border-radius: 0 8px 8px 0;
    padding: 14px 16px;
    margin-bottom: 10px;
    animation: pulse-red 1.8s ease-in-out infinite;
}
.success-banner {
    background: rgba(0,255,136,0.08);
    border-left: 3px solid #00ff88;
    border-radius: 0 8px 8px 0;
    padding: 14px 16px;
    margin-bottom: 10px;
}
.xp-bar-track {
    background: #1f2d48;
    border-radius: 999px;
    height: 10px;
    width: 100%;
    overflow: hidden;
    margin-top: 6px;
}
.xp-bar-fill {
    background: linear-gradient(90deg, #8b5cf6, #00f5ff);
    height: 100%;
    border-radius: 999px;
    animation: fill-bar 0.8s ease-out forwards;
    transition: width 0.8s ease;
}
.glow-border {
    box-shadow: 0 0 16px rgba(0,245,255,0.35);
    border: 1px solid #00f5ff !important;
}
.glow-purple {
    box-shadow: 0 0 16px rgba(139,92,246,0.35);
    border: 1px solid #8b5cf6 !important;
}
.level-badge {
    font-family: monospace;
    background: #1a1f35;
    border: 1px solid currentColor;
    border-radius: 6px;
    padding: 3px 12px;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    display: inline-block;
}
.metric-mono {
    font-family: monospace;
    font-size: 1.4rem;
    font-weight: 700;
}
.section-title {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    color: #4a5568;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.unlock-tag {
    background: rgba(0,245,255,0.1);
    border: 1px solid rgba(0,245,255,0.3);
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.7rem;
    color: #00f5ff;
    font-family: monospace;
    display: inline-block;
    margin: 2px;
    animation: unlock-pulse 2s ease-in-out infinite;
}
.diagnosis-overlay {
    background: linear-gradient(180deg, transparent 0%, rgba(255,59,92,0.04) 100%);
    border: 1px solid rgba(255,59,92,0.2);
    border-radius: 12px;
    padding: 16px;
    animation: scan-red 3s linear infinite;
}
.hp-bar-track {
    background: #1f2d48;
    border-radius: 4px;
    height: 16px;
    width: 100%;
    overflow: hidden;
    margin: 4px 0;
}
.hp-bar-fill {
    height: 100%;
    border-radius: 4px;
    animation: hp-grow 1s ease-out forwards;
    transition: width 1s ease;
}
.vs-text {
    font-size: 3rem;
    font-weight: 900;
    text-align: center;
    animation: vs-pulse 1.2s ease-in-out infinite;
    color: #ff6b2b;
    line-height: 1;
    letter-spacing: 0.1em;
}
.dna-label {
    font-family: monospace;
    font-size: 0.65rem;
    color: #4a5568;
}
.stagger-in {
    animation: stagger-in 0.4s ease-out both;
}
.insight-tag {
    background: rgba(245,158,11,0.1);
    border: 1px solid rgba(245,158,11,0.4);
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.72rem;
    color: #f59e0b;
    font-family: monospace;
    display: inline-block;
    margin: 2px;
}
.page-header {
    padding: 12px 0 8px 0;
    border-bottom: 1px solid #1f2d48;
    margin-bottom: 16px;
}
"""


def inject_theme():
    st.markdown(f"<style>{ANIMATIONS_CSS}{MAIN_CSS}</style>", unsafe_allow_html=True)
