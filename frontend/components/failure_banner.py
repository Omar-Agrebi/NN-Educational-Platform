import streamlit as st
from utils.formatters import failure_icon


def render_failure_banners(failure_report: dict, explanation_cards: list,
                            severity_threshold: float = 0.6):
    if not failure_report:
        return

    flags = failure_report.get("flags", [])
    severity = failure_report.get("severity", 0.0)

    if not flags:
        st.markdown("""
        <div class="success-banner">
          <span style="color:#00ff88;font-weight:700;">✓ No issues detected</span>
          <span style="color:#4a5568;font-size:0.82rem;margin-left:10px;">Training looks healthy.</span>
        </div>
        """, unsafe_allow_html=True)
        return

    # DIAGNOSIS MODE trigger
    if severity > severity_threshold:
        st.markdown("""
        <div style="background:rgba(255,59,92,0.05);border:1px solid rgba(255,59,92,0.2);
             border-radius:8px;padding:8px 12px;margin-bottom:8px;">
          <span style="color:#ff3b5c;font-weight:800;letter-spacing:0.1em;font-size:0.75rem;">
            ⚡ DIAGNOSIS MODE ACTIVE
          </span>
        </div>
        """, unsafe_allow_html=True)

    card_map = {c.get("title", ""): c for c in explanation_cards} if explanation_cards else {}

    for i, flag in enumerate(flags):
        icon = failure_icon(flag)
        title = flag.replace("_", " ").title()
        card = explanation_cards[i] if i < len(explanation_cards) else {}
        body = card.get("body", "") if card else ""
        hint = card.get("action_hint", "") if card else ""

        # Stagger delay
        delay = i * 0.15

        st.markdown(f"""
        <div class="failure-banner stagger-in" style="animation-delay:{delay}s;">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
            <span style="font-size:1.1rem;">{icon}</span>
            <span style="color:#ff3b5c;font-weight:700;font-size:0.9rem;">
              Model is {title}
            </span>
          </div>
          <div style="color:#e2e8f0;font-size:0.82rem;margin-bottom:6px;">{body}</div>
          <div style="color:#00f5ff;font-size:0.78rem;font-style:italic;">{hint}</div>
        </div>
        """, unsafe_allow_html=True)

    # Fix suggestions in diagnosis mode
    if severity > severity_threshold:
        suggestions = {
            "overfitting":  "Try L2 regularization (λ=0.01) or reduce hidden layers.",
            "underfitting": "Add a hidden layer or increase epochs to 200+.",
            "diverging":    "Lower your learning rate by 10× immediately.",
            "stalled":      "Increase learning rate or switch activation to ReLU.",
        }
        st.markdown('<div class="section-title" style="margin-top:12px;">FIX SUGGESTIONS</div>', unsafe_allow_html=True)
        for flag in flags:
            if flag in suggestions:
                st.markdown(f"""
                <div style="background:#111827;border:1px solid #1f2d48;border-radius:6px;
                     padding:8px 12px;margin-bottom:6px;font-size:0.82rem;">
                  <span style="color:#ff6b2b;">{failure_icon(flag)}</span>
                  <span style="color:#e2e8f0;margin-left:6px;">{suggestions[flag]}</span>
                </div>
                """, unsafe_allow_html=True)
