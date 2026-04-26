import streamlit as st
from utils.formatters import failure_icon


def render_failure_banners(failure_report: dict, explanation_cards: list):
    if not failure_report:
        return

    flags = failure_report.get("flags", [])
    severity = failure_report.get("severity", 0.0)

    if not flags:
        st.markdown(
            "<div class='success-banner'>"
            "<span style='color:var(--green);font-weight:700'>✓ No issues detected</span>"
            " — training looks healthy."
            "</div>",
            unsafe_allow_html=True,
        )
        return

    # Diagnosis mode for high severity
    if severity >= 0.6:
        st.markdown(
            f"<div style='font-family:var(--mono);font-size:9px;color:var(--red);"
            f"letter-spacing:2px;margin-bottom:8px'>⚠ DIAGNOSIS MODE · severity {severity:.2f}</div>",
            unsafe_allow_html=True,
        )

    card_map = {c.get("flag_type", ""): c for c in explanation_cards} if explanation_cards else {}

    for flag in flags:
        icon = failure_icon(flag)
        card = card_map.get(flag, {})
        title = card.get("title", f"⚠ {flag.replace('_', ' ').title()}")
        body  = card.get("body", "")
        hint  = card.get("action_hint", "")

        color_class = "failure-banner" if flag in ("diverging", "exploding_loss", "overfitting") else "warn-banner"
        title_color = "var(--red)" if color_class == "failure-banner" else "var(--orange)"

        st.markdown(f"""
<div class="{color_class}">
  <div style="font-size:16px;flex-shrink:0">{icon}</div>
  <div>
    <div style="font-size:13px;font-weight:700;color:{title_color};margin-bottom:3px">{title}</div>
    <div style="font-size:11px;color:var(--muted);line-height:1.5">{body}</div>
    <div style="font-size:11px;color:var(--cyan);font-style:italic;margin-top:4px">→ {hint}</div>
  </div>
</div>
""", unsafe_allow_html=True)

    # Fix suggestions for high severity
    if severity >= 0.6:
        suggestions = _build_suggestions(flags)
        if suggestions:
            st.markdown(
                "<div style='background:rgba(139,92,246,0.06);border:1px solid rgba(139,92,246,0.2);"
                "border-radius:8px;padding:12px 16px;margin-top:8px'>"
                "<div style='font-family:var(--mono);font-size:9px;color:var(--purple);"
                "letter-spacing:1px;margin-bottom:8px'>FIX SUGGESTIONS</div>"
                + "".join(f"<div style='font-size:12px;color:var(--text);margin-bottom:5px'>• {s}</div>" for s in suggestions)
                + "</div>",
                unsafe_allow_html=True,
            )


def _build_suggestions(flags):
    suggestions = []
    if "overfitting" in flags:
        suggestions.append("Try adding L2 regularization (λ=0.001–0.01)")
    if "diverging" in flags or "oscillating" in flags:
        suggestions.append("Lower your learning rate by 10× (e.g. 0.01 → 0.001)")
    if "underfitting" in flags:
        suggestions.append("Add a hidden layer or increase neurons per layer")
    if "stalled" in flags:
        suggestions.append("Increase learning rate slightly or switch to ReLU activation")
    if "exploding_loss" in flags:
        suggestions.append("Reduce learning rate drastically — try 0.0001")
    return suggestions
