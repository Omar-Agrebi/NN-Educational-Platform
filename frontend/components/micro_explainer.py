import streamlit as st
from utils import state_manager as sm
from utils.formatters import insight_message


def render_micro_explainer(card: dict, trigger: bool = True, key: str = "explainer"):
    if not trigger or not card:
        return
    title = card.get("title", "")
    body  = card.get("body", "")
    hint  = card.get("action_hint", "")
    dismissed = sm.get("dismissed_explainers") or []
    if key in dismissed:
        return
    with st.expander(f"📖 {title}", expanded=True):
        st.markdown(f"""
<div style="padding:4px 0">
  <div style="font-size:13px;color:var(--text);line-height:1.6;margin-bottom:8px">{body}</div>
  <div style="font-size:12px;color:var(--cyan);font-style:italic">→ {hint}</div>
</div>
""", unsafe_allow_html=True)
        if st.button("Dismiss", key=f"dismiss_{key}"):
            dismissed.append(key)
            sm.set("dismissed_explainers", dismissed)
            st.rerun()


def render_insight_toast(insight_key: str):
    """Show a toast notification for a new insight."""
    title, msg = insight_message(insight_key)
    st.toast(f"{title}: {msg}", icon="🪙")


def check_and_show_insights(result: dict):
    """Check training result for new insights and show toasts."""
    if not result:
        return
    failure = result.get("failure_report", {})
    flags = failure.get("flags", [])
    dataset = result.get("dataset", "")

    # XOR failure insight
    model_cfg = result.get("model_config", {})
    if "xor_failure_witnessed" not in (sm.get("insights") or []):
        hl = model_cfg.get("hidden_layers", 0) if model_cfg else 0
        if hl == 0 and "underfitting" in flags:
            if sm.add_insight("xor_failure_witnessed"):
                render_insight_toast("xor_failure_witnessed")

    # Overfit insight
    if "overfit_witnessed" not in (sm.get("insights") or []):
        if "overfitting" in flags:
            if sm.add_insight("overfit_witnessed"):
                render_insight_toast("overfit_witnessed")

    # Regularization insight
    if "regularization_used" not in (sm.get("insights") or []):
        reg = model_cfg.get("regularization", "none") if model_cfg else "none"
        if reg != "none":
            if sm.add_insight("regularization_used"):
                render_insight_toast("regularization_used")
