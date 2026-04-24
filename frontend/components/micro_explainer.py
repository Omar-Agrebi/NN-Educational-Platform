import streamlit as st


def render_micro_explainer(explanation_card: dict, trigger_condition: bool,
                            explainer_key: str = "explainer"):
    if not trigger_condition or not explanation_card:
        return

    dismissed_key = f"dismissed_{explainer_key}"
    if st.session_state.get(dismissed_key, False):
        return

    title = explanation_card.get("title", "Note")
    body = explanation_card.get("body", "")
    hint = explanation_card.get("action_hint", "")

    with st.expander(f"💡 {title}", expanded=True):
        st.markdown(f"""
        <div style="padding:6px 0;">
          <div style="color:#e2e8f0;font-size:0.86rem;line-height:1.6;margin-bottom:8px;">{body}</div>
          <div style="color:#00f5ff;font-size:0.82rem;font-style:italic;">{hint}</div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("✕ Dismiss", key=f"dismiss_{explainer_key}"):
            st.session_state[dismissed_key] = True
            st.rerun()
