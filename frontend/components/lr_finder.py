import streamlit as st
import plotly.graph_objects as go
import numpy as np
from utils.api_client import run_lr_finder, APIError
from utils import state_manager as sm
from styles.theme import card_title


def render_lr_finder(dataset: str, model_config: dict):
    card_title("Learning Rate Finder", color="var(--orange)")

    st.markdown(
        "<div style='font-size:12px;color:var(--muted);margin-bottom:12px;line-height:1.5'>"
        "Runs 10 quick experiments across a range of learning rates. "
        "The recommended LR is where loss drops fastest."
        "</div>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        lr_min = st.number_input("Min LR", value=0.0001, format="%.4f", step=0.0001)
    with col2:
        lr_max = st.number_input("Max LR", value=1.0, format="%.3f", step=0.01)

    if st.button("🔍 Find Best LR", key="lr_finder_btn"):
        with st.spinner("Running LR sweep…"):
            try:
                result = run_lr_finder(
                    session_id=sm.get_session_id(),
                    dataset=dataset,
                    model_config=model_config,
                    lr_min=lr_min,
                    lr_max=lr_max,
                    n_trials=10,
                )
                sm.set("lr_finder_result", result)
            except APIError as e:
                st.error(f"LR Finder failed: {e}")
                return

    result = sm.get("lr_finder_result")
    if not result:
        return

    lrs   = result.get("learning_rates", [])
    losses = result.get("losses", [])
    rec_lr = result.get("recommended_lr", None)
    rec_idx = result.get("recommended_idx", 0)

    if not lrs:
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=lrs, y=losses,
        mode="lines+markers",
        line=dict(color="#00f5ff", width=2),
        marker=dict(size=6, color="#00f5ff"),
        name="Loss",
    ))

    if rec_lr is not None:
        fig.add_vline(
            x=rec_lr,
            line_dash="dash",
            line_color="#00ff88",
            annotation_text=f"Recommended: {rec_lr:.4f}",
            annotation_font_color="#00ff88",
            annotation_font_size=9,
        )

    fig.update_layout(
        paper_bgcolor="#111827",
        plot_bgcolor="#0a0f1e",
        height=200,
        margin=dict(l=48, r=12, t=12, b=36),
        xaxis=dict(
            type="log",
            title="Learning Rate (log scale)",
            title_font=dict(size=9, color="#4a5568"),
            gridcolor="#1f2d48",
            tickfont=dict(size=9, color="#4a5568", family="Share Tech Mono"),
        ),
        yaxis=dict(
            title="Train Loss (20 epochs)",
            title_font=dict(size=9, color="#4a5568"),
            gridcolor="#1f2d48",
            tickfont=dict(size=9, color="#4a5568", family="Share Tech Mono"),
        ),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, key="lr_finder_chart")

    if rec_lr:
        st.markdown(
            f"<div class='success-banner'>"
            f"<span style='color:var(--green);font-weight:700'>Recommended LR: "
            f"<span style='font-family:var(--mono)'>{rec_lr:.4f}</span></span>"
            f" — lowest loss after 20 epochs of quick training."
            f"</div>",
            unsafe_allow_html=True,
        )
