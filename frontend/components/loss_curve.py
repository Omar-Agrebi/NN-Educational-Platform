import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional


PLOT_LAYOUT = dict(
    plot_bgcolor="#0a0f1e",
    paper_bgcolor="#111827",
    font=dict(color="#e2e8f0", size=11),
    margin=dict(l=40, r=20, t=30, b=40),
    legend=dict(bgcolor="#111827", bordercolor="#1f2d48", borderwidth=1),
    xaxis=dict(gridcolor="#1f2d48", zerolinecolor="#1f2d48", title="Epoch"),
    yaxis=dict(gridcolor="#1f2d48", zerolinecolor="#1f2d48", title="Loss"),
)


def _build_fig(history: dict, show_accuracy: bool = True) -> go.Figure:
    epochs = history.get("epochs", [])
    train_loss = history.get("train_loss", [])
    test_loss = history.get("test_loss", [])
    train_acc = history.get("train_acc", [])
    test_acc = history.get("test_acc", [])

    if show_accuracy and (train_acc or test_acc):
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.6, 0.4], vertical_spacing=0.06,
            subplot_titles=["Loss", "Accuracy"]
        )
        row_loss, row_acc = 1, 2
    else:
        fig = go.Figure()
        row_loss, row_acc = None, None

    # Train loss
    if row_loss:
        fig.add_trace(go.Scatter(
            x=epochs, y=train_loss, name="Train Loss",
            line=dict(color="#00f5ff", width=2),
            mode="lines",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=epochs, y=test_loss, name="Test Loss",
            line=dict(color="#ff6b2b", width=2),
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(255,107,43,0.06)",
        ), row=1, col=1)
        if train_acc:
            fig.add_trace(go.Scatter(
                x=epochs, y=train_acc, name="Train Acc",
                line=dict(color="#00f5ff", width=2, dash="dot"),
                mode="lines",
            ), row=2, col=1)
        if test_acc:
            fig.add_trace(go.Scatter(
                x=epochs, y=test_acc, name="Test Acc",
                line=dict(color="#ff6b2b", width=2, dash="dot"),
                mode="lines",
            ), row=2, col=1)
    else:
        fig.add_trace(go.Scatter(
            x=epochs, y=train_loss, name="Train Loss",
            line=dict(color="#00f5ff", width=2), mode="lines",
        ))
        fig.add_trace(go.Scatter(
            x=epochs, y=test_loss, name="Test Loss",
            line=dict(color="#ff6b2b", width=2), mode="lines",
            fill="tonexty", fillcolor="rgba(255,107,43,0.06)",
        ))

    layout = dict(
        plot_bgcolor="#0a0f1e",
        paper_bgcolor="#111827",
        font=dict(color="#e2e8f0", size=11),
        margin=dict(l=40, r=20, t=30, b=40),
        legend=dict(bgcolor="#111827", bordercolor="#1f2d48", borderwidth=1),
        height=340,
    )
    if row_loss:
        layout["xaxis2"] = dict(gridcolor="#1f2d48", title="Epoch")
        layout["yaxis"] = dict(gridcolor="#1f2d48", title="Loss")
        layout["yaxis2"] = dict(gridcolor="#1f2d48", title="Accuracy")
    else:
        layout["xaxis"] = dict(gridcolor="#1f2d48", title="Epoch")
        layout["yaxis"] = dict(gridcolor="#1f2d48", title="Loss")

    fig.update_layout(**layout)
    return fig


def render_loss_curve(history: dict, key: str = "loss_curve",
                      show_accuracy: bool = True, height: int = 340):
    if not history or not history.get("epochs"):
        st.markdown('<div class="forge-card" style="text-align:center;color:#4a5568;padding:40px;">No training data yet</div>', unsafe_allow_html=True)
        return
    fig = _build_fig(history, show_accuracy)
    fig.update_layout(height=height)
    st.plotly_chart(fig, use_container_width=True, key=key)


def render_loss_curve_live(placeholder, history: dict, key: str = "loss_live"):
    """Update an st.empty placeholder with latest history."""
    fig = _build_fig(history, show_accuracy=True)
    placeholder.plotly_chart(fig, use_container_width=True, key=key)
