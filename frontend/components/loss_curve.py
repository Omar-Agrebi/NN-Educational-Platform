import streamlit as st
import plotly.graph_objects as go
from typing import Optional
from styles.theme import card_title

PLOT_LAYOUT = dict(
    paper_bgcolor="#111827",
    plot_bgcolor="#0a0f1e",
    font=dict(family="Share Tech Mono", color="#4a5568", size=10),
    margin=dict(l=48, r=12, t=24, b=36),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor="#1f2d48",
        borderwidth=1,
        font=dict(size=10),
    ),
    xaxis=dict(
        gridcolor="#1f2d48",
        linecolor="#1f2d48",
        tickfont=dict(size=9),
        title_font=dict(size=10),
    ),
    yaxis=dict(
        gridcolor="#1f2d48",
        linecolor="#1f2d48",
        tickfont=dict(size=9),
        title_font=dict(size=10),
    ),
)


def render_loss_curve(
    history: dict,
    title: str = "Loss Curves",
    show_accuracy: bool = True,
    height: int = 220,
    key: str = "loss_curve",
):
    card_title(title)

    epochs = history.get("epochs", [])
    train_loss = history.get("train_loss", [])
    test_loss = history.get("test_loss", [])
    train_acc = history.get("train_acc", [])
    test_acc = history.get("test_acc", [])

    if not epochs:
        st.markdown(
            "<div style='height:120px;display:flex;align-items:center;justify-content:center;"
            "color:var(--muted);font-family:var(--mono);font-size:11px'>"
            "Run training to see loss curves</div>",
            unsafe_allow_html=True,
        )
        return

    rows = 2 if show_accuracy and train_acc else 1
    fig = go.Figure() if rows == 1 else None

    if rows == 2:
        from plotly.subplots import make_subplots
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=("Loss", "Accuracy"),
        )
        # Loss
        fig.add_trace(go.Scatter(
            x=epochs, y=train_loss,
            mode="lines", name="Train Loss",
            line=dict(color="#00f5ff", width=2),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=epochs, y=test_loss,
            mode="lines", name="Test Loss",
            line=dict(color="#ff6b2b", width=2),
            fill="tonexty",
            fillcolor="rgba(255,107,43,0.06)",
        ), row=1, col=1)
        # Accuracy
        fig.add_trace(go.Scatter(
            x=epochs, y=[a * 100 for a in train_acc],
            mode="lines", name="Train Acc %",
            line=dict(color="#00f5ff", width=2, dash="dot"),
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=epochs, y=[a * 100 for a in test_acc],
            mode="lines", name="Test Acc %",
            line=dict(color="#ff6b2b", width=2, dash="dot"),
        ), row=2, col=1)

        fig.update_layout(
            **PLOT_LAYOUT,
            height=height * 2,
            showlegend=True,
        )
        for i in [1, 2]:
            fig.update_xaxes(gridcolor="#1f2d48", linecolor="#1f2d48", row=i, col=1)
            fig.update_yaxes(gridcolor="#1f2d48", linecolor="#1f2d48", row=i, col=1)
        # style subplot titles
        for ann in fig.layout.annotations:
            ann.font.color = "#4a5568"
            ann.font.size = 9
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=epochs, y=train_loss,
            mode="lines", name="Train Loss",
            line=dict(color="#00f5ff", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=epochs, y=test_loss,
            mode="lines", name="Test Loss",
            line=dict(color="#ff6b2b", width=2),
            fill="tonexty",
            fillcolor="rgba(255,107,43,0.06)",
        ))
        fig.update_layout(
            **PLOT_LAYOUT,
            height=height,
            xaxis_title="Epoch",
            yaxis_title="Loss",
        )

    # Best epoch marker
    if test_loss:
        best_ep = epochs[test_loss.index(min(test_loss))]
        best_val = min(test_loss)
        fig.add_vline(
            x=best_ep,
            line_dash="dash",
            line_color="rgba(0,255,136,0.4)",
            annotation_text=f"best@{best_ep}",
            annotation_font_color="#00ff88",
            annotation_font_size=8,
        )

    st.plotly_chart(fig, use_container_width=True, key=key)


def render_loss_curve_live(placeholder, history: dict, height: int = 180):
    """Used during live training — updates an st.empty() placeholder."""
    epochs = history.get("epochs", [])
    train_loss = history.get("train_loss", [])
    test_loss = history.get("test_loss", [])

    if not epochs:
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs, y=train_loss,
        mode="lines", name="Train",
        line=dict(color="#00f5ff", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=epochs, y=test_loss,
        mode="lines", name="Test",
        line=dict(color="#ff6b2b", width=2),
        fill="tonexty",
        fillcolor="rgba(255,107,43,0.06)",
    ))
    fig.update_layout(
        **PLOT_LAYOUT,
        height=height,
        showlegend=True,
        xaxis_title="Epoch",
        yaxis_title="Loss",
    )
    placeholder.plotly_chart(fig, use_container_width=True, key="live_loss")
