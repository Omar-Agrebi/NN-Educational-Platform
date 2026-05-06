import streamlit as st
import plotly.graph_objects as go
import numpy as np
from styles.theme import card_title

COLORSCALE = [
    [0.0, "#ff3b5c"],
    [0.4, "#1a0a14"],
    [0.5, "#111827"],
    [0.6, "#0a1a1f"],
    [1.0, "#00f5ff"],
]


def render_boundary(
    boundary_data: dict,
    title: str = "Decision Boundary",
    epoch: int = None,
    height: int = 340,
    key: str = "boundary",
):
    card_title(title)

    if not boundary_data or not boundary_data.get("xx"):
        st.markdown(
            "<div style='height:200px;display:flex;align-items:center;justify-content:center;"
            "color:var(--muted);font-family:var(--mono);font-size:11px;background:#050a15;"
            "border-radius:8px'>Run training to see decision boundary</div>",
            unsafe_allow_html=True,
        )
        return

    xx = np.array(boundary_data["xx"])
    yy = np.array(boundary_data["yy"])
    Z  = np.array(boundary_data["Z"])
    x_pts = boundary_data.get("x_points", [])
    y_pts = boundary_data.get("y_points", [])
    labels = boundary_data.get("labels", [])

    fig = go.Figure()

    # Filled contour
    fig.add_trace(go.Contour(
        x=xx[0], y=yy[:, 0], z=Z,
        colorscale=COLORSCALE,
        showscale=False,
        contours=dict(showlines=False),
        opacity=0.85,
        name="Boundary",
    ))

    # Data points — class 0
    if x_pts and labels:
        x_arr = np.array(x_pts)
        y_arr = np.array(y_pts)
        lbl_arr = np.array(labels)

        mask0 = lbl_arr == 0
        mask1 = lbl_arr == 1

        fig.add_trace(go.Scatter(
            x=x_arr[mask0], y=y_arr[mask0],
            mode="markers",
            marker=dict(
                symbol="circle", size=7,
                color="#ff3b5c",
                line=dict(color="white", width=0.8),
            ),
            name="Class 0",
        ))
        fig.add_trace(go.Scatter(
            x=x_arr[mask1], y=y_arr[mask1],
            mode="markers",
            marker=dict(
                symbol="diamond", size=7,
                color="#00f5ff",
                line=dict(color="white", width=0.8),
            ),
            name="Class 1",
        ))

    annotation_text = f"Epoch {epoch}" if epoch is not None else "Final"
    fig.update_layout(
        paper_bgcolor="#0a0f1e",
        plot_bgcolor="#050a15",
        margin=dict(l=8, r=8, t=8, b=8),
        height=height,
        showlegend=True,
        legend=dict(
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="#1f2d48",
            borderwidth=1,
            font=dict(color="#c9d4e8", size=9, family="Share Tech Mono"),
            x=0.01, y=0.99,
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        annotations=[dict(
            text=annotation_text,
            x=0.99, y=0.99,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(color="#00f5ff", size=9, family="Share Tech Mono"),
            bgcolor="rgba(0,0,0,0.5)",
            borderpad=4,
        )],
    )

    st.plotly_chart(fig, use_container_width=True, key=key)


def render_boundary_live(placeholder, boundary_data: dict, epoch: int, height: int = 300):
    """Updates an st.empty() placeholder during live training."""
    if not boundary_data or not boundary_data.get("xx"):
        return
    xx = np.array(boundary_data["xx"])
    yy = np.array(boundary_data["yy"])
    Z  = np.array(boundary_data["Z"])
    x_pts = np.array(boundary_data.get("x_points", []))
    y_pts = np.array(boundary_data.get("y_points", []))
    labels = np.array(boundary_data.get("labels", []))

    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=xx[0], y=yy[:, 0], z=Z,
        colorscale=COLORSCALE,
        showscale=False,
        contours=dict(showlines=False),
        opacity=0.85,
    ))
    if len(x_pts):
        for cls, sym, col in [(0, "circle", "#ff3b5c"), (1, "diamond", "#00f5ff")]:
            m = labels == cls
            fig.add_trace(go.Scatter(
                x=x_pts[m], y=y_pts[m], mode="markers",
                marker=dict(symbol=sym, size=6, color=col, line=dict(color="white", width=0.5)),
                showlegend=False,
            ))
    fig.update_layout(
        paper_bgcolor="#0a0f1e", plot_bgcolor="#050a15",
        margin=dict(l=4, r=4, t=4, b=4), height=height,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        annotations=[dict(
            text=f"Epoch {epoch}", x=0.99, y=0.99,
            xref="paper", yref="paper", showarrow=False,
            font=dict(color="#00f5ff", size=9, family="Share Tech Mono"),
            bgcolor="rgba(0,0,0,0.6)", borderpad=3,
        )],
    )
    placeholder.plotly_chart(fig, use_container_width=True, key=f"live_boundary_{epoch}")
