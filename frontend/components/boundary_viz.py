import streamlit as st
import plotly.graph_objects as go
from typing import Optional


def _build_boundary_fig(boundary_data: dict, title: str = "Decision Boundary",
                         epoch: Optional[int] = None) -> go.Figure:
    xx = boundary_data.get("xx", [])
    yy = boundary_data.get("yy", [])
    Z = boundary_data.get("Z", [])
    x_pts = boundary_data.get("x_points", [])
    y_pts = boundary_data.get("y_points", [])
    labels = boundary_data.get("labels", [])

    fig = go.Figure()

    if xx and yy and Z:
        fig.add_trace(go.Contour(
            x=xx[0] if xx else [],
            y=[row[0] for row in yy] if yy else [],
            z=Z,
            colorscale=[[0, "#ff3b5c"], [0.45, "#1a0d1a"], [0.55, "#0d1a1a"], [1, "#00f5ff"]],
            showscale=False,
            opacity=0.7,
            contours=dict(showlines=False),
            name="Boundary",
        ))

    if x_pts and labels is not None:
        class0_x = [x for x, l in zip(x_pts, labels) if l == 0]
        class0_y = [y for y, l in zip(y_pts, labels) if l == 0]
        class1_x = [x for x, l in zip(x_pts, labels) if l == 1]
        class1_y = [y for y, l in zip(y_pts, labels) if l == 1]

        fig.add_trace(go.Scatter(
            x=class0_x, y=class0_y, mode="markers", name="Class 0",
            marker=dict(symbol="circle", color="#ff3b5c", size=6,
                        line=dict(color="#ffffff", width=1)),
        ))
        fig.add_trace(go.Scatter(
            x=class1_x, y=class1_y, mode="markers", name="Class 1",
            marker=dict(symbol="diamond", color="#00f5ff", size=6,
                        line=dict(color="#ffffff", width=1)),
        ))

    annotations = []
    if epoch is not None:
        annotations.append(dict(
            text=f"Epoch {epoch}",
            xref="paper", yref="paper",
            x=0.98, y=0.98,
            showarrow=False,
            font=dict(color="#4a5568", size=11, family="monospace"),
            align="right",
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(color="#4a5568", size=12), x=0.02),
        plot_bgcolor="#0a0f1e",
        paper_bgcolor="#111827",
        font=dict(color="#e2e8f0", size=11),
        margin=dict(l=30, r=20, t=40, b=30),
        xaxis=dict(gridcolor="rgba(31,45,72,0.3)", zerolinecolor="#1f2d48", title="x₁"),
        yaxis=dict(gridcolor="rgba(31,45,72,0.3)", zerolinecolor="#1f2d48", title="x₂"),
        legend=dict(bgcolor="#111827", bordercolor="#1f2d48", borderwidth=1),
        height=360,
        annotations=annotations,
    )
    return fig


def render_boundary(boundary_data: Optional[dict], title: str = "Decision Boundary",
                    epoch: Optional[int] = None, key: str = "boundary"):
    if not boundary_data:
        st.markdown(
            '<div class="forge-card" style="text-align:center;color:#4a5568;height:300px;'
            'display:flex;align-items:center;justify-content:center;">'
            '<span>Train a model to see the decision boundary</span></div>',
            unsafe_allow_html=True,
        )
        return
    fig = _build_boundary_fig(boundary_data, title, epoch)
    st.plotly_chart(fig, use_container_width=True, key=key)


def render_boundary_live(placeholder, boundary_data: dict,
                          epoch: Optional[int] = None, key: str = "boundary_live"):
    fig = _build_boundary_fig(boundary_data, "Decision Boundary", epoch)
    placeholder.plotly_chart(fig, use_container_width=True, key=key)
