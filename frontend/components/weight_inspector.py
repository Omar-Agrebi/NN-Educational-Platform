import streamlit as st
import plotly.graph_objects as go
import numpy as np
from styles.theme import card_title


def render_weight_inspector(weight_snapshots: list, model_config: dict):
    """
    Weight Inspector: click a neuron, see its weights as a heatmap.
    Shows the final snapshot's weights.
    """
    card_title("Weight Inspector", color="var(--purple)")

    if not weight_snapshots:
        st.markdown(
            "<div style='color:var(--muted);font-size:11px;font-family:var(--mono)'>No weights yet — run training first.</div>",
            unsafe_allow_html=True,
        )
        return

    # Use last snapshot
    snap = weight_snapshots[-1]
    weights = snap.get("weights", [])
    biases  = snap.get("biases", [])
    n_hidden = model_config.get("hidden_layers", 0)
    acts = model_config.get("activations_per_layer", [])

    if not weights:
        return

    layer_options = [f"Layer {i+1} → Layer {i+2}" for i in range(len(weights))]
    if not layer_options:
        return

    selected_layer = st.selectbox(
        "Inspect layer",
        options=layer_options,
        index=0,
        key="weight_inspector_layer",
    )
    li = layer_options.index(selected_layer)
    W = np.array(weights[li])  # shape (fan_in, fan_out)

    fig = go.Figure(data=go.Heatmap(
        z=W.T,
        colorscale=[
            [0.0, "#ff3b5c"],
            [0.5, "#111827"],
            [1.0, "#00f5ff"],
        ],
        zmid=0,
        colorbar=dict(
            tickfont=dict(color="#4a5568", size=8, family="Share Tech Mono"),
            len=0.8,
        ),
        hovertemplate="in:%{x} out:%{y}<br>w:%{z:.4f}<extra></extra>",
    ))

    in_label  = "Input" if li == 0 else f"H{li}"
    out_label = f"H{li+1}" if li < len(weights) - 1 else "Output"

    fig.update_layout(
        paper_bgcolor="#111827",
        plot_bgcolor="#0a0f1e",
        height=180,
        margin=dict(l=8, r=8, t=24, b=8),
        xaxis=dict(
            title=f"← {in_label} neurons",
            title_font=dict(size=9, color="#4a5568"),
            tickfont=dict(size=8, color="#4a5568", family="Share Tech Mono"),
            gridcolor="#1f2d48",
        ),
        yaxis=dict(
            title=f"{out_label} neurons →",
            title_font=dict(size=9, color="#4a5568"),
            tickfont=dict(size=8, color="#4a5568", family="Share Tech Mono"),
            gridcolor="#1f2d48",
        ),
        title=dict(
            text=f"Weight matrix: {in_label}→{out_label} ({W.shape[0]}×{W.shape[1]})",
            font=dict(size=9, color="#4a5568", family="Share Tech Mono"),
        ),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"weight_heatmap_{li}")

    # Weight stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"<div style='font-family:var(--mono);font-size:10px;color:var(--muted)'>Max weight</div>"
            f"<div style='font-family:var(--mono);font-size:14px;color:var(--cyan)'>{W.max():.4f}</div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"<div style='font-family:var(--mono);font-size:10px;color:var(--muted)'>Min weight</div>"
            f"<div style='font-family:var(--mono);font-size:14px;color:var(--red)'>{W.min():.4f}</div>",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"<div style='font-family:var(--mono);font-size:10px;color:var(--muted)'>Std dev</div>"
            f"<div style='font-family:var(--mono);font-size:14px;color:var(--purple)'>{W.std():.4f}</div>",
            unsafe_allow_html=True,
        )

    # Weight magnitude bar chart (importance)
    importance = np.abs(W).mean(axis=1)
    max_imp = importance.max() + 1e-8
    bars = importance / max_imp

    bar_html = "<div style='margin-top:10px'>"
    bar_html += f"<div style='font-family:var(--mono);font-size:9px;color:var(--muted);letter-spacing:1px;margin-bottom:6px'>INPUT FEATURE IMPORTANCE (L1 norm)</div>"
    bar_html += "<div style='display:flex;align-items:flex-end;gap:4px;height:40px'>"
    for i, b in enumerate(bars):
        h = int(b * 36) + 4
        bar_html += f"<div style='display:flex;flex-direction:column;align-items:center;gap:2px'>"
        bar_html += f"<div style='width:20px;height:{h}px;background:var(--purple);border-radius:2px 2px 0 0;opacity:0.8'></div>"
        bar_html += f"<div style='font-family:var(--mono);font-size:8px;color:var(--muted)'>x{i+1}</div>"
        bar_html += "</div>"
    bar_html += "</div></div>"
    st.markdown(bar_html, unsafe_allow_html=True)
