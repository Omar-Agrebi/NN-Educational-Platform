import streamlit as st
import plotly.graph_objects as go
from styles.theme import card_title


def render_architecture(model_config: dict, animate_pulse: bool = False, key: str = "arch"):
    card_title("Network Architecture")

    n_hidden = model_config.get("hidden_layers", 0)
    neurons  = model_config.get("neurons_per_layer", [])
    acts     = model_config.get("activations_per_layer", [])

    if len(neurons) < n_hidden:
        neurons = list(neurons) + [8] * (n_hidden - len(neurons))
    if len(acts) < n_hidden:
        acts = list(acts) + ["relu"] * (n_hidden - len(acts))

    layer_sizes = [2] + list(neurons[:n_hidden]) + [1]
    layer_labels = ["Input"] + [f"H{i+1}\n{acts[i].upper()}" for i in range(n_hidden)] + ["Output"]
    n_layers = len(layer_sizes)

    COLORS = {
        "input":  "#00f5ff",
        "hidden": "#8b5cf6",
        "output": "#00ff88",
    }

    max_neurons = max(layer_sizes)
    W, H = 500, max(160, max_neurons * 38)
    x_positions = [i / (n_layers - 1) * 0.85 + 0.075 for i in range(n_layers)] if n_layers > 1 else [0.5]

    node_x, node_y, node_color, node_text = [], [], [], []
    edge_x, edge_y = [], []

    neuron_coords = []  # list of list of (x,y) per layer
    for li, (n, lx) in enumerate(zip(layer_sizes, x_positions)):
        coords = []
        for ni in range(n):
            y = 0.5 + (ni - (n - 1) / 2) * (0.75 / max(max_neurons - 1, 1))
            coords.append((lx, y))
        neuron_coords.append(coords)

    # Draw edges
    for li in range(n_layers - 1):
        for (x1, y1) in neuron_coords[li]:
            for (x2, y2) in neuron_coords[li + 1]:
                edge_x += [x1, x2, None]
                edge_y += [y1, y2, None]

    # Draw nodes
    for li, coords in enumerate(neuron_coords):
        if li == 0:
            col = COLORS["input"]
        elif li == n_layers - 1:
            col = COLORS["output"]
        else:
            col = COLORS["hidden"]
        for ni, (nx, ny) in enumerate(coords):
            node_x.append(nx)
            node_y.append(ny)
            node_color.append(col)
            if li == 0:
                node_text.append(f"x{ni+1}")
            elif li == n_layers - 1:
                node_text.append("ŷ")
            else:
                node_text.append(f"N{ni+1}")

    fig = go.Figure()

    # Edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(color="#1f2d48", width=0.8),
        hoverinfo="none",
        showlegend=False,
    ))

    # Nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(
            size=22,
            color=node_color,
            opacity=0.85,
            line=dict(color="#0a0f1e", width=2),
        ),
        text=node_text,
        textposition="middle center",
        textfont=dict(size=8, color="#0a0f1e", family="Share Tech Mono"),
        hoverinfo="none",
        showlegend=False,
    ))

    # Layer labels below
    for li, (lx, label) in enumerate(zip(x_positions, layer_labels)):
        fig.add_annotation(
            x=lx, y=0.04,
            text=label.replace("\n", "<br>"),
            showarrow=False,
            font=dict(size=8, color="#4a5568", family="Share Tech Mono"),
            xref="paper", yref="paper",
        )
        # Neuron count
        fig.add_annotation(
            x=lx, y=0.96,
            text=str(layer_sizes[li]),
            showarrow=False,
            font=dict(size=9, color="#8b5cf6" if 0 < li < n_layers - 1 else "#4a5568", family="Share Tech Mono"),
            xref="paper", yref="paper",
        )

    fig.update_layout(
        paper_bgcolor="#111827",
        plot_bgcolor="#0a0f1e",
        margin=dict(l=8, r=8, t=8, b=8),
        height=max(130, n_hidden * 40 + 90),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.05, 1.05]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.05, 1.05]),
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True, key=key)

    # Summary label
    summary = " → ".join(str(s) for s in layer_sizes)
    act_str = " | ".join(acts[:n_hidden]) if acts else "—"
    st.markdown(
        f"<div style='font-family:var(--mono);font-size:9px;color:var(--muted);text-align:center;margin-top:-8px'>"
        f"{summary} &nbsp;·&nbsp; {act_str if n_hidden else 'no hidden layers'}</div>",
        unsafe_allow_html=True,
    )
