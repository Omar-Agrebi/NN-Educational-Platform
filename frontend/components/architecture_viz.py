import streamlit as st
import plotly.graph_objects as go
import math


def render_architecture(model_config: dict, key: str = "arch_viz"):
    hidden_layers = model_config.get("hidden_layers", 0)
    neurons = model_config.get("neurons_per_layer", 8)
    activation = model_config.get("activation", "relu")

    layer_defs = []
    layer_defs.append({"n": 2, "label": "Input", "color": "#00f5ff", "names": ["x₁", "x₂"]})
    for i in range(hidden_layers):
        display_n = min(neurons, 6)
        layer_defs.append({
            "n": display_n, "label": f"H{i+1} ({activation})",
            "color": "#8b5cf6", "names": [f"h{i+1}_{j+1}" for j in range(display_n)],
            "actual_n": neurons,
        })
    layer_defs.append({"n": 1, "label": "Output", "color": "#00ff88", "names": ["ŷ"]})

    n_layers = len(layer_defs)
    x_gap = 1.0 / (n_layers + 1)
    max_neurons = max(d["n"] for d in layer_defs)

    fig = go.Figure()

    node_positions = []
    for li, layer in enumerate(layer_defs):
        x = (li + 1) * x_gap
        n = layer["n"]
        ys = [(j + 1) / (n + 1) for j in range(n)]
        node_positions.append([(x, y) for y in ys])

    # Draw connections
    for li in range(len(node_positions) - 1):
        for (x1, y1) in node_positions[li]:
            for (x2, y2) in node_positions[li + 1]:
                fig.add_shape(type="line",
                    x0=x1, y0=y1, x1=x2, y1=y2,
                    line=dict(color="#1f2d48", width=0.8),
                    layer="below",
                )

    # Draw nodes
    for li, (layer, positions) in enumerate(zip(layer_defs, node_positions)):
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        names = layer["names"]

        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers+text",
            marker=dict(size=22, color=layer["color"],
                        line=dict(color="#0a0f1e", width=2)),
            text=names,
            textposition="middle center",
            textfont=dict(size=8, color="#0a0f1e", family="monospace"),
            hoverinfo="none",
            showlegend=False,
        ))

        # Layer label below
        fig.add_annotation(
            x=(li + 1) * x_gap, y=-0.06,
            text=layer["label"],
            showarrow=False,
            font=dict(size=9, color="#4a5568", family="monospace"),
            xanchor="center",
        )

        # Neuron count badge if hidden layer with more than display
        if layer.get("actual_n") and layer["actual_n"] > layer["n"]:
            fig.add_annotation(
                x=(li + 1) * x_gap + 0.06, y=0.5,
                text=f"×{layer['actual_n']}",
                showarrow=False,
                font=dict(size=9, color="#8b5cf6", family="monospace"),
            )

    fig.update_layout(
        plot_bgcolor="#0a0f1e",
        paper_bgcolor="#111827",
        margin=dict(l=10, r=10, t=10, b=30),
        xaxis=dict(visible=False, range=[-0.05, 1.05]),
        yaxis=dict(visible=False, range=[-0.15, 1.1]),
        height=200,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, key=key)
