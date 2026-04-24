import streamlit as st
import plotly.graph_objects as go
import math
from utils.formatters import level_color


def _locked_param(label: str, unlock_hint: str = ""):
    st.markdown(f"""
    <div style="background:#0d1117;border:1px solid #1f2d48;border-radius:6px;
         padding:8px 12px;margin:6px 0;opacity:0.5;cursor:not-allowed;">
      <span style="color:#4a5568;font-size:0.78rem;">🔒 {label}</span>
      {"<br><span style='color:#4a5568;font-size:0.68rem;'>" + unlock_hint + "</span>" if unlock_hint else ""}
    </div>
    """, unsafe_allow_html=True)


def render_dna_chart(config: dict, level_id: int = 1):
    """Compact parameter fingerprint bar chart."""
    color = level_color(level_id)
    params = {
        "LR": min(math.log10(max(config.get("learning_rate", 0.01), 1e-6)) / math.log10(10) + 3, 1.0),
        "Ep": min(config.get("epochs", 100) / 500, 1.0),
        "Layers": min(config.get("hidden_layers", 0) / 5, 1.0),
        "N": min(config.get("neurons_per_layer", 8) / 64, 1.0),
        "λ": min(config.get("reg_lambda", 0.0) / 0.1, 1.0),
    }
    fig = go.Figure(go.Bar(
        x=list(params.keys()),
        y=list(params.values()),
        marker_color=color,
        marker_line_color="#0a0f1e",
        marker_line_width=1,
    ))
    fig.update_layout(
        plot_bgcolor="#0a0f1e",
        paper_bgcolor="#111827",
        margin=dict(l=5, r=5, t=5, b=20),
        height=60,
        xaxis=dict(visible=True, tickfont=dict(size=8, color="#4a5568", family="monospace"),
                   showgrid=False, zeroline=False),
        yaxis=dict(visible=False, range=[0, 1.1]),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, key="dna_chart")


def render_training_panel(level_config: dict, unlocked_features: list,
                           level_id: int = 1) -> dict:
    """
    Render hyperparameter controls. Returns config dict ready for API.
    """
    default = level_config.get("default_config", {})
    color = level_color(level_id)

    st.markdown(f'<div class="section-title" style="color:{color};">HYPERPARAMETERS</div>',
                unsafe_allow_html=True)

    # ── Learning Rate (always shown) ────────────────────────────────────
    lr_log = st.slider("Learning Rate (log scale)", -4.0, 0.0,
                       value=math.log10(default.get("learning_rate", 0.01)),
                       step=0.25, key=f"lr_log_{level_id}")
    lr = 10 ** lr_log
    st.caption(f"LR = {lr:.5f}")

    # ── Epochs (always shown) ────────────────────────────────────────────
    epochs = st.slider("Epochs", 10, 500,
                       value=default.get("epochs", 100),
                       step=10, key=f"epochs_{level_id}")

    # ── Slow mode (always shown) ─────────────────────────────────────────
    slow_mode = st.toggle("🐢 Slow Mode (live updates)", value=False,
                          key=f"slow_{level_id}")

    # ── Hidden Layers ────────────────────────────────────────────────────
    hidden_layers = default.get("hidden_layers", 0)
    if "hidden_layers" in unlocked_features:
        hidden_layers = st.slider("Hidden Layers", 0, 6,
                                  value=hidden_layers,
                                  key=f"hl_{level_id}")
        st.caption("More layers = more expressive but slower to train")
    else:
        _locked_param("Hidden Layers", "Unlock at Level 2")

    # ── Neurons per Layer ─────────────────────────────────────────────────
    neurons = default.get("neurons_per_layer", 8)
    if "hidden_layers" in unlocked_features and hidden_layers > 0:
        neurons = st.slider("Neurons per Layer", 2, 64,
                             value=neurons, step=2, key=f"neurons_{level_id}")
        st.caption("Width of each hidden layer")
    elif "hidden_layers" not in unlocked_features:
        _locked_param("Neurons per Layer", "Unlock at Level 2")

    # ── Activation ────────────────────────────────────────────────────────
    activation = default.get("activation", "relu")
    if "activation" in unlocked_features:
        activation = st.selectbox("Activation",
                                   ["relu", "sigmoid", "tanh", "linear"],
                                   index=["relu","sigmoid","tanh","linear"].index(activation),
                                   key=f"act_{level_id}")
        st.caption("Non-linearity applied after each hidden layer")
    else:
        _locked_param("Activation Function", "Unlock at Level 2")

    # ── Regularization ────────────────────────────────────────────────────
    regularization = default.get("regularization", "none")
    reg_lambda = default.get("reg_lambda", 0.001)
    if "regularization" in unlocked_features:
        regularization = st.selectbox("Regularization",
                                       ["none", "l2", "l1"],
                                       index=["none","l2","l1"].index(regularization),
                                       key=f"reg_{level_id}")
        if regularization != "none":
            reg_lambda = st.slider("λ (strength)", 0.0001, 0.1,
                                    value=reg_lambda, step=0.001,
                                    format="%.4f", key=f"lambda_{level_id}")
            st.caption("Higher λ = stronger regularization (less overfitting, more underfitting risk)")
    else:
        _locked_param("Regularization", "Unlock at Level 4")

    # ── Batch Size ────────────────────────────────────────────────────────
    batch_size = default.get("batch_size", 32)
    if "batch_size" in unlocked_features:
        batch_size = st.select_slider("Batch Size", [8, 16, 32, 64, 128, 256],
                                       value=batch_size, key=f"bs_{level_id}")
        st.caption("Smaller = more noise in updates; larger = smoother")

    config = {
        "learning_rate": lr,
        "epochs": epochs,
        "hidden_layers": hidden_layers,
        "neurons_per_layer": neurons,
        "activation": activation,
        "regularization": regularization,
        "reg_lambda": reg_lambda,
        "batch_size": batch_size,
        "slow_mode": slow_mode,
    }
    return config
