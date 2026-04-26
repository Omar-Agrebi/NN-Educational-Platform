import streamlit as st
import numpy as np


def render_parameter_dna(model_config: dict, train_config: dict, color: str = "#8b5cf6"):
    """Compact visual fingerprint of current model config."""
    lr     = train_config.get("learning_rate", 0.01)
    epochs = train_config.get("epochs", 100)
    layers = model_config.get("hidden_layers", 0)
    neurons = np.mean(model_config.get("neurons_per_layer", [8])) if model_config.get("neurons_per_layer") else 0
    reg    = model_config.get("reg_lambda", 0.0)

    # Normalize each to 0-1
    lr_n  = np.clip(np.log10(lr + 1e-6) / np.log10(1.0) + 1, 0, 1)
    ep_n  = min(1.0, epochs / 1000)
    hl_n  = min(1.0, layers / 6)
    nn_n  = min(1.0, neurons / 64)
    reg_n = min(1.0, reg / 0.5)

    vals   = [lr_n, ep_n, hl_n, nn_n, reg_n]
    labels = ["LR", "EP", "HL", "NN", "Reg"]
    max_h  = 32

    bars_html = "<div style='display:flex;align-items:flex-end;gap:3px;height:{}px'>".format(max_h + 4)
    for v, lbl in zip(vals, labels):
        h = max(4, int(v * max_h))
        bars_html += (
            f"<div style='display:flex;flex-direction:column;align-items:center;gap:2px'>"
            f"<div style='width:18px;height:{h}px;background:{color};border-radius:2px 2px 0 0;opacity:0.75'></div>"
            f"<div style='font-family:var(--mono);font-size:7px;color:var(--muted)'>{lbl}</div>"
            f"</div>"
        )
    bars_html += "</div>"

    st.markdown(
        f"<div style='margin-top:4px'>"
        f"<div style='font-family:var(--mono);font-size:9px;color:var(--muted);letter-spacing:1px;margin-bottom:4px'>PARAMETER DNA</div>"
        f"{bars_html}"
        f"</div>",
        unsafe_allow_html=True,
    )
