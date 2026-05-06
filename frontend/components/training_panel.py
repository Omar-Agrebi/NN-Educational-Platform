import streamlit as st
from utils.state_manager import is_feature_unlocked
from utils.formatters import activation_tip
from styles.theme import card_title

ACTIVATIONS = ["relu", "sigmoid", "tanh", "leaky_relu", "elu", "linear"]

LEVEL_TIPS = {
    1: [
        "💡 Default LR=0.001 is intentionally slow — try increasing to 0.01.",
        "💡 Watch the loss curve: if it's flat, raise the learning rate.",
        "💡 Linear data is easy — a perceptron (0 hidden layers) is enough.",
        "💡 Enable Slow Mode to see the boundary evolve epoch by epoch.",
    ],
    2: [
        "💡 First, run with 0 hidden layers — watch it fail on XOR.",
        "💡 Add 1 hidden layer with 4-8 neurons to break linearity.",
        "💡 ReLU activation often converges faster than Sigmoid.",
        "💡 XOR needs at least 1 hidden layer — this is a mathematical fact.",
        "💡 Try 2 layers × 8 neurons for reliable >90% accuracy.",
    ],
    3: [
        "💡 Run 1-layer baseline first, note the accuracy.",
        "💡 Then try 3 layers and compare — the gain should be visible.",
        "💡 Too many layers can hurt (vanishing gradients) — try Tanh or ReLU.",
        "💡 More depth ≠ always better. Find the sweet spot.",
    ],
    4: [
        "💡 First train WITHOUT regularization — watch the gap grow.",
        "💡 Start with L2, λ=0.01 — a safe default.",
        "💡 If accuracy drops too much, lower λ. If gap stays large, raise it.",
        "💡 Early stopping is another powerful anti-overfitting tool.",
        "💡 The goal is to close the train/test gap to <5%.",
    ],
    5: [
        "💡 Check accuracy first — it's probably >80% already.",
        "💡 Then check F1 — if it's low, the model is ignoring the minority class.",
        "💡 Look at the confusion matrix: you need true positives, not just true negatives.",
        "💡 Try different architectures — this dataset rewards moderate complexity.",
    ],
    6: [
        "💡 No hints — this is the final boss.",
        "💡 The opponent uses LR=0.1, no hidden layers.",
        "💡 Choose your dataset and architecture wisely.",
    ],
}


def render_tips_button(level_id: int):
    tips = LEVEL_TIPS.get(level_id, [])
    if not tips:
        return
    with st.expander("💡 Stuck? Get a tip", expanded=False):
        st.markdown("""
<div class="tips-card">
""", unsafe_allow_html=True)
        for tip in tips:
            st.markdown(f"<div style='font-size:13px;color:var(--text);margin-bottom:8px;line-height:1.5'>{tip}</div>",
                        unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


def render_training_panel(level_id: int, default_config: dict = None) -> dict:
    """
    Renders hyperparameter controls based on unlocked features.
    Returns a dict with model_config and train_config.
    """
    if default_config is None:
        default_config = {}

    card_title("Hyperparameters")

    # ── Always visible ──────────────────────────────────────
    lr = st.slider(
        "Learning Rate",
        min_value=0.0001, max_value=1.0,
        value=float(default_config.get("learning_rate", 0.01)),
        step=0.0001, format="%.4f",
        help="Controls step size during gradient descent. Too high → diverge. Too low → slow."
    )

    epochs = st.slider(
        "Epochs",
        min_value=10, max_value=1000,
        value=int(default_config.get("epochs", 100)),
        step=10,
        help="Number of full passes over training data."
    )

    slow_mode = st.checkbox(
        "🐢 Slow Mode",
        value=False,
        help="Visualize training epoch by epoch. Great for understanding instability."
    )

    st.markdown("<hr style='margin:10px 0;border-color:var(--border)'>", unsafe_allow_html=True)

    # ── Hidden layers (Level 2+) ─────────────────────────────
    n_hidden = 0
    neurons_per_layer = []
    activations_per_layer = []
    activation_global = "relu"

    if is_feature_unlocked("hidden_layers"):
        n_hidden = st.slider(
            "Hidden Layers",
            min_value=0, max_value=6,
            value=int(default_config.get("hidden_layers", 1)),
            help="Number of hidden layers. More layers = more expressive power (but harder to train)."
        )

        if n_hidden > 0:
            if is_feature_unlocked("neurons_per_layer"):
                st.markdown("<div style='font-family:var(--mono);font-size:9px;color:var(--muted);letter-spacing:1px;margin-bottom:6px'>NEURONS PER LAYER</div>", unsafe_allow_html=True)
                for i in range(n_hidden):
                    n = st.slider(
                        f"Layer {i+1} neurons",
                        min_value=2, max_value=64,
                        value=int(default_config.get("neurons_per_layer", [8] * n_hidden)[i] if i < len(default_config.get("neurons_per_layer", [])) else 8),
                        key=f"neurons_layer_{i}_{level_id}",
                        help=f"Number of neurons in hidden layer {i+1}."
                    )
                    neurons_per_layer.append(n)
            else:
                neurons_per_layer = [8] * n_hidden
                st.markdown(f"<div class='locked-card'>Neurons/Layer — <em>unlock at higher level</em></div>", unsafe_allow_html=True)

            # ── Per-layer activations (KEY NEW FEATURE) ─────
            if is_feature_unlocked("activation"):
                st.markdown("<div style='font-family:var(--mono);font-size:9px;color:var(--muted);letter-spacing:1px;margin:8px 0 6px'>ACTIVATION PER LAYER</div>", unsafe_allow_html=True)
                for i in range(n_hidden):
                    col_act, col_tip = st.columns([2, 3])
                    with col_act:
                        default_act = "relu"
                        if i < len(default_config.get("activations_per_layer", [])):
                            default_act = default_config["activations_per_layer"][i]
                        act = st.selectbox(
                            f"Layer {i+1}",
                            options=ACTIVATIONS,
                            index=ACTIVATIONS.index(default_act) if default_act in ACTIVATIONS else 0,
                            key=f"act_layer_{i}_{level_id}",
                        )
                        activations_per_layer.append(act)
                    with col_tip:
                        st.markdown(f"<div style='font-size:10px;color:var(--muted);padding-top:28px;line-height:1.3'>{activation_tip(act).split('—')[1].strip() if '—' in activation_tip(act) else ''}</div>", unsafe_allow_html=True)
                activation_global = activations_per_layer[0] if activations_per_layer else "relu"
            else:
                activations_per_layer = ["relu"] * n_hidden
                st.markdown("<div class='locked-card'>Activation — <em>unlock at Level 2</em></div>", unsafe_allow_html=True)
        else:
            neurons_per_layer = []
            activations_per_layer = []
    else:
        st.markdown("<div class='locked-card'>🔒 Hidden Layers — unlock at Level 2</div>", unsafe_allow_html=True)
        st.markdown("<div class='locked-card'>🔒 Neurons/Layer — unlock at Level 2</div>", unsafe_allow_html=True)
        st.markdown("<div class='locked-card'>🔒 Activation — unlock at Level 2</div>", unsafe_allow_html=True)

    st.markdown("<hr style='margin:10px 0;border-color:var(--border)'>", unsafe_allow_html=True)

    # ── Regularization (Level 4+) ────────────────────────────
    reg_type = "none"
    reg_lambda = 0.0

    if is_feature_unlocked("regularization"):
        reg_type = st.selectbox(
            "Regularization",
            options=["none", "l1", "l2"],
            index=["none", "l1", "l2"].index(default_config.get("regularization", "none")),
            help="Penalizes large weights to prevent overfitting."
        )
        if reg_type != "none":
            reg_lambda = st.slider(
                "λ (reg strength)",
                min_value=0.0001, max_value=0.5,
                value=float(default_config.get("reg_lambda", 0.01)),
                step=0.0001, format="%.4f",
                help="Higher λ = stronger regularization. Too high → underfitting."
            )
    else:
        st.markdown("<div class='locked-card'>🔒 Regularization — unlock at Level 4</div>", unsafe_allow_html=True)

    # ── Batch size (Level 4+) ────────────────────────────────
    batch_size = 32
    if is_feature_unlocked("batch_size"):
        batch_size = st.select_slider(
            "Batch Size",
            options=[8, 16, 32, 64, 128, 256],
            value=32,
            help="Smaller batches = noisy but fast updates. Larger = stable but slow."
        )
    else:
        st.markdown("<div class='locked-card'>🔒 Batch Size — unlock at Level 4</div>", unsafe_allow_html=True)

    # ── Early stopping (Level 4+) ───────────────────────────
    early_stopping = False
    patience = 10
    if is_feature_unlocked("early_stopping"):
        early_stopping = st.checkbox("Early Stopping", value=False, help="Stop training when validation loss stops improving.")
        if early_stopping:
            patience = st.slider("Patience", 5, 50, 10, help="Epochs to wait before stopping.")

    # ── Tips button ──────────────────────────────────────────
    render_tips_button(level_id)

    model_config = {
        "hidden_layers": n_hidden,
        "neurons_per_layer": neurons_per_layer,
        "activations_per_layer": activations_per_layer,
        "activation": activation_global,
        "regularization": reg_type,
        "reg_lambda": reg_lambda,
        "dropout_rate": 0.0,
    }

    train_config = {
        "learning_rate": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "slow_mode": slow_mode,
        "early_stopping": early_stopping,
        "patience": patience,
        "momentum": 0.9,
        "gradient_clip": 5.0,
    }

    return model_config, train_config
