def fmt_pct(val: float) -> str:
    if val is None:
        return "—"
    return f"{val * 100:.1f}%"


def fmt_loss(val: float) -> str:
    if val is None:
        return "—"
    return f"{val:.4f}"


def fmt_xp(val: int) -> str:
    return f"+{val} XP"


LEVEL_COLORS = {
    1: "#00f5ff",
    2: "#8b5cf6",
    3: "#00ff88",
    4: "#ff6b2b",
    5: "#ff3b5c",
    6: "#ff6b2b",
}


def level_color(level_id: int) -> str:
    return LEVEL_COLORS.get(level_id, "#00f5ff")


METRIC_LABELS = {
    "accuracy": "Accuracy",
    "f1_score": "F1 Score",
    "precision": "Precision",
    "recall": "Recall",
    "gen_gap": "Generalization Gap",
    "train_acc": "Train Accuracy",
    "test_acc": "Test Accuracy",
    "train_loss": "Train Loss",
    "test_loss": "Test Loss",
}


def metric_label(metric_name: str) -> str:
    return METRIC_LABELS.get(metric_name, metric_name.replace("_", " ").title())


FAILURE_ICONS = {
    "overfitting": "📈",
    "underfitting": "📉",
    "diverging": "🔥",
    "oscillating": "〰️",
    "stalled": "⏸",
    "exploding_loss": "💥",
}


def failure_icon(flag: str) -> str:
    return FAILURE_ICONS.get(flag, "⚠️")


INSIGHT_MESSAGES = {
    "xor_failure_witnessed": ("🪙 INSIGHT UNLOCKED", "The XOR Problem — a straight line cannot separate non-linear patterns."),
    "overfit_witnessed": ("🪙 INSIGHT UNLOCKED", "The Generalization Gap — your model memorized noise instead of learning."),
    "regularization_used": ("🪙 INSIGHT UNLOCKED", "Regularization Works — penalizing complexity forces true generalization."),
    "duel_won": ("🏆 MASTER BADGE", "You Are The Architect — you understand neural networks from the inside."),
}


def insight_message(insight_key: str):
    return INSIGHT_MESSAGES.get(insight_key, ("🪙 INSIGHT", "New discovery unlocked."))


ACTIVATION_DESCRIPTIONS = {
    "relu": "ReLU — fast, avoids vanishing gradients. Best default for hidden layers.",
    "sigmoid": "Sigmoid — squashes to (0,1). Can cause vanishing gradients in deep nets.",
    "tanh": "Tanh — squashes to (-1,1). Better than sigmoid for hidden layers.",
    "linear": "Linear — no non-linearity. Only useful for output layer in regression.",
    "leaky_relu": "Leaky ReLU — like ReLU but allows small negative values. Avoids dead neurons.",
    "elu": "ELU — smooth negative region. Often converges faster than ReLU.",
}


def activation_tip(act: str) -> str:
    return ACTIVATION_DESCRIPTIONS.get(act, "")
