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
    1: "#00f5ff",   # cyan
    2: "#8b5cf6",   # purple
    3: "#ff6b2b",   # orange
    4: "#ff3b5c",   # red
    5: "#00ff88",   # green
    6: "#f59e0b",   # gold
}


def level_color(level_id: int) -> str:
    return LEVEL_COLORS.get(level_id, "#00f5ff")


METRIC_LABELS = {
    "train_accuracy": "Train Accuracy",
    "test_accuracy":  "Test Accuracy",
    "f1_score":       "F1 Score",
    "precision":      "Precision",
    "recall":         "Recall",
    "train_loss":     "Train Loss",
    "test_loss":      "Test Loss",
    "generalization_gap": "Generalization Gap",
    "train_test_gap": "Train/Test Gap",
    "accuracy_gain":  "Accuracy Gain",
}


def metric_label(metric_name: str) -> str:
    return METRIC_LABELS.get(metric_name, metric_name.replace("_", " ").title())


FAILURE_ICONS = {
    "overfitting":   "🔥",
    "underfitting":  "❄️",
    "diverging":     "📉",
    "stalled":       "🔁",
    "no_data":       "⚠️",
}


def failure_icon(flag: str) -> str:
    return FAILURE_ICONS.get(flag, "⚠️")


def short_session_id(session_id: str) -> str:
    if not session_id:
        return "—"
    return f"{session_id[:8]}…"


def dataset_emoji(dataset: str) -> str:
    return {
        "linear":     "📏",
        "xor":        "⊕",
        "noisy":      "🌪️",
        "imbalanced": "⚖️",
    }.get(dataset, "📊")
