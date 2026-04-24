from dataclasses import dataclass
from typing import List
from analysis.failure_detector import FailureReport


@dataclass
class ExplanationCard:
    title: str
    body: str
    action_hint: str


# Explanation templates keyed by failure flag
_EXPLANATIONS = {
    "underfitting": ExplanationCard(
        title="⚠️ Underfitting Detected",
        body="Your model is too simple or undertrained to learn the pattern. Train accuracy is below the minimum threshold.",
        action_hint="Try adding hidden layers, increasing neurons, training for more epochs, or raising the learning rate.",
    ),
    "overfitting": ExplanationCard(
        title="🔥 Overfitting Detected",
        body="Your model memorized the training data but fails to generalize. Train accuracy is much higher than test accuracy.",
        action_hint="Add L2 regularization, reduce model depth/neurons, or gather more training data.",
    ),
    "diverging": ExplanationCard(
        title="💥 Training Diverging",
        body="Your loss is increasing or oscillating wildly. The optimizer is taking steps that are too large.",
        action_hint="Lower the learning rate significantly (try dividing by 10). Check for NaN values in loss.",
    ),
    "stalled": ExplanationCard(
        title="🧊 Training Stalled",
        body="Loss stopped decreasing — the optimizer is stuck in a plateau or local minimum.",
        action_hint="Try a higher learning rate, add momentum, or change the activation function.",
    ),
    "healthy": ExplanationCard(
        title="✅ Training Looks Healthy",
        body="No major failure modes detected. Train and test metrics are converging nicely.",
        action_hint="Consider training for more epochs or tuning the learning rate for further improvement.",
    ),
}


def explain(report: FailureReport) -> List[ExplanationCard]:
    """
    Map a FailureReport to actionable ExplanationCards.
    Returns one card per detected issue (max 2 sentences per card).
    Falls back to 'healthy' card if no issues found.
    """
    cards = []
    for flag in report.flags:
        if flag in _EXPLANATIONS:
            cards.append(_EXPLANATIONS[flag])
    if not cards:
        cards.append(_EXPLANATIONS["healthy"])
    return cards
