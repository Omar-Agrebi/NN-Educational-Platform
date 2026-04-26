from models.responses import FailureReport, ExplanationCard
from typing import List

EXPLANATIONS = {
    "overfitting": ExplanationCard(
        title="⚠ Model Is Overfitting",
        body="Your model memorized the training data including its noise. The test loss is much higher than train loss — it can't generalize.",
        action_hint="Try adding L2 regularization (λ=0.001–0.01) or reducing network size.",
        flag_type="overfitting",
    ),
    "underfitting": ExplanationCard(
        title="⚠ Model Is Underfitting",
        body="Train accuracy is too low — the model lacks the capacity or training time to learn the pattern.",
        action_hint="Add hidden layers, increase neurons, raise learning rate, or train for more epochs.",
        flag_type="underfitting",
    ),
    "diverging": ExplanationCard(
        title="🔥 Loss Is Diverging",
        body="Training loss is increasing — a classic sign of too-high learning rate. Gradients are exploding.",
        action_hint="Lower your learning rate by 10× (e.g. 0.01 → 0.001).",
        flag_type="diverging",
    ),
    "oscillating": ExplanationCard(
        title="📉 Loss Is Oscillating",
        body="The loss jumps up and down instead of decreasing smoothly — the learning rate is too aggressive.",
        action_hint="Reduce learning rate by 3–5× or enable momentum.",
        flag_type="oscillating",
    ),
    "stalled": ExplanationCard(
        title="⏸ Training Has Stalled",
        body="Loss hasn't changed in many epochs — the model is stuck in a flat region or local minimum.",
        action_hint="Try a different activation (ReLU often escapes flat regions), or increase learning rate slightly.",
        flag_type="stalled",
    ),
    "exploding_loss": ExplanationCard(
        title="💥 Exploding Loss",
        body="Loss reached extreme values — gradients exploded and destroyed the weights.",
        action_hint="Lower learning rate drastically (try 0.0001) and enable gradient clipping.",
        flag_type="exploding_loss",
    ),
}


def explain(failure_report: FailureReport) -> List[ExplanationCard]:
    cards = []
    for flag in failure_report.flags:
        if flag in EXPLANATIONS:
            cards.append(EXPLANATIONS[flag])
    return cards
