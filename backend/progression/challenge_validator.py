from typing import Dict, Any, List
from progression.level_manager import get_level
from models.responses import ChallengeResult


def validate(level_id: int, training_result: Dict[str, Any], baseline_result: Dict[str, Any] = None) -> ChallengeResult:
    level = get_level(level_id)
    challenge = level["challenge"]
    ctype = challenge["type"]
    threshold = challenge["target_value"]
    xp_reward = level["xp_reward"]

    history = training_result.get("history", {})
    metrics = training_result.get("final_metrics", {})

    train_accs = history.get("train_acc", [])
    test_accs = history.get("test_acc", [])
    final_train_acc = train_accs[-1] if train_accs else 0.0
    final_test_acc = test_accs[-1] if test_accs else 0.0

    score = 0.0
    passed = False
    message = ""
    partial = 0.0

    if ctype in ("train_accuracy", "test_accuracy"):
        score = final_test_acc
        passed = score >= threshold
        partial = min(1.0, score / threshold)
        if passed:
            message = f"✓ Challenge complete! Test accuracy: {score:.1%}"
        else:
            message = f"Not yet — {score:.1%} achieved, need {threshold:.0%}. Keep iterating."

    elif ctype == "generalization_gap":
        gen_gap = abs(final_train_acc - final_test_acc)
        score = max(0.0, 1.0 - gen_gap / threshold) if threshold > 0 else 0.0
        passed = gen_gap <= threshold
        partial = 1.0 - min(1.0, gen_gap / (threshold * 2))
        if passed:
            message = f"✓ Gap reduced to {gen_gap:.1%}! Regularization mastered."
        else:
            message = f"Gap is {gen_gap:.1%}, need <{threshold:.0%}. Try more regularization."

    elif ctype == "f1_score":
        f1 = metrics.get("f1_score", 0.0)
        score = f1
        passed = f1 >= threshold
        partial = min(1.0, f1 / threshold)
        if passed:
            message = f"✓ F1 Score: {f1:.3f}! You understand the metric."
        else:
            message = f"F1: {f1:.3f}, need >{threshold}. Check your confusion matrix."

    elif ctype == "accuracy_gain":
        baseline_acc = 0.0
        if baseline_result:
            b_accs = baseline_result.get("history", {}).get("test_acc", [])
            baseline_acc = b_accs[-1] if b_accs else 0.0
        gain = final_test_acc - baseline_acc
        score = gain
        passed = gain >= threshold
        partial = min(1.0, max(0.0, gain / threshold))
        if passed:
            message = f"✓ Accuracy gain: +{gain:.1%}! Depth makes a difference."
        else:
            message = f"Gain: +{gain:.1%}, need +{threshold:.0%}. Add more layers."

    elif ctype == "duel":
        wins = training_result.get("duel_wins", 0)
        passed = wins >= int(threshold)
        score = float(wins)
        partial = wins / threshold
        if passed:
            message = "✓ VICTORY! You've mastered the forge. MASTER BADGE awarded."
        else:
            message = f"Won {wins}/2 metrics. Refine your architecture."

    xp_earned = xp_reward if passed else int(xp_reward * partial * 0.3)

    return ChallengeResult(
        passed=passed,
        score=round(score, 4),
        xp_earned=xp_earned,
        message=message,
        unlocked_features=level["unlock_reward"] if passed else [],
        is_gate_passed=passed,
        partial_credit=round(partial, 3),
    )
