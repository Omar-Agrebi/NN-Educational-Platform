from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from progression.level_manager import ChallengeDefinition, get_level


@dataclass
class ChallengeResult:
    passed: bool
    is_gate_passed: bool
    score: float          # 0.0–1.0
    xp_earned: int
    message: str
    unlocked_features: List[str]


def _compare(value: float, operator: str, threshold: float) -> bool:
    ops = {
        "gt":  lambda v, t: v > t,
        "lt":  lambda v, t: v < t,
        "gte": lambda v, t: v >= t,
        "lte": lambda v, t: v <= t,
    }
    if operator not in ops:
        raise ValueError(f"Unknown operator '{operator}'")
    return ops[operator](value, threshold)


def _compute_score(value: float, operator: str, threshold: float) -> float:
    """
    Compute a 0.0–1.0 score based on how close the value is to the threshold.
    For "gt"/"gte": score = value / threshold (capped at 1.0).
    For "lt"/"lte": score = threshold / value (capped at 1.0, if value > 0).
    """
    if threshold == 0:
        return 1.0 if _compare(value, operator, threshold) else 0.0

    if operator in ("gt", "gte"):
        return min(value / threshold, 1.0)
    else:  # lt, lte
        if value == 0:
            return 1.0
        return min(threshold / value, 1.0)


def validate(
    challenge_def: ChallengeDefinition,
    training_result: Dict[str, Any],
    level_id: int,
    partial_credit_threshold: float = 0.80,
) -> ChallengeResult:
    """
    Validate a training result against the challenge definition.
    
    training_result must contain:
      - train_accuracy, test_accuracy (floats)
      - f1_score (float)
      - train_test_gap (float) — absolute gap between train_acc and test_acc
      - accuracy_gain (float, optional) — improvement over baseline
      - duel_win (bool, optional)
    
    Returns ChallengeResult with strict gate and partial credit logic.
    """
    level = get_level(level_id)
    metric = challenge_def.metric

    # Extract the metric value from result
    if metric == "train_accuracy":
        value = float(training_result.get("train_accuracy", 0.0))
    elif metric == "test_accuracy":
        value = float(training_result.get("test_accuracy", 0.0))
    elif metric == "f1_score":
        value = float(training_result.get("f1_score", 0.0))
    elif metric == "train_test_gap":
        # Gap = train_acc - test_acc (absolute). Challenge is to keep it small (lt).
        train_acc = float(training_result.get("train_accuracy", 0.0))
        test_acc = float(training_result.get("test_accuracy", 0.0))
        value = abs(train_acc - test_acc)
    elif metric == "accuracy_gain":
        value = float(training_result.get("accuracy_gain", 0.0))
    elif metric == "duel_win":
        # Special case: duel win requires beating opponent on both accuracy AND f1
        player_acc = float(training_result.get("test_accuracy", 0.0))
        player_f1 = float(training_result.get("f1_score", 0.0))
        opp_acc = float(training_result.get("opponent_test_accuracy", 0.0))
        opp_f1 = float(training_result.get("opponent_f1_score", 0.0))
        value = 1.0 if (player_acc > opp_acc and player_f1 > opp_f1) else 0.0
    else:
        value = 0.0

    # Compute pass/fail
    passed = _compare(value, challenge_def.operator, challenge_def.threshold)
    score = _compute_score(value, challenge_def.operator, challenge_def.threshold)

    # Partial credit: near-miss (score >= 80% of threshold)
    near_miss = (not passed) and (score >= partial_credit_threshold)

    # Strict gate: must fully pass to unlock features and advance
    is_gate_passed = passed

    # XP logic
    if passed:
        xp_earned = level.xp_reward
        message = (
            f"🏆 Challenge complete! You achieved {metric} = {round(value, 4)} "
            f"(required {challenge_def.operator} {challenge_def.threshold}). "
            f"Earned {xp_earned} XP!"
        )
        unlocked_features = level.unlock_reward
    elif near_miss:
        xp_earned = int(level.xp_reward * score)
        message = (
            f"⚡ So close! {metric} = {round(value, 4)} "
            f"(needed {challenge_def.operator} {challenge_def.threshold}). "
            f"Partial credit: {xp_earned} XP. Keep tweaking!"
        )
        unlocked_features = []
    else:
        xp_earned = int(level.xp_reward * score * 0.5)
        message = (
            f"❌ Challenge not passed. {metric} = {round(value, 4)} "
            f"(needed {challenge_def.operator} {challenge_def.threshold}). "
            f"Consolation: {xp_earned} XP. Read the explanation cards and try again."
        )
        unlocked_features = []

    return ChallengeResult(
        passed=passed,
        is_gate_passed=is_gate_passed,
        score=round(score, 4),
        xp_earned=xp_earned,
        message=message,
        unlocked_features=unlocked_features,
    )
