import numpy as np
from typing import List
from dataclasses import dataclass, field


@dataclass
class FailureReport:
    is_underfitting: bool
    is_overfitting: bool
    is_diverging: bool
    is_stalled: bool
    severity: float  # 0.0 – 1.0
    flags: List[str] = field(default_factory=list)


def analyze(
    train_losses: List[float],
    test_losses: List[float],
    train_accs: List[float],
    test_accs: List[float],
    underfitting_acc_threshold: float = 0.70,
    overfitting_gap_threshold: float = 0.05,
    stall_window: int = 10,
    stall_delta: float = 1e-4,
    diverge_window: int = 5,
) -> FailureReport:
    """
    Stateless analysis of training curves.
    Returns FailureReport describing detected failure modes.
    """
    flags = []
    severity_scores = []

    if len(train_losses) == 0:
        return FailureReport(False, False, False, False, 0.0, ["no_data"])

    train_arr = np.array(train_losses)
    test_arr = np.array(test_losses)
    train_acc_arr = np.array(train_accs)
    test_acc_arr = np.array(test_accs)

    # ── UNDERFITTING ─────────────────────────────────────────────────────────
    final_train_acc = float(train_acc_arr[-1])
    is_underfitting = final_train_acc < underfitting_acc_threshold

    if is_underfitting:
        flags.append("underfitting")
        under_severity = 1.0 - (final_train_acc / underfitting_acc_threshold)
        severity_scores.append(min(under_severity, 1.0))

    # ── OVERFITTING ──────────────────────────────────────────────────────────
    final_train_loss = float(train_arr[-1])
    final_test_loss = float(test_arr[-1])
    loss_gap = final_test_loss - final_train_loss
    is_overfitting = loss_gap > overfitting_gap_threshold

    acc_gap = float(train_acc_arr[-1]) - float(test_acc_arr[-1])
    if acc_gap > 0.10:
        is_overfitting = True

    if is_overfitting:
        flags.append("overfitting")
        over_severity = min(loss_gap / (overfitting_gap_threshold * 4), 1.0)
        severity_scores.append(over_severity)

    # ── DIVERGING ────────────────────────────────────────────────────────────
    is_diverging = False
    if len(train_arr) >= diverge_window:
        recent = train_arr[-diverge_window:]
        # Loss increasing trend
        increasing = np.all(np.diff(recent) > 0)
        # Oscillating (high variance in recent window)
        oscillating = float(np.std(recent)) > 0.5 * float(np.mean(np.abs(recent)))
        # NaN/Inf
        has_nan = not np.all(np.isfinite(train_arr))
        is_diverging = increasing or oscillating or has_nan
        if is_diverging:
            flags.append("diverging")
            severity_scores.append(0.8)

    # ── STALLED ──────────────────────────────────────────────────────────────
    is_stalled = False
    if len(train_arr) >= stall_window:
        recent_loss = train_arr[-stall_window:]
        loss_change = float(np.max(recent_loss) - np.min(recent_loss))
        is_stalled = loss_change < stall_delta
        if is_stalled:
            flags.append("stalled")
            severity_scores.append(0.4)

    # ── OVERALL SEVERITY ─────────────────────────────────────────────────────
    overall_severity = float(np.max(severity_scores)) if severity_scores else 0.0

    return FailureReport(
        is_underfitting=is_underfitting,
        is_overfitting=is_overfitting,
        is_diverging=is_diverging,
        is_stalled=is_stalled,
        severity=round(overall_severity, 3),
        flags=flags,
    )
