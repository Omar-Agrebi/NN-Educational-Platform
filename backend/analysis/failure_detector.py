import numpy as np
from typing import List
from models.responses import FailureReport


def analyze(
    train_losses: List[float],
    test_losses: List[float],
    train_accs: List[float],
    test_accs: List[float],
) -> FailureReport:
    if not train_losses:
        return FailureReport()

    n = len(train_losses)
    flags = []
    severity = 0.0
    overfit_epoch = None
    best_test_epoch = int(np.argmin(test_losses)) + 1

    final_train = train_losses[-1]
    final_test = test_losses[-1]
    final_train_acc = train_accs[-1]
    final_test_acc = test_accs[-1]

    # Underfitting: train accuracy still low
    is_underfitting = False
    if final_train_acc < 0.65 and n >= 20:
        is_underfitting = True
        flags.append("underfitting")
        severity = max(severity, 0.6 + (0.65 - final_train_acc))

    # Overfitting: significant gap
    is_overfitting = False
    gap = final_test_acc - final_train_acc  # negative means train > test
    loss_gap = final_test - final_train
    if loss_gap > 0.1 or (final_train_acc - final_test_acc) > 0.08:
        is_overfitting = True
        flags.append("overfitting")
        sev = min(1.0, abs(loss_gap) / 0.3)
        severity = max(severity, sev)
        # Find epoch where overfit started
        for i in range(1, n):
            if test_losses[i] > test_losses[i - 1] and train_losses[i] < train_losses[i - 1]:
                overfit_epoch = i + 1
                break

    # Diverging: loss increasing over time
    is_diverging = False
    if n >= 10:
        recent = train_losses[max(0, n - 10):]
        if np.polyfit(range(len(recent)), recent, 1)[0] > 0.005:
            is_diverging = True
            flags.append("diverging")
            severity = max(severity, 0.85)

        # Oscillation check
        diffs = np.diff(train_losses[-min(20, n):])
        if np.std(diffs) > 0.05 and n > 20:
            flags.append("oscillating")
            severity = max(severity, 0.7)
            is_diverging = True

    # Stalled: loss barely changed
    is_stalled = False
    if n >= 20:
        window = train_losses[-20:]
        change = max(window) - min(window)
        if change < 0.005 and final_train_acc < 0.85:
            is_stalled = True
            flags.append("stalled")
            severity = max(severity, 0.5)

    # High LR sign: loss NaN or very high
    if any(np.isnan(l) or l > 10 for l in train_losses[-5:]):
        flags.append("exploding_loss")
        severity = 1.0
        is_diverging = True

    return FailureReport(
        flags=flags,
        severity=round(severity, 3),
        is_overfitting=is_overfitting,
        is_underfitting=is_underfitting,
        is_diverging=is_diverging,
        is_stalled=is_stalled,
        overfit_epoch=overfit_epoch,
        best_test_loss_epoch=best_test_epoch,
    )
