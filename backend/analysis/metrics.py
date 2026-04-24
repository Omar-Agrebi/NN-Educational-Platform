import numpy as np
from typing import List


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of correct predictions."""
    return float(np.mean(y_true.flatten() == y_pred.flatten()))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    2x2 confusion matrix for binary classification.
    Returns [[TN, FP], [FN, TP]]
    """
    y_t = y_true.flatten().astype(int)
    y_p = y_pred.flatten().astype(int)
    tn = int(np.sum((y_t == 0) & (y_p == 0)))
    fp = int(np.sum((y_t == 0) & (y_p == 1)))
    fn = int(np.sum((y_t == 1) & (y_p == 0)))
    tp = int(np.sum((y_t == 1) & (y_p == 1)))
    return np.array([[tn, fp], [fn, tp]])


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """TP / (TP + FP). Returns 0.0 if no positive predictions."""
    cm = confusion_matrix(y_true, y_pred)
    tp = cm[1, 1]
    fp = cm[0, 1]
    denom = tp + fp
    return float(tp / denom) if denom > 0 else 0.0


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """TP / (TP + FN). Returns 0.0 if no actual positives."""
    cm = confusion_matrix(y_true, y_pred)
    tp = cm[1, 1]
    fn = cm[1, 0]
    denom = tp + fn
    return float(tp / denom) if denom > 0 else 0.0


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Harmonic mean of precision and recall."""
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    denom = p + r
    return float(2 * p * r / denom) if denom > 0 else 0.0


def generalization_gap(train_metric: float, test_metric: float) -> float:
    """
    Signed gap between train and test metric.
    Positive means train > test (overfitting signal).
    """
    return float(train_metric - test_metric)
