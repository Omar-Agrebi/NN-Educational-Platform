import numpy as np
from typing import List

EPS = 1e-8


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    return float(tp / (tp + fp + EPS))


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return float(tp / (tp + fn + EPS))


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return float(2 * p * r / (p + r + EPS))


def generalization_gap(train_metric: float, test_metric: float) -> float:
    return float(abs(train_metric - test_metric))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> List[List[int]]:
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    return [[tn, fp], [fn, tp]]


def compute_all_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray) -> dict:
    y_pred = (y_pred_prob >= 0.5).astype(int).flatten()
    y_true = y_true.flatten().astype(int)
    return {
        "accuracy": round(accuracy(y_true, y_pred), 4),
        "precision": round(precision(y_true, y_pred), 4),
        "recall": round(recall(y_true, y_pred), 4),
        "f1_score": round(f1_score(y_true, y_pred), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }
