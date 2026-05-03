import numpy as np
from typing import Tuple


def softmax(x: np.ndarray) -> np.ndarray:
    x_shifted = x - x.max(axis=1, keepdims=True)
    exp_x = np.exp(np.clip(x_shifted, -500, 0))
    return exp_x / (exp_x.sum(axis=1, keepdims=True) + 1e-15)


def categorical_crossentropy(y_true_onehot: np.ndarray, y_pred: np.ndarray) -> Tuple[float, np.ndarray]:
    y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
    loss = -np.mean(np.sum(y_true_onehot * np.log(y_pred_clipped), axis=1))
    grad = (y_pred_clipped - y_true_onehot) / len(y_true_onehot)
    return float(loss), grad


def to_onehot(y: np.ndarray, n_classes: int) -> np.ndarray:
    y = y.astype(int)
    oh = np.zeros((len(y), n_classes))
    oh[np.arange(len(y)), y] = 1.0
    return oh


def mse_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, np.ndarray]:
    diff = y_pred - y_true
    loss = float(np.mean(diff ** 2))
    grad = 2.0 * diff / len(y_true)
    return loss, grad


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-10))
