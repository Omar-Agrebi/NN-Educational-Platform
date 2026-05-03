import numpy as np
from typing import Tuple

EPS = 1e-15


def binary_crossentropy(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, np.ndarray]:
    y_pred = np.clip(y_pred, EPS, 1 - EPS)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    grad = (y_pred - y_true) / (y_pred * (1 - y_pred) * len(y_true))
    return float(loss), grad


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, np.ndarray]:
    diff = y_pred - y_true
    loss = np.mean(diff ** 2)
    grad = 2.0 * diff / len(y_true)
    return float(loss), grad


def categorical_crossentropy(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, np.ndarray]:
    y_pred = np.clip(y_pred, EPS, 1 - EPS)
    loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    grad = (y_pred - y_true) / len(y_true)
    return float(loss), grad


LOSS_MAP = {
    "binary_crossentropy": binary_crossentropy,
    "mse": mse,
    "categorical_crossentropy": categorical_crossentropy,
}


def get_loss(name: str):
    if name not in LOSS_MAP:
        raise ValueError(f"Unknown loss: {name}")
    return LOSS_MAP[name]
