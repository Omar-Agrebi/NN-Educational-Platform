import numpy as np
from typing import Tuple


def binary_crossentropy(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Binary cross-entropy loss.
    Returns (loss_value, gradient w.r.t. y_pred).
    """
    eps = 1e-15
    y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
    n = y_true.shape[0]
    loss = -np.mean(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
    # Gradient of BCE w.r.t. output (before sigmoid if needed, here raw output gradient)
    grad = (y_pred_clipped - y_true) / (y_pred_clipped * (1 - y_pred_clipped) * n + eps)
    return float(loss), grad


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Mean squared error loss.
    Returns (loss_value, gradient w.r.t. y_pred).
    """
    n = y_true.shape[0]
    diff = y_pred - y_true
    loss = float(np.mean(diff ** 2))
    grad = 2.0 * diff / n
    return loss, grad


def categorical_crossentropy(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Categorical cross-entropy loss.
    y_true: one-hot encoded (n, C) or integer labels (n,)
    y_pred: softmax probabilities (n, C)
    Returns (loss_value, gradient w.r.t. y_pred).
    """
    eps = 1e-15
    n = y_true.shape[0]
    y_pred_clipped = np.clip(y_pred, eps, 1.0)

    # Convert integer labels to one-hot if needed
    if y_true.ndim == 1:
        n_classes = y_pred.shape[1]
        y_onehot = np.zeros((n, n_classes))
        y_onehot[np.arange(n), y_true.astype(int)] = 1.0
    else:
        y_onehot = y_true

    loss = -np.sum(y_onehot * np.log(y_pred_clipped)) / n
    grad = (y_pred_clipped - y_onehot) / n
    return float(loss), grad


LOSSES = {
    "binary_crossentropy": binary_crossentropy,
    "mse": mse,
    "categorical_crossentropy": categorical_crossentropy,
}


def get_loss(name: str):
    """Return loss function by name."""
    if name not in LOSSES:
        raise ValueError(f"Unknown loss '{name}'. Choose from: {list(LOSSES.keys())}")
    return LOSSES[name]
