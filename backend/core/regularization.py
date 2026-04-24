import numpy as np
from typing import Tuple, List


def l1_penalty(weights: List[np.ndarray], lambda_: float) -> Tuple[float, List[np.ndarray]]:
    """
    L1 regularization penalty and gradient.
    Returns (penalty_value, list of weight gradients).
    """
    penalty = 0.0
    grads = []
    for W in weights:
        penalty += lambda_ * np.sum(np.abs(W))
        grads.append(lambda_ * np.sign(W))
    return float(penalty), grads


def l2_penalty(weights: List[np.ndarray], lambda_: float) -> Tuple[float, List[np.ndarray]]:
    """
    L2 regularization penalty and gradient.
    Returns (penalty_value, list of weight gradients).
    """
    penalty = 0.0
    grads = []
    for W in weights:
        penalty += 0.5 * lambda_ * np.sum(W ** 2)
        grads.append(lambda_ * W)
    return float(penalty), grads


def no_penalty(weights: List[np.ndarray], lambda_: float = 0.0) -> Tuple[float, List[np.ndarray]]:
    """
    No regularization. Returns zero penalty and zero gradients.
    """
    grads = [np.zeros_like(W) for W in weights]
    return 0.0, grads


REGULARIZATIONS = {
    "none": no_penalty,
    "l1": l1_penalty,
    "l2": l2_penalty,
}


def get_regularization(name: str):
    """Return regularization function by name."""
    name = name.lower()
    if name not in REGULARIZATIONS:
        raise ValueError(f"Unknown regularization '{name}'. Choose from: {list(REGULARIZATIONS.keys())}")
    return REGULARIZATIONS[name]
