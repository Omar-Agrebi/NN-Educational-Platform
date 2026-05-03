import numpy as np
from typing import Tuple


def l1_penalty(weights: np.ndarray, lam: float) -> Tuple[float, np.ndarray]:
    penalty = lam * np.sum(np.abs(weights))
    grad = lam * np.sign(weights)
    return float(penalty), grad


def l2_penalty(weights: np.ndarray, lam: float) -> Tuple[float, np.ndarray]:
    penalty = 0.5 * lam * np.sum(weights ** 2)
    grad = lam * weights
    return float(penalty), grad


def no_penalty(weights: np.ndarray, lam: float) -> Tuple[float, np.ndarray]:
    return 0.0, np.zeros_like(weights)


REG_MAP = {
    "none": no_penalty,
    "l1": l1_penalty,
    "l2": l2_penalty,
}


def get_regularizer(name: str):
    name = name.lower().strip()
    if name not in REG_MAP:
        return no_penalty
    return REG_MAP[name]
