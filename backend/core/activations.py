import numpy as np
from typing import Tuple


def sigmoid(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x_clipped = np.clip(x, -500, 500)
    out = 1.0 / (1.0 + np.exp(-x_clipped))
    deriv = out * (1.0 - out)
    return out, deriv


def relu(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    out = np.maximum(0.0, x)
    deriv = (x > 0).astype(float)
    return out, deriv


def tanh(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    out = np.tanh(x)
    deriv = 1.0 - out ** 2
    return out, deriv


def linear(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    out = x.copy()
    deriv = np.ones_like(x)
    return out, deriv


def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    out = np.where(x > 0, x, alpha * x)
    deriv = np.where(x > 0, 1.0, alpha)
    return out, deriv


def elu(x: np.ndarray, alpha: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    out = np.where(x > 0, x, alpha * (np.exp(np.clip(x, -500, 0)) - 1))
    deriv = np.where(x > 0, 1.0, out + alpha)
    return out, deriv


ACTIVATION_MAP = {
    "sigmoid": sigmoid,
    "relu": relu,
    "tanh": tanh,
    "linear": linear,
    "leaky_relu": leaky_relu,
    "elu": elu,
}


def get_activation(name: str):
    name = name.lower().strip()
    if name not in ACTIVATION_MAP:
        raise ValueError(f"Unknown activation: {name}. Choose from {list(ACTIVATION_MAP.keys())}")
    return ACTIVATION_MAP[name]
