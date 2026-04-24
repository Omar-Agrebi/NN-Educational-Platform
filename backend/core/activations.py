import numpy as np
from typing import Tuple


def sigmoid(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Sigmoid activation. Returns (output, derivative)."""
    # Clip for numerical stability
    z_clipped = np.clip(z, -500, 500)
    out = 1.0 / (1.0 + np.exp(-z_clipped))
    deriv = out * (1.0 - out)
    return out, deriv


def relu(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """ReLU activation. Returns (output, derivative)."""
    out = np.maximum(0.0, z)
    deriv = (z > 0).astype(float)
    return out, deriv


def tanh(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Tanh activation. Returns (output, derivative)."""
    out = np.tanh(z)
    deriv = 1.0 - out ** 2
    return out, deriv


def linear(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Linear (identity) activation. Returns (output, derivative)."""
    out = z.copy()
    deriv = np.ones_like(z)
    return out, deriv


ACTIVATIONS = {
    "sigmoid": sigmoid,
    "relu": relu,
    "tanh": tanh,
    "linear": linear,
}


def get_activation(name: str):
    """Return activation function by name."""
    if name not in ACTIVATIONS:
        raise ValueError(f"Unknown activation '{name}'. Choose from: {list(ACTIVATIONS.keys())}")
    return ACTIVATIONS[name]
