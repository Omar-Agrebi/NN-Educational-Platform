import numpy as np
from typing import Tuple


def generate_linear(n_samples: int = 200, noise: float = 0.1, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(seed)
    X0 = np.random.randn(n_samples // 2, 2) + np.array([1.5, 1.5])
    X1 = np.random.randn(n_samples // 2, 2) + np.array([-1.5, -1.5])
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n_samples // 2), np.ones(n_samples // 2)])
    if noise > 0:
        X += np.random.randn(*X.shape) * noise
    idx = np.random.permutation(len(y))
    return X[idx], y[idx]


def generate_xor(n_samples: int = 200, noise: float = 0.15, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(seed)
    n = n_samples // 4
    X_list, y_list = [], []
    centers = [(1, 1, 0), (-1, 1, 1), (1, -1, 1), (-1, -1, 0)]
    for cx, cy, label in centers:
        pts = np.random.randn(n, 2) * 0.4 + np.array([cx, cy])
        X_list.append(pts)
        y_list.extend([label] * n)
    X = np.vstack(X_list)
    y = np.array(y_list, dtype=float)
    if noise > 0:
        X += np.random.randn(*X.shape) * noise
    idx = np.random.permutation(len(y))
    return X[idx], y[idx]


def generate_noisy(n_samples: int = 300, noise: float = 0.3, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(seed)
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(float)
    # Partial label noise
    flip_mask = np.random.rand(n_samples) < noise
    y[flip_mask] = 1.0 - y[flip_mask]
    return X, y


def generate_imbalanced(n_samples: int = 300, noise: float = 0.1, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(seed)
    n_majority = int(n_samples * 0.85)
    n_minority = n_samples - n_majority
    X0 = np.random.randn(n_majority, 2) + np.array([0.5, 0.5])
    X1 = np.random.randn(n_minority, 2) + np.array([-1.5, -1.5])
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n_majority), np.ones(n_minority)])
    if noise > 0:
        X += np.random.randn(*X.shape) * noise
    idx = np.random.permutation(len(y))
    return X[idx], y[idx]


DATASET_REGISTRY = {
    "linear": {
        "fn": generate_linear,
        "description": "Two linearly separable Gaussian clusters in 2D.",
        "purpose": "Establishes baseline — perceptron should succeed here.",
        "expected_failure": "None — this is the easy case.",
        "n_classes": 2,
        "n_features": 2,
        "default_samples": 200,
    },
    "xor": {
        "fn": generate_xor,
        "description": "Classic XOR pattern: 4 quadrants, alternating labels.",
        "purpose": "Perceptron guaranteed to fail — motivates hidden layers.",
        "expected_failure": "Perceptron achieves ~50% (random chance).",
        "n_classes": 2,
        "n_features": 2,
        "default_samples": 200,
    },
    "noisy": {
        "fn": generate_noisy,
        "description": "Partially separable data with 30% label noise.",
        "purpose": "MLP overfits the noise — motivates regularization.",
        "expected_failure": "Large train/test gap without regularization.",
        "n_classes": 2,
        "n_features": 2,
        "default_samples": 300,
    },
    "imbalanced": {
        "fn": generate_imbalanced,
        "description": "85:15 class imbalance in 2D space.",
        "purpose": "Accuracy misleads — motivates F1 score.",
        "expected_failure": "Model predicts majority class only, 85% accuracy but 0% recall.",
        "n_classes": 2,
        "n_features": 2,
        "default_samples": 300,
    },
}


def get_dataset(name: str, n_samples: int = None, noise: float = None, seed: int = 42):
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(DATASET_REGISTRY.keys())}")
    info = DATASET_REGISTRY[name]
    fn = info["fn"]
    kwargs = {"seed": seed}
    if n_samples is not None:
        kwargs["n_samples"] = n_samples
    if noise is not None:
        kwargs["noise"] = noise
    return fn(**kwargs)
