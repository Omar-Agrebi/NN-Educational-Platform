import numpy as np
from typing import Tuple


def generate_linear(
    n_samples: int = 200,
    noise_level: float = 0.05,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    2D, 2-class linearly separable dataset.
    Perceptron succeeds here.
    """
    np.random.seed(random_seed)
    half = n_samples // 2
    # Class 0: centered at (-1, -1)
    X0 = np.random.randn(half, 2) * 0.6 + np.array([-1.5, -1.5])
    # Class 1: centered at (1, 1)
    X1 = np.random.randn(n_samples - half, 2) * 0.6 + np.array([1.5, 1.5])
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(half), np.ones(n_samples - half)])
    # Light noise: flip some labels
    n_flip = int(noise_level * n_samples)
    flip_idx = np.random.choice(n_samples, n_flip, replace=False)
    y[flip_idx] = 1 - y[flip_idx]
    # Shuffle
    idx = np.random.permutation(n_samples)
    return X[idx].astype(np.float32), y[idx].astype(int)


def generate_xor(
    n_samples: int = 200,
    noise_level: float = 0.1,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    2D classic XOR pattern with slight noise.
    Perceptron fails; motivates hidden layers.
    """
    np.random.seed(random_seed)
    quarter = n_samples // 4
    remainder = n_samples - 3 * quarter

    # XOR quadrants: (-, -) -> 0, (+, +) -> 0, (-, +) -> 1, (+, -) -> 1
    X00 = np.random.randn(quarter, 2) * 0.35 + np.array([-1.0, -1.0])  # class 0
    X11 = np.random.randn(quarter, 2) * 0.35 + np.array([1.0, 1.0])    # class 0
    X01 = np.random.randn(quarter, 2) * 0.35 + np.array([-1.0, 1.0])   # class 1
    X10 = np.random.randn(remainder, 2) * 0.35 + np.array([1.0, -1.0]) # class 1

    X = np.vstack([X00, X11, X01, X10])
    y = np.hstack([
        np.zeros(quarter),
        np.zeros(quarter),
        np.ones(quarter),
        np.ones(remainder),
    ])

    # Add noise: flip labels
    n_flip = int(noise_level * n_samples)
    flip_idx = np.random.choice(n_samples, n_flip, replace=False)
    y[flip_idx] = 1 - y[flip_idx]

    idx = np.random.permutation(n_samples)
    return X[idx].astype(np.float32), y[idx].astype(int)


def generate_noisy(
    n_samples: int = 300,
    noise_level: float = 0.30,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    2D partially separable with heavy label noise (30%).
    MLP overfits; motivates regularization.
    """
    np.random.seed(random_seed)
    half = n_samples // 2
    X0 = np.random.randn(half, 2) * 1.0 + np.array([-0.8, -0.8])
    X1 = np.random.randn(n_samples - half, 2) * 1.0 + np.array([0.8, 0.8])
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(half), np.ones(n_samples - half)])

    # Heavy noise flip
    n_flip = int(noise_level * n_samples)
    flip_idx = np.random.choice(n_samples, n_flip, replace=False)
    y[flip_idx] = 1 - y[flip_idx]

    idx = np.random.permutation(n_samples)
    return X[idx].astype(np.float32), y[idx].astype(int)


def generate_imbalanced(
    n_samples: int = 300,
    noise_level: float = 0.05,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    2D, class ratio 85:15.
    Accuracy misleads; motivates F1.
    """
    np.random.seed(random_seed)
    n_majority = int(0.85 * n_samples)
    n_minority = n_samples - n_majority

    X_maj = np.random.randn(n_majority, 2) * 0.8 + np.array([-0.5, -0.5])
    X_min = np.random.randn(n_minority, 2) * 0.5 + np.array([2.0, 2.0])
    X = np.vstack([X_maj, X_min])
    y = np.hstack([np.zeros(n_majority), np.ones(n_minority)])

    # Light noise
    n_flip = int(noise_level * n_samples)
    flip_idx = np.random.choice(n_samples, n_flip, replace=False)
    y[flip_idx] = 1 - y[flip_idx]

    idx = np.random.permutation(n_samples)
    return X[idx].astype(np.float32), y[idx].astype(int)


DATASET_INFO = {
    "linear": {
        "name": "linear",
        "description": "2D binary classification with a linear decision boundary.",
        "purpose": "Demonstrate that a simple perceptron (0 hidden layers) can solve linearly separable problems.",
        "expected_failure_mode": "None — a single neuron suffices. Fails only with extremely poor hyperparameters.",
        "default_samples": 200,
        "n_features": 2,
        "n_classes": 2,
    },
    "xor": {
        "name": "xor",
        "description": "Classic XOR pattern: 4 Gaussian clusters arranged in XOR configuration.",
        "purpose": "Show that a perceptron fails on non-linear problems; motivates hidden layers.",
        "expected_failure_mode": "Perceptron plateaus ~50% accuracy. A single hidden layer solves it.",
        "default_samples": 200,
        "n_features": 2,
        "n_classes": 2,
    },
    "noisy": {
        "name": "noisy",
        "description": "Overlapping Gaussian clusters with 30% label noise.",
        "purpose": "Show overfitting with large MLPs; motivates regularization.",
        "expected_failure_mode": "Deep MLP memorizes noise: high train accuracy, poor test accuracy.",
        "default_samples": 300,
        "n_features": 2,
        "n_classes": 2,
    },
    "imbalanced": {
        "name": "imbalanced",
        "description": "85/15 class ratio — majority class overwhelms minority.",
        "purpose": "Show that accuracy is misleading on imbalanced data; motivates F1 score.",
        "expected_failure_mode": "Model predicts only majority class: 85% accuracy, 0% F1 for minority.",
        "default_samples": 300,
        "n_features": 2,
        "n_classes": 2,
    },
}

GENERATORS = {
    "linear": generate_linear,
    "xor": generate_xor,
    "noisy": generate_noisy,
    "imbalanced": generate_imbalanced,
}


def generate_dataset(
    name: str,
    n_samples: int = None,
    noise_level: float = None,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a named dataset with optional overrides."""
    if name not in GENERATORS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {list(GENERATORS.keys())}")
    info = DATASET_INFO[name]
    if n_samples is None:
        n_samples = info["default_samples"]
    kwargs = {"n_samples": n_samples, "random_seed": random_seed}
    if noise_level is not None:
        kwargs["noise_level"] = noise_level
    return GENERATORS[name](**kwargs)
