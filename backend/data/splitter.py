import numpy as np
from typing import Tuple


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    stratified: bool = True,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(seed)
    n = len(y)

    if stratified:
        classes = np.unique(y)
        train_idx, test_idx = [], []
        for c in classes:
            idx = np.where(y == c)[0]
            np.random.shuffle(idx)
            n_test = max(1, int(len(idx) * test_size))
            test_idx.extend(idx[:n_test].tolist())
            train_idx.extend(idx[n_test:].tolist())
        train_idx = np.array(train_idx)
        test_idx = np.array(test_idx)
    else:
        idx = np.random.permutation(n)
        n_test = int(n * test_size)
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
