import numpy as np
from typing import Tuple


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.20,
    stratified: bool = False,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train/test sets.
    Supports stratified splitting for imbalanced data.
    Returns: X_train, X_test, y_train, y_test
    """
    np.random.seed(random_seed)
    n = X.shape[0]

    if stratified:
        classes = np.unique(y)
        train_idx = []
        test_idx = []
        for cls in classes:
            cls_idx = np.where(y == cls)[0]
            np.random.shuffle(cls_idx)
            n_test = max(1, int(len(cls_idx) * test_size))
            test_idx.extend(cls_idx[:n_test].tolist())
            train_idx.extend(cls_idx[n_test:].tolist())
        train_idx = np.array(train_idx)
        test_idx = np.array(test_idx)
        # Shuffle within splits
        np.random.shuffle(train_idx)
        np.random.shuffle(test_idx)
    else:
        idx = np.random.permutation(n)
        n_test = max(1, int(n * test_size))
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
