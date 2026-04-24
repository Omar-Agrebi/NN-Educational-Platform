import numpy as np
from typing import Tuple


class StandardNormalizer:
    """
    Zero-mean, unit-variance normalization.
    Fit on train data only; apply to test (no data leakage).
    """

    def __init__(self):
        self.mean_: np.ndarray = None
        self.std_: np.ndarray = None
        self._fitted = False

    def fit(self, X_train: np.ndarray) -> "StandardNormalizer":
        """Compute mean and std from training data."""
        self.mean_ = np.mean(X_train, axis=0)
        self.std_ = np.std(X_train, axis=0)
        # Avoid division by zero
        self.std_ = np.where(self.std_ == 0, 1.0, self.std_)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply fitted normalization."""
        if not self._fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        """Fit on train data and transform it."""
        return self.fit(X_train).transform(X_train)

    def inverse_transform(self, X_norm: np.ndarray) -> np.ndarray:
        """Reverse normalization."""
        if not self._fitted:
            raise RuntimeError("Normalizer not fitted.")
        return X_norm * self.std_ + self.mean_


def normalize_split(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, StandardNormalizer]:
    """
    Convenience function: fit normalizer on train, apply to both.
    Returns: X_train_norm, X_test_norm, fitted_normalizer
    """
    norm = StandardNormalizer()
    X_train_norm = norm.fit_transform(X_train)
    X_test_norm = norm.transform(X_test)
    return X_train_norm, X_test_norm, norm
