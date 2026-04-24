import numpy as np
from typing import List, Tuple, Optional


class SGDMomentum:
    """
    Stochastic Gradient Descent with momentum.
    Supports mini-batch gradient descent.
    Includes gradient clipping to prevent explosion.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        clip_value: float = 5.0,
    ):
        self.lr = learning_rate
        self.momentum = momentum
        self.clip_value = clip_value
        self.velocity_w: List[np.ndarray] = []
        self.velocity_b: List[np.ndarray] = []
        self._initialized = False

    def _init_velocities(self, weights: List[np.ndarray], biases: List[np.ndarray]) -> None:
        self.velocity_w = [np.zeros_like(W) for W in weights]
        self.velocity_b = [np.zeros_like(b) for b in biases]
        self._initialized = True

    def clip_gradients(
        self,
        weight_grads: List[np.ndarray],
        bias_grads: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Clip gradients by value to prevent explosion."""
        clipped_w = [np.clip(dW, -self.clip_value, self.clip_value) for dW in weight_grads]
        clipped_b = [np.clip(db, -self.clip_value, self.clip_value) for db in bias_grads]
        return clipped_w, clipped_b

    def step(
        self,
        weights: List[np.ndarray],
        biases: List[np.ndarray],
        weight_grads: List[np.ndarray],
        bias_grads: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Apply one optimizer step.
        Returns updated weights and biases.
        """
        if not self._initialized:
            self._init_velocities(weights, biases)

        # Clip gradients
        weight_grads, bias_grads = self.clip_gradients(weight_grads, bias_grads)

        updated_weights = []
        updated_biases = []

        for i in range(len(weights)):
            # Update velocity
            self.velocity_w[i] = (
                self.momentum * self.velocity_w[i] - self.lr * weight_grads[i]
            )
            self.velocity_b[i] = (
                self.momentum * self.velocity_b[i] - self.lr * bias_grads[i]
            )
            # Update parameters
            updated_weights.append(weights[i] + self.velocity_w[i])
            updated_biases.append(biases[i] + self.velocity_b[i])

        return updated_weights, updated_biases

    def reset(self) -> None:
        """Reset optimizer state."""
        self.velocity_w = []
        self.velocity_b = []
        self._initialized = False


def make_mini_batches(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    random_seed: Optional[int] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Split data into mini-batches.
    Returns list of (X_batch, y_batch) tuples.
    """
    n = X.shape[0]
    if random_seed is not None:
        np.random.seed(random_seed)
    indices = np.random.permutation(n) if shuffle else np.arange(n)
    batches = []
    for start in range(0, n, batch_size):
        idx = indices[start : start + batch_size]
        batches.append((X[idx], y[idx].reshape(-1, 1)))
    return batches



