import numpy as np
from typing import List


class SGDMomentum:
    def __init__(self, lr: float = 0.01, momentum: float = 0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocity_w: List[np.ndarray] = []
        self.velocity_b: List[np.ndarray] = []
        self._initialized = False

    def _init_velocity(self, weights, biases):
        self.velocity_w = [np.zeros_like(w) for w in weights]
        self.velocity_b = [np.zeros_like(b) for b in biases]
        self._initialized = True

    def step(self, weights, biases, weight_grads, bias_grads):
        if not self._initialized:
            self._init_velocity(weights, biases)
        for i in range(len(weights)):
            self.velocity_w[i] = self.momentum * self.velocity_w[i] - self.lr * weight_grads[i]
            self.velocity_b[i] = self.momentum * self.velocity_b[i] - self.lr * bias_grads[i]
            weights[i] += self.velocity_w[i]
            biases[i] += self.velocity_b[i]
        return weights, biases


def create_mini_batches(X, y, batch_size, seed=None):
    if seed is not None:
        np.random.seed(seed)
    n = X.shape[0]
    indices = np.random.permutation(n)
    X_shuf = X[indices]
    y_shuf = y[indices]
    batches = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batches.append((X_shuf[start:end], y_shuf[start:end]))
    return batches
