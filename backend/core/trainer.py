import numpy as np
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

from core.neural_net import NeuralNetwork
from core.losses import binary_crossentropy
from core.regularization import get_regularization
from core.optimizer import SGDMomentum, make_mini_batches


@dataclass
class TrainingHistory:
    epochs: List[int] = field(default_factory=list)
    train_loss: List[float] = field(default_factory=list)
    test_loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    test_acc: List[float] = field(default_factory=list)
    snapshots: List[Dict[str, Any]] = field(default_factory=list)


class Trainer:
    """
    Full epoch training loop for NeuralNetwork.
    Captures per-epoch metrics and emits snapshots.
    Supports stop signal, step-by-step mode, slow mode delay.
    """

    def __init__(
        self,
        model: NeuralNetwork,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        regularization: str = "none",
        reg_lambda: float = 0.001,
        momentum: float = 0.9,
        clip_value: float = 5.0,
    ):
        self.model = model
        self.batch_size = batch_size
        self.reg_fn = get_regularization(regularization)
        self.reg_lambda = reg_lambda
        self.optimizer = SGDMomentum(
            learning_rate=learning_rate,
            momentum=momentum,
            clip_value=clip_value,
        )
        self._stop_signal = False

    def stop(self) -> None:
        """Signal training to stop after current epoch."""
        self._stop_signal = True

    def _compute_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.model.predict(X)
        return float(np.mean(preds == y.flatten()))

    def _compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        probs = self.model.forward(X)
        loss, _ = binary_crossentropy(y.reshape(-1, 1), probs)
        # Add regularization penalty
        pen, _ = self.reg_fn(self.model.weights, self.reg_lambda)
        return loss + pen

    def _train_epoch(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epoch_seed: int,
    ) -> float:
        """Run one epoch of mini-batch gradient descent. Returns mean batch loss."""
        batches = make_mini_batches(X_train, y_train, self.batch_size, shuffle=True, random_seed=epoch_seed)
        epoch_loss = 0.0

        for X_batch, y_batch in batches:
            # Forward
            probs = self.model.forward(X_batch)
            # Loss + gradient
            loss_val, loss_grad = binary_crossentropy(y_batch, probs)
            # Regularization
            pen, reg_grads = self.reg_fn(self.model.weights, self.reg_lambda)
            # Backward
            weight_grads, bias_grads = self.model.backward(loss_grad, reg_grads)
            # Optimizer step
            self.model.weights, self.model.biases = self.optimizer.step(
                self.model.weights, self.model.biases, weight_grads, bias_grads
            )
            epoch_loss += loss_val + pen

        return epoch_loss / max(len(batches), 1)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        epochs: int = 100,
        slow_mode: bool = False,
        slow_delay: float = 0.05,
        capture_boundary_every: int = 10,
    ) -> TrainingHistory:
        """
        Full training loop.
        Returns TrainingHistory with per-epoch metrics and snapshots.
        """
        self._stop_signal = False
        history = TrainingHistory()

        for epoch in range(1, epochs + 1):
            if self._stop_signal:
                break

            # Train one epoch
            train_loss = self._train_epoch(X_train, y_train, epoch_seed=epoch)

            # Evaluate
            test_loss = self._compute_loss(X_test, y_test)
            train_acc = self._compute_accuracy(X_train, y_train)
            test_acc = self._compute_accuracy(X_test, y_test)

            history.epochs.append(epoch)
            history.train_loss.append(round(train_loss, 6))
            history.test_loss.append(round(test_loss, 6))
            history.train_acc.append(round(train_acc, 4))
            history.test_acc.append(round(test_acc, 4))

            # Capture snapshot
            snapshot = {
                "epoch": epoch,
                "weights": self.model.get_weights(),
                "train_loss": round(train_loss, 6),
                "test_loss": round(test_loss, 6),
                "train_acc": round(train_acc, 4),
                "test_acc": round(test_acc, 4),
                "boundary_data": None,  # filled by boundary_computer if needed
            }
            history.snapshots.append(snapshot)

            if slow_mode:
                time.sleep(slow_delay)

        return history
