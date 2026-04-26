import numpy as np
import uuid
import time
from typing import Optional, Callable
from core.neural_net import NeuralNetwork
from core.losses import get_loss, binary_crossentropy
from core.regularization import get_regularizer
from core.optimizer import SGDMomentum, create_mini_batches
from models.responses import TrainingHistory, WeightSnapshot, InsightEvent


class Trainer:
    def __init__(self):
        self._stop_flag = False

    def stop(self):
        self._stop_flag = True

    def train(
        self,
        model: NeuralNetwork,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        learning_rate: float = 0.01,
        epochs: int = 100,
        batch_size: int = 32,
        regularization: str = "none",
        reg_lambda: float = 0.0,
        momentum: float = 0.9,
        gradient_clip: float = 5.0,
        early_stopping: bool = False,
        patience: int = 10,
        slow_mode: bool = False,
        on_epoch: Optional[Callable] = None,
    ) -> dict:
        self._stop_flag = False
        run_id = str(uuid.uuid4())[:8]

        optimizer = SGDMomentum(lr=learning_rate, momentum=momentum)
        reg_fn = get_regularizer(regularization)
        loss_fn = binary_crossentropy

        history = TrainingHistory()
        weight_snapshots = []
        insight_events = []

        best_test_loss = float("inf")
        best_test_loss_epoch = 0
        patience_counter = 0
        prev_train_loss = None
        overfit_detected = False
        overfit_epoch = None

        y_train_2d = y_train.reshape(-1, 1)
        y_test_2d = y_test.reshape(-1, 1)

        for epoch in range(1, epochs + 1):
            if self._stop_flag:
                break

            # Mini-batch training
            batches = create_mini_batches(X_train, y_train_2d, batch_size, seed=epoch)
            for X_batch, y_batch in batches:
                y_pred = model.forward(X_batch, training=True)
                _, loss_grad = loss_fn(y_batch, y_pred)

                # Regularization penalty
                reg_penalty = 0.0
                for w in model.weights:
                    p, _ = reg_fn(w, reg_lambda)
                    reg_penalty += p

                w_grads, b_grads = model.backward(y_batch, loss_grad, reg_fn, reg_lambda, gradient_clip)
                model.weights, model.biases = optimizer.step(model.weights, model.biases, w_grads, b_grads)

            # Eval
            train_pred = model.forward(X_train, training=False)
            test_pred = model.forward(X_test, training=False)

            train_loss, _ = loss_fn(y_train_2d, train_pred)
            test_loss, _ = loss_fn(y_test_2d, test_pred)

            train_acc = float(np.mean((train_pred >= 0.5).flatten() == y_train))
            test_acc = float(np.mean((test_pred >= 0.5).flatten() == y_test))

            # Reg penalty added to loss
            reg_total = sum(reg_fn(w, reg_lambda)[0] for w in model.weights)
            train_loss += reg_total

            history.epochs.append(epoch)
            history.train_loss.append(round(float(train_loss), 6))
            history.test_loss.append(round(float(test_loss), 6))
            history.train_acc.append(round(train_acc, 4))
            history.test_acc.append(round(test_acc, 4))

            # Insight events
            gap = test_loss - train_loss
            if not overfit_detected and gap > 0.15 and epoch > 10:
                overfit_detected = True
                overfit_epoch = epoch
                insight_events.append(InsightEvent(
                    epoch=epoch,
                    event_type="overfit_start",
                    message="Generalization gap widening — model starting to memorize training data",
                    severity=0.6,
                ))

            if train_acc < 0.6 and epoch == min(20, epochs):
                insight_events.append(InsightEvent(
                    epoch=epoch,
                    event_type="underfitting",
                    message="Model barely above random after 20 epochs — likely underfitting",
                    severity=0.7,
                ))

            if prev_train_loss is not None and train_loss > prev_train_loss * 1.5 and epoch > 5:
                insight_events.append(InsightEvent(
                    epoch=epoch,
                    event_type="divergence",
                    message="Loss spiked — learning rate may be too high",
                    severity=0.9,
                ))

            prev_train_loss = train_loss

            # Best test loss tracking
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_test_loss_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1

            # Snapshot every 10 epochs or last epoch
            if epoch % max(1, epochs // 20) == 0 or epoch == epochs:
                snap = WeightSnapshot(
                    epoch=epoch,
                    weights=[w.tolist() for w in model.weights],
                    biases=[b.flatten().tolist() for b in model.biases],
                    train_loss=round(float(train_loss), 6),
                    test_loss=round(float(test_loss), 6),
                    train_acc=round(train_acc, 4),
                    test_acc=round(test_acc, 4),
                )
                weight_snapshots.append(snap)

            # Epoch callback (slow mode)
            if on_epoch:
                on_epoch(epoch, float(train_loss), float(test_loss), train_acc, test_acc)

            if slow_mode:
                time.sleep(0.05)

            # Early stopping
            if early_stopping and patience_counter >= patience:
                insight_events.append(InsightEvent(
                    epoch=epoch,
                    event_type="early_stop",
                    message=f"Early stopping triggered at epoch {epoch} — best was epoch {best_test_loss_epoch}",
                    severity=0.2,
                ))
                break

        return {
            "run_id": run_id,
            "history": history,
            "weight_snapshots": weight_snapshots,
            "insight_events": insight_events,
            "best_epoch": best_test_loss_epoch,
            "overfit_epoch": overfit_epoch,
        }
