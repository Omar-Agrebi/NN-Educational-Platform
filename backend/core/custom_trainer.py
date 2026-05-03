"""
Unified trainer for custom datasets.
Handles: binary classification, multi-class classification, regression.
Pure NumPy only.
"""
import numpy as np
from typing import Optional, List, Dict, Any
from core.activations import get_activation
from core.regularization import get_regularizer
from core.optimizer import SGDMomentum, create_mini_batches
from core.multiclass import softmax, categorical_crossentropy, to_onehot, mse_regression, r2_score
from core.losses import binary_crossentropy


class CustomNeuralNet:
    """
    Flexible net supporting binary, multi-class, and regression outputs.
    """
    def __init__(self, layer_sizes: List[int], hidden_activations: List[str],
                 output_type: str = "sigmoid", seed: int = 42):
        np.random.seed(seed)
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1
        self.output_type = output_type  # "sigmoid", "softmax", "linear"
        self.hidden_activations = hidden_activations
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        self._z: List[np.ndarray] = []
        self._a: List[np.ndarray] = []
        self._init_weights()

    def _init_weights(self):
        for i in range(self.n_layers):
            fi, fo = self.layer_sizes[i], self.layer_sizes[i + 1]
            limit = np.sqrt(6.0 / (fi + fo))
            self.weights.append(np.random.uniform(-limit, limit, (fi, fo)))
            self.biases.append(np.zeros((1, fo)))

    def forward(self, X: np.ndarray) -> np.ndarray:
        self._z, self._a = [], [X]
        cur = X
        for i in range(self.n_layers):
            z = cur @ self.weights[i] + self.biases[i]
            self._z.append(z)
            if i < self.n_layers - 1:
                fn = get_activation(self.hidden_activations[i] if i < len(self.hidden_activations) else "relu")
                cur, _ = fn(z)
            else:
                if self.output_type == "sigmoid":
                    cur = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
                elif self.output_type == "softmax":
                    cur = softmax(z)
                else:  # linear
                    cur = z
            self._a.append(cur)
        return cur

    def backward(self, loss_grad: np.ndarray, reg_fn, reg_lambda: float,
                 clip: float = 5.0) -> tuple:
        n = self._a[0].shape[0]
        wg, bg = [None] * self.n_layers, [None] * self.n_layers
        delta = loss_grad

        for i in reversed(range(self.n_layers)):
            if i < self.n_layers - 1:
                fn = get_activation(self.hidden_activations[i] if i < len(self.hidden_activations) else "relu")
                _, deriv = fn(self._z[i])
                delta = delta * deriv

            dW = (self._a[i].T @ delta) / n
            db = delta.mean(axis=0, keepdims=True)
            _, rg = reg_fn(self.weights[i], reg_lambda)
            dW = np.clip(dW + rg, -clip, clip)
            db = np.clip(db, -clip, clip)
            wg[i], bg[i] = dW, db

            if i > 0:
                delta = delta @ self.weights[i].T
        return wg, bg


def train_custom(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    problem_type: str,
    n_classes: int = 2,
    hidden_layers: int = 2,
    neurons_per_layer: List[int] = None,
    activations_per_layer: List[str] = None,
    learning_rate: float = 0.01,
    epochs: int = 100,
    batch_size: int = 32,
    regularization: str = "none",
    reg_lambda: float = 0.0,
    momentum: float = 0.9,
    seed: int = 42,
) -> Dict[str, Any]:

    n_features = X_train.shape[1]
    if neurons_per_layer is None or len(neurons_per_layer) < hidden_layers:
        neurons_per_layer = [64] * hidden_layers
    if activations_per_layer is None or len(activations_per_layer) < hidden_layers:
        activations_per_layer = ["relu"] * hidden_layers

    # Build layer sizes
    if problem_type == "binary_classification":
        output_size = 1; output_type = "sigmoid"
    elif problem_type == "multiclass_classification":
        output_size = n_classes; output_type = "softmax"
    else:  # regression
        output_size = 1; output_type = "linear"

    layer_sizes = [n_features] + list(neurons_per_layer[:hidden_layers]) + [output_size]
    model = CustomNeuralNet(layer_sizes, activations_per_layer[:hidden_layers], output_type, seed)
    optimizer = SGDMomentum(lr=learning_rate, momentum=momentum)
    reg_fn = get_regularizer(regularization)

    # Prepare targets
    if problem_type == "multiclass_classification":
        y_train_enc = to_onehot(y_train.astype(int), n_classes)
        y_test_enc = to_onehot(y_test.astype(int), n_classes)
    else:
        y_train_enc = y_train.reshape(-1, 1)
        y_test_enc = y_test.reshape(-1, 1)

    history = {"epochs": [], "train_loss": [], "test_loss": [],
               "train_metric": [], "test_metric": []}

    best_test_loss = float("inf")
    best_epoch = 1

    for epoch in range(1, epochs + 1):
        batches = create_mini_batches(X_train, y_train_enc, batch_size, seed=epoch)
        for Xb, yb in batches:
            pred = model.forward(Xb)
            if problem_type == "multiclass_classification":
                _, grad = categorical_crossentropy(yb, pred)
            elif problem_type == "binary_classification":
                _, grad = binary_crossentropy(yb, pred)
            else:
                _, grad = mse_regression(yb, pred)
            wg, bg = model.backward(grad, reg_fn, reg_lambda)
            model.weights, model.biases = optimizer.step(model.weights, model.biases, wg, bg)

        # Eval
        train_pred = model.forward(X_train)
        test_pred = model.forward(X_test)

        if problem_type == "multiclass_classification":
            tl, _ = categorical_crossentropy(y_train_enc, train_pred)
            vl, _ = categorical_crossentropy(y_test_enc, test_pred)
            tm = float(np.mean(np.argmax(train_pred, axis=1) == y_train.astype(int)))
            vm = float(np.mean(np.argmax(test_pred, axis=1) == y_test.astype(int)))
        elif problem_type == "binary_classification":
            tl, _ = binary_crossentropy(y_train_enc, train_pred)
            vl, _ = binary_crossentropy(y_test_enc, test_pred)
            tm = float(np.mean((train_pred >= 0.5).flatten() == y_train.astype(int)))
            vm = float(np.mean((test_pred >= 0.5).flatten() == y_test.astype(int)))
        else:
            tl, _ = mse_regression(y_train_enc, train_pred)
            vl, _ = mse_regression(y_test_enc, test_pred)
            tm = r2_score(y_train, train_pred.flatten())
            vm = r2_score(y_test, test_pred.flatten())

        # Reg penalty
        reg_pen = sum(reg_fn(w, reg_lambda)[0] for w in model.weights)
        tl += reg_pen

        history["epochs"].append(epoch)
        history["train_loss"].append(round(float(tl), 6))
        history["test_loss"].append(round(float(vl), 6))
        history["train_metric"].append(round(tm, 4))
        history["test_metric"].append(round(vm, 4))

        if vl < best_test_loss:
            best_test_loss = vl
            best_epoch = epoch

    # Final predictions for download
    final_pred = model.forward(X_test)
    if problem_type == "multiclass_classification":
        predictions = np.argmax(final_pred, axis=1).tolist()
        probabilities = final_pred.tolist()
    elif problem_type == "binary_classification":
        predictions = (final_pred >= 0.5).astype(int).flatten().tolist()
        probabilities = final_pred.flatten().tolist()
    else:
        predictions = final_pred.flatten().tolist()
        probabilities = predictions

    # Feature importance via weight magnitude
    W0 = np.abs(model.weights[0])  # shape: (n_features, n_neurons)
    importance = W0.mean(axis=1)
    importance = (importance / (importance.max() + 1e-8)).tolist()

    return {
        "history": history,
        "final_metrics": {
            "train_metric": round(history["train_metric"][-1], 4),
            "test_metric": round(history["test_metric"][-1], 4),
            "train_loss": round(history["train_loss"][-1], 6),
            "test_loss": round(history["test_loss"][-1], 6),
            "best_epoch": best_epoch,
        },
        "predictions": predictions,
        "probabilities": probabilities,
        "feature_importance": importance,
        "metric_name": "accuracy" if problem_type != "regression" else "R²",
    }
