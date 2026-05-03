import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from core.activations import get_activation, sigmoid


class NeuralNetwork:
    """
    Pure NumPy neural network.
    Supports variable depth, per-layer activation functions, Xavier init.
    """

    def __init__(
        self,
        layer_sizes: List[int],
        activations: Optional[List[str]] = None,
        output_activation: str = "sigmoid",
        seed: int = 42,
    ):
        np.random.seed(seed)
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1

        # Per-layer activations (hidden layers only; output always sigmoid for binary)
        if activations is None:
            self.activations = ["relu"] * (self.n_layers - 1) + [output_activation]
        else:
            if len(activations) == self.n_layers - 1:
                self.activations = list(activations) + [output_activation]
            elif len(activations) == self.n_layers:
                self.activations = list(activations)
            else:
                self.activations = (list(activations) + [output_activation] * self.n_layers)[:self.n_layers]

        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        self._init_weights()

        # Cache for backprop
        self._z: List[np.ndarray] = []
        self._a: List[np.ndarray] = []

    def _init_weights(self):
        self.weights = []
        self.biases = []
        for i in range(self.n_layers):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            # Xavier/Glorot uniform initialization
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            W = np.random.uniform(-limit, limit, (fan_in, fan_out))
            b = np.zeros((1, fan_out))
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        self._z = []
        self._a = [X]
        current = X
        for i in range(self.n_layers):
            z = current @ self.weights[i] + self.biases[i]
            self._z.append(z)
            act_fn = get_activation(self.activations[i])
            a, _ = act_fn(z)
            self._a.append(a)
            current = a
        return current

    def backward(
        self,
        y_true: np.ndarray,
        loss_grad: np.ndarray,
        reg_fn,
        reg_lambda: float,
        clip_value: float = 5.0,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        n = y_true.shape[0]
        weight_grads = [None] * self.n_layers
        bias_grads = [None] * self.n_layers

        delta = loss_grad
        for i in reversed(range(self.n_layers)):
            act_fn = get_activation(self.activations[i])
            _, deriv = act_fn(self._z[i])
            delta = delta * deriv

            dW = (self._a[i].T @ delta) / n
            db = np.mean(delta, axis=0, keepdims=True)

            # Regularization gradient
            _, reg_grad = reg_fn(self.weights[i], reg_lambda)
            dW += reg_grad

            # Gradient clipping
            dW = np.clip(dW, -clip_value, clip_value)
            db = np.clip(db, -clip_value, clip_value)

            weight_grads[i] = dW
            bias_grads[i] = db

            if i > 0:
                delta = delta @ self.weights[i].T

        return weight_grads, bias_grads

    def get_weights(self) -> Dict[str, Any]:
        return {
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
            "layer_sizes": self.layer_sizes,
            "activations": self.activations,
        }

    def set_weights(self, state: Dict[str, Any]):
        self.weights = [np.array(w) for w in state["weights"]]
        self.biases = [np.array(b) for b in state["biases"]]
        self.layer_sizes = state["layer_sizes"]
        self.activations = state.get("activations", self.activations)
        self.n_layers = len(self.weights)

    def get_layer_activations(self, X: np.ndarray) -> List[np.ndarray]:
        """Returns activations for each layer (for weight inspector)."""
        self.forward(X)
        return [a.copy() for a in self._a]

    def get_weight_importance(self, layer_idx: int, neuron_idx: int) -> List[float]:
        """L2 norm of incoming weights for a neuron."""
        if layer_idx >= len(self.weights):
            return []
        w = self.weights[layer_idx]
        if neuron_idx >= w.shape[1]:
            return []
        importance = np.abs(w[:, neuron_idx])
        max_val = importance.max() + 1e-8
        return (importance / max_val).tolist()

    @classmethod
    def from_config(cls, model_config) -> "NeuralNetwork":
        """Build network from ModelConfig schema."""
        n_hidden = model_config.hidden_layers
        neurons = model_config.neurons_per_layer
        activations_per = model_config.activations_per_layer
        fallback = model_config.activation

        if len(neurons) < n_hidden:
            neurons = neurons + [8] * (n_hidden - len(neurons))
        neurons = neurons[:n_hidden]

        if len(activations_per) < n_hidden:
            activations_per = activations_per + [fallback] * (n_hidden - len(activations_per))
        activations_per = activations_per[:n_hidden]

        layer_sizes = [2] + list(neurons) + [1]
        return cls(
            layer_sizes=layer_sizes,
            activations=activations_per,
            output_activation="sigmoid",
        )
