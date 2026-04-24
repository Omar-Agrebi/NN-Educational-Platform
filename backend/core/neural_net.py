import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from core.activations import get_activation, sigmoid


class NeuralNetwork:
    """
    Pure NumPy neural network.
    Supports variable depth (0 to N hidden layers).
    Weight initialization: Xavier/Glorot.
    """

    def __init__(
        self,
        input_size: int,
        hidden_layers: int,
        neurons_per_layer: int,
        output_size: int,
        activation: str = "relu",
        random_seed: int = 42,
    ):
        np.random.seed(random_seed)
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.neurons_per_layer = neurons_per_layer
        self.output_size = output_size
        self.activation_name = activation
        self.activation_fn = get_activation(activation)

        # Build layer sizes
        self.layer_sizes = self._build_layer_sizes()

        # Initialize weights and biases (Xavier/Glorot)
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        for i in range(len(self.layer_sizes) - 1):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            W = np.random.uniform(-limit, limit, (fan_in, fan_out))
            b = np.zeros((1, fan_out))
            self.weights.append(W)
            self.biases.append(b)

        # Cache for backprop
        self._cache: Dict[str, Any] = {}

    def _build_layer_sizes(self) -> List[int]:
        sizes = [self.input_size]
        for _ in range(self.hidden_layers):
            sizes.append(self.neurons_per_layer)
        sizes.append(self.output_size)
        return sizes

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass through the network.
        Returns output probabilities.
        Caches intermediate values for backprop.
        """
        self._cache = {"activations": [X], "pre_activations": [], "derivatives": []}
        current = X

        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = current @ W + b
            self._cache["pre_activations"].append(z)

            is_last = (i == len(self.weights) - 1)
            if is_last:
                # Output layer: always sigmoid for binary classification
                out, deriv = sigmoid(z)
            else:
                out, deriv = self.activation_fn(z)

            self._cache["activations"].append(out)
            self._cache["derivatives"].append(deriv)
            current = out

        return current

    def backward(
        self,
        loss_grad: np.ndarray,
        reg_weight_grads: Optional[List[np.ndarray]] = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Backpropagation.
        loss_grad: gradient of loss w.r.t. output (from loss function)
        reg_weight_grads: per-layer regularization gradients (optional)
        Returns: (weight_gradients, bias_gradients)
        """
        n = self._cache["activations"][0].shape[0]
        weight_grads = []
        bias_grads = []

        delta = loss_grad  # Start with output gradient

        for i in reversed(range(len(self.weights))):
            activation_in = self._cache["activations"][i]
            # Gradient w.r.t. weights
            dW = activation_in.T @ delta / n
            db = np.mean(delta, axis=0, keepdims=True)

            # Add regularization gradient if provided
            if reg_weight_grads is not None:
                dW = dW + reg_weight_grads[i]

            weight_grads.insert(0, dW)
            bias_grads.insert(0, db)

            if i > 0:
                # Propagate gradient back through previous layer activation
                deriv = self._cache["derivatives"][i - 1]
                delta = (delta @ self.weights[i].T) * deriv

        return weight_grads, bias_grads

    def get_weights(self) -> Dict[str, Any]:
        """Return serializable copy of all weights and biases."""
        return {
            "weights": [W.tolist() for W in self.weights],
            "biases": [b.tolist() for b in self.biases],
            "config": {
                "input_size": self.input_size,
                "hidden_layers": self.hidden_layers,
                "neurons_per_layer": self.neurons_per_layer,
                "output_size": self.output_size,
                "activation": self.activation_name,
            },
        }

    def set_weights(self, weights_dict: Dict[str, Any]) -> None:
        """Restore weights from serialized dict."""
        self.weights = [np.array(W) for W in weights_dict["weights"]]
        self.biases = [np.array(b) for b in weights_dict["biases"]]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Run forward pass and threshold to binary predictions."""
        probs = self.forward(X)
        return (probs >= threshold).astype(int).flatten()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return raw probabilities."""
        return self.forward(X).flatten()
