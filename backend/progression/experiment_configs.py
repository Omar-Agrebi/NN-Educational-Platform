from dataclasses import dataclass, field
from typing import Dict, Any, List


@dataclass
class ModelSpec:
    hidden_layers: int
    neurons_per_layer: int
    activation: str
    regularization: str
    reg_lambda: float
    learning_rate: float
    epochs: int
    batch_size: int
    label: str  # Human-readable label for this model
    description: str


@dataclass
class ExperimentConfig:
    experiment_id: str
    name: str
    description: str
    dataset: str
    model_a: ModelSpec
    model_b: ModelSpec
    locked_params: List[str]
    hypothesis: str  # What the experiment is designed to demonstrate


EXPERIMENTS: Dict[str, ExperimentConfig] = {
    "PERCEPTRON_VS_MLP_XOR": ExperimentConfig(
        experiment_id="PERCEPTRON_VS_MLP_XOR",
        name="Perceptron vs MLP on XOR",
        description=(
            "Compare a linear model (0 hidden layers) against a deep MLP (2 hidden layers) "
            "on the XOR dataset. The perceptron cannot learn XOR; the MLP can."
        ),
        dataset="xor",
        model_a=ModelSpec(
            hidden_layers=0,
            neurons_per_layer=8,
            activation="sigmoid",
            regularization="none",
            reg_lambda=0.0,
            learning_rate=0.01,
            epochs=200,
            batch_size=32,
            label="Perceptron (0 hidden layers)",
            description="Linear decision boundary — provably cannot solve XOR.",
        ),
        model_b=ModelSpec(
            hidden_layers=2,
            neurons_per_layer=8,
            activation="relu",
            regularization="none",
            reg_lambda=0.0,
            learning_rate=0.01,
            epochs=200,
            batch_size=32,
            label="MLP (2 hidden layers)",
            description="Non-linear decision boundary — solves XOR with ease.",
        ),
        locked_params=["hidden_layers", "activation", "learning_rate", "regularization",
                       "reg_lambda", "batch_size", "neurons_per_layer"],
        hypothesis=(
            "The perceptron will plateau around 50% accuracy on XOR. "
            "The MLP will exceed 90% accuracy, demonstrating the power of depth."
        ),
    ),
    "DEPTH_EXPERIMENT": ExperimentConfig(
        experiment_id="DEPTH_EXPERIMENT",
        name="Shallow vs Deep Network",
        description=(
            "Compare a 1-layer and 3-layer network on the same XOR problem. "
            "Observe how depth affects learning capacity and convergence speed."
        ),
        dataset="xor",
        model_a=ModelSpec(
            hidden_layers=1,
            neurons_per_layer=8,
            activation="relu",
            regularization="none",
            reg_lambda=0.0,
            learning_rate=0.01,
            epochs=200,
            batch_size=32,
            label="Shallow (1 hidden layer)",
            description="Minimal depth — may struggle with complex boundaries.",
        ),
        model_b=ModelSpec(
            hidden_layers=3,
            neurons_per_layer=16,
            activation="relu",
            regularization="none",
            reg_lambda=0.0,
            learning_rate=0.01,
            epochs=200,
            batch_size=32,
            label="Deep (3 hidden layers)",
            description="More depth — hierarchical feature learning.",
        ),
        locked_params=["activation", "learning_rate", "regularization", "reg_lambda", "batch_size"],
        hypothesis=(
            "The 3-layer network will converge faster and achieve higher accuracy. "
            "Depth provides representational power beyond what width alone can offer."
        ),
    ),
    "REGULARIZATION_EXPERIMENT": ExperimentConfig(
        experiment_id="REGULARIZATION_EXPERIMENT",
        name="No Regularization vs L2 on Noisy Data",
        description=(
            "Train two identical networks on the noisy dataset: one without regularization, "
            "one with L2 regularization. Compare train vs test accuracy gap."
        ),
        dataset="noisy",
        model_a=ModelSpec(
            hidden_layers=2,
            neurons_per_layer=32,
            activation="relu",
            regularization="none",
            reg_lambda=0.0,
            learning_rate=0.01,
            epochs=300,
            batch_size=32,
            label="No Regularization",
            description="Unconstrained model — will overfit to noise.",
        ),
        model_b=ModelSpec(
            hidden_layers=2,
            neurons_per_layer=32,
            activation="relu",
            regularization="l2",
            reg_lambda=0.01,
            learning_rate=0.01,
            epochs=300,
            batch_size=32,
            label="L2 Regularization (λ=0.01)",
            description="Weight penalty — discourages memorizing noise.",
        ),
        locked_params=["hidden_layers", "neurons_per_layer", "activation", "learning_rate",
                       "epochs", "batch_size"],
        hypothesis=(
            "The unregularized model will overfit (high train, low test accuracy). "
            "L2 regularization will close the generalization gap significantly."
        ),
    ),
}


def get_experiment(experiment_id: str) -> ExperimentConfig:
    if experiment_id not in EXPERIMENTS:
        raise ValueError(
            f"Unknown experiment '{experiment_id}'. Available: {list(EXPERIMENTS.keys())}"
        )
    return EXPERIMENTS[experiment_id]


def get_all_experiments() -> List[ExperimentConfig]:
    return list(EXPERIMENTS.values())
