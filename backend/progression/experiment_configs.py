EXPERIMENTS = {
    "perceptron_vs_mlp_xor": {
        "id": "perceptron_vs_mlp_xor",
        "title": "Perceptron vs MLP on XOR",
        "lesson": "A perceptron can only draw a straight line. XOR requires non-linear boundaries — impossible without hidden layers.",
        "dataset": "xor",
        "model_a": {"hidden_layers": 0, "neurons_per_layer": [], "activations_per_layer": [], "activation": "sigmoid", "regularization": "none", "reg_lambda": 0.0},
        "model_b": {"hidden_layers": 2, "neurons_per_layer": [8, 8], "activations_per_layer": ["relu", "relu"], "activation": "relu", "regularization": "none", "reg_lambda": 0.0},
        "train_config": {"learning_rate": 0.01, "epochs": 200, "batch_size": 32},
        "locked_except": ["epochs"],
        "model_a_label": "Perceptron (0 layers)",
        "model_b_label": "MLP (2 layers, ReLU)",
        "archetype_a": "The Overconfident Perceptron",
        "archetype_b": "The Enlightened MLP",
    },
    "depth_experiment": {
        "id": "depth_experiment",
        "title": "1 Layer vs 3 Layers",
        "lesson": "Depth allows the network to compose simple features into complex representations. More layers = more expressive power.",
        "dataset": "xor",
        "model_a": {"hidden_layers": 1, "neurons_per_layer": [8], "activations_per_layer": ["relu"], "activation": "relu", "regularization": "none", "reg_lambda": 0.0},
        "model_b": {"hidden_layers": 3, "neurons_per_layer": [8, 8, 8], "activations_per_layer": ["relu", "relu", "relu"], "activation": "relu", "regularization": "none", "reg_lambda": 0.0},
        "train_config": {"learning_rate": 0.01, "epochs": 200, "batch_size": 32},
        "locked_except": ["hidden_layers"],
        "model_a_label": "Shallow (1 layer)",
        "model_b_label": "Deep (3 layers)",
        "archetype_a": "The Shallow Thinker",
        "archetype_b": "The Deep Reasoner",
    },
    "regularization_experiment": {
        "id": "regularization_experiment",
        "title": "No Regularization vs L2 on Noisy Data",
        "lesson": "Without regularization, large weights memorize noise. L2 penalizes complexity — forcing the model to find the true signal.",
        "dataset": "noisy",
        "model_a": {"hidden_layers": 2, "neurons_per_layer": [16, 16], "activations_per_layer": ["relu", "relu"], "activation": "relu", "regularization": "none", "reg_lambda": 0.0},
        "model_b": {"hidden_layers": 2, "neurons_per_layer": [16, 16], "activations_per_layer": ["relu", "relu"], "activation": "relu", "regularization": "l2", "reg_lambda": 0.01},
        "train_config": {"learning_rate": 0.01, "epochs": 200, "batch_size": 32},
        "locked_except": ["regularization", "reg_lambda"],
        "model_a_label": "No Regularization",
        "model_b_label": "L2 Regularization (λ=0.01)",
        "archetype_a": "The Regularization Denier",
        "archetype_b": "The Disciplined Generalizer",
    },
}


def get_experiment(experiment_id: str) -> dict:
    if experiment_id not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment_id}")
    return EXPERIMENTS[experiment_id]


def list_experiments() -> list:
    return list(EXPERIMENTS.values())
