from typing import List, Dict, Any


LEVELS = [
    {
        "level_id": 1,
        "name": "THE PERCEPTRON FORGE",
        "description": "Train your first neuron on linearly separable data. Master the basics before the real forge begins.",
        "dataset": "linear",
        "locked_params": ["hidden_layers", "neurons_per_layer", "activation", "regularization", "reg_lambda", "batch_size"],
        "unlocked_params": ["learning_rate", "epochs", "slow_mode"],
        "default_config": {"learning_rate": 0.001, "epochs": 50, "hidden_layers": 0},
        "challenge": {
            "type": "train_accuracy",
            "threshold": 0.85,
            "description": "Achieve >85% test accuracy on the linear dataset",
            "target_metric": "test_acc",
            "target_value": 0.85,
        },
        "unlock_reward": ["hidden_layers", "activation", "neurons_per_layer", "xor_dataset"],
        "xp_reward": 100,
        "tips": [
            "Start with the default learning rate and observe the loss curve.",
            "If training is slow, try increasing the learning rate from 0.001 to 0.01.",
            "Watch the decision boundary — it should eventually split the two clusters.",
            "Linear data is easy — a perceptron (0 hidden layers) is enough here.",
        ],
        "color": "#00f5ff",
    },
    {
        "level_id": 2,
        "name": "THE XOR CRUCIBLE",
        "description": "A perceptron draws one straight line. XOR needs a curve. Watch it fail, then fix it.",
        "dataset": "xor",
        "locked_params": ["regularization", "reg_lambda", "batch_size"],
        "unlocked_params": ["learning_rate", "epochs", "hidden_layers", "neurons_per_layer", "activation", "slow_mode"],
        "default_config": {"learning_rate": 0.01, "epochs": 100, "hidden_layers": 0},
        "challenge": {
            "type": "test_accuracy",
            "threshold": 0.90,
            "description": "Solve XOR with >90% test accuracy",
            "target_metric": "test_acc",
            "target_value": 0.90,
        },
        "unlock_reward": ["depth_experiments", "noisy_dataset"],
        "xp_reward": 200,
        "tips": [
            "First, train with 0 hidden layers and observe the straight-line boundary failing on XOR.",
            "Add 1 hidden layer with at least 4 neurons — this introduces the non-linearity needed.",
            "ReLU activation often converges faster than sigmoid for hidden layers.",
            "XOR is symmetric — try different random seeds if you get stuck.",
            "With 2 hidden layers and 8 neurons each, 90%+ accuracy is very achievable.",
        ],
        "color": "#8b5cf6",
    },
    {
        "level_id": 3,
        "name": "DEPTH TRIALS",
        "description": "Does adding more layers always help? Prove it with numbers.",
        "dataset": "xor",
        "locked_params": ["regularization", "reg_lambda"],
        "unlocked_params": ["learning_rate", "epochs", "hidden_layers", "neurons_per_layer", "activation", "batch_size", "slow_mode"],
        "default_config": {"learning_rate": 0.01, "epochs": 100, "hidden_layers": 1},
        "challenge": {
            "type": "accuracy_gain",
            "threshold": 0.10,
            "description": "Show >10% accuracy gain by adding depth vs baseline",
            "target_metric": "accuracy_gain",
            "target_value": 0.10,
        },
        "unlock_reward": ["regularization", "reg_lambda", "noisy_dataset"],
        "xp_reward": 200,
        "tips": [
            "First run with 1 hidden layer to establish your baseline.",
            "Then increase to 3-4 layers and compare the test accuracy.",
            "More depth can sometimes hurt (vanishing gradients) — try different activations.",
            "Tanh and ReLU handle depth better than Sigmoid for hidden layers.",
        ],
        "color": "#00ff88",
    },
    {
        "level_id": 4,
        "name": "THE OVERFIT ARENA",
        "description": "Watch your model memorize noise. Then regularize it into submission.",
        "dataset": "noisy",
        "locked_params": ["batch_size"],
        "unlocked_params": ["learning_rate", "epochs", "hidden_layers", "neurons_per_layer", "activation", "regularization", "reg_lambda", "slow_mode"],
        "default_config": {"learning_rate": 0.01, "epochs": 200, "hidden_layers": 2, "regularization": "none"},
        "challenge": {
            "type": "generalization_gap",
            "threshold": 0.05,
            "description": "Reduce train/test accuracy gap to <5%",
            "target_metric": "gen_gap",
            "target_value": 0.05,
        },
        "unlock_reward": ["imbalanced_dataset", "full_architecture", "batch_size", "early_stopping"],
        "xp_reward": 300,
        "tips": [
            "First train without regularization and watch the train/test gap grow — this is overfitting.",
            "Try L2 regularization with λ=0.01 as a starting point.",
            "If gap is still large, increase λ. If accuracy drops too much, decrease λ.",
            "L1 regularization creates sparse weights — good for feature selection.",
            "Early stopping is another powerful anti-overfitting tool.",
        ],
        "color": "#ff6b2b",
    },
    {
        "level_id": 5,
        "name": "THE ARCHITECT CHAMBER",
        "description": "85% accuracy. 0% recall. Accuracy is lying to you — use F1.",
        "dataset": "imbalanced",
        "locked_params": [],
        "unlocked_params": ["learning_rate", "epochs", "hidden_layers", "neurons_per_layer", "activation", "regularization", "reg_lambda", "batch_size", "slow_mode", "early_stopping"],
        "default_config": {"learning_rate": 0.01, "epochs": 100, "hidden_layers": 1},
        "challenge": {
            "type": "f1_score",
            "threshold": 0.88,
            "description": "Achieve >88% F1 score on imbalanced dataset",
            "target_metric": "f1_score",
            "target_value": 0.88,
        },
        "unlock_reward": ["duel_mode"],
        "xp_reward": 400,
        "tips": [
            "The dataset is 85% class 0 — a model predicting only class 0 gets 85% accuracy but 0% F1.",
            "Watch the confusion matrix: you need both precision AND recall to be high.",
            "Try different network depths — this dataset benefits from moderate complexity.",
            "L2 regularization can help prevent overfitting on the majority class.",
            "Lower learning rate (0.001) sometimes helps with imbalanced convergence.",
        ],
        "color": "#ff3b5c",
    },
    {
        "level_id": 6,
        "name": "THE DUEL",
        "description": "No hints. Two models. One dataset. Prove your model is superior on two metrics.",
        "dataset": "player_choice",
        "locked_params": [],
        "unlocked_params": ["all"],
        "default_config": {"learning_rate": 0.1, "epochs": 50, "hidden_layers": 0},
        "challenge": {
            "type": "duel",
            "threshold": 2,
            "description": "Beat the opponent model on accuracy AND F1",
            "target_metric": "wins",
            "target_value": 2,
        },
        "unlock_reward": ["master_badge", "full_forge", "replay_mode", "sandbox"],
        "xp_reward": 500,
        "tips": [
            "No tips here — this is the final boss.",
            "The opponent uses LR=0.1 with no hidden layers on a hard dataset.",
            "You have full control. Choose your dataset wisely.",
        ],
        "color": "#ff6b2b",
    },
]


def get_level(level_id: int) -> Dict[str, Any]:
    for lvl in LEVELS:
        if lvl["level_id"] == level_id:
            return lvl
    raise ValueError(f"Level {level_id} not found")


def get_all_levels() -> List[Dict[str, Any]]:
    return LEVELS


def is_param_unlocked(level_id: int, param: str, unlocked_features: List[str]) -> bool:
    level = get_level(level_id)
    return param in level["unlocked_params"] or param in unlocked_features or "all" in level["unlocked_params"]
