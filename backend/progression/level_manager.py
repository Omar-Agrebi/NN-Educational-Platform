from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class ChallengeDefinition:
    metric: str          # "train_accuracy", "test_accuracy", "f1_score", "accuracy_gain", "train_test_gap"
    operator: str        # "gt", "lt", "gte", "lte"
    threshold: float
    description: str
    baseline_run_id: Optional[str] = None  # for accuracy_gain metric


@dataclass
class LevelConfig:
    level_id: int
    name: str
    description: str
    dataset: str
    locked_params: List[str]
    unlocked_params: List[str]
    default_config: Dict[str, Any]
    challenge: ChallengeDefinition
    unlock_reward: List[str]
    xp_reward: int


LEVELS: Dict[int, LevelConfig] = {
    1: LevelConfig(
        level_id=1,
        name="THE PERCEPTRON FORGE",
        description=(
            "Master the simplest neural network: a single neuron with no hidden layers. "
            "Your mission is to separate two linearly-arranged classes. "
            "But watch out — with bad learning rate or too few epochs, even this can fail."
        ),
        dataset="linear",
        locked_params=["hidden_layers", "neurons_per_layer", "activation", "regularization", "reg_lambda"],
        unlocked_params=["learning_rate", "epochs", "batch_size"],
        default_config={
            "learning_rate": 0.001,
            "epochs": 50,
            "hidden_layers": 0,
            "neurons_per_layer": 8,
            "activation": "sigmoid",
            "regularization": "none",
            "reg_lambda": 0.0,
            "batch_size": 32,
        },
        challenge=ChallengeDefinition(
            metric="train_accuracy",
            operator="gt",
            threshold=0.85,
            description="Achieve train accuracy > 85% on the linear dataset.",
        ),
        unlock_reward=["hidden_layers", "activation", "xor_dataset"],
        xp_reward=100,
    ),
    2: LevelConfig(
        level_id=2,
        name="THE XOR CRUCIBLE",
        description=(
            "XOR cannot be solved by a perceptron. Watch it fail — then add hidden layers "
            "to build the first true multi-layer network and break the XOR barrier."
        ),
        dataset="xor",
        locked_params=["neurons_per_layer", "regularization", "reg_lambda"],
        unlocked_params=["learning_rate", "epochs", "hidden_layers", "activation", "batch_size"],
        default_config={
            "learning_rate": 0.01,
            "epochs": 100,
            "hidden_layers": 0,
            "neurons_per_layer": 8,
            "activation": "relu",
            "regularization": "none",
            "reg_lambda": 0.0,
            "batch_size": 32,
        },
        challenge=ChallengeDefinition(
            metric="test_accuracy",
            operator="gt",
            threshold=0.90,
            description="Achieve test accuracy > 90% on the XOR dataset.",
        ),
        unlock_reward=["layer_depth", "noisy_dataset"],
        xp_reward=200,
    ),
    3: LevelConfig(
        level_id=3,
        name="DEPTH TRIALS",
        description=(
            "Experiment with network depth. Can adding more layers always help? "
            "Compare a 1-layer network vs deeper architectures and quantify the accuracy gain."
        ),
        dataset="xor",
        locked_params=["regularization", "reg_lambda"],
        unlocked_params=["learning_rate", "epochs", "hidden_layers", "neurons_per_layer", "activation", "batch_size"],
        default_config={
            "learning_rate": 0.01,
            "epochs": 100,
            "hidden_layers": 1,
            "neurons_per_layer": 8,
            "activation": "relu",
            "regularization": "none",
            "reg_lambda": 0.0,
            "batch_size": 32,
        },
        challenge=ChallengeDefinition(
            metric="accuracy_gain",
            operator="gt",
            threshold=0.10,
            description="Improve test accuracy by more than 10% vs the 1-layer baseline.",
        ),
        unlock_reward=["regularization", "noisy_dataset"],
        xp_reward=200,
    ),
    4: LevelConfig(
        level_id=4,
        name="THE OVERFIT ARENA",
        description=(
            "A noisy dataset lies before you. A large network will memorize every noise point. "
            "Your challenge: tame the beast using regularization and prevent the train/test gap from widening."
        ),
        dataset="noisy",
        locked_params=[],
        unlocked_params=["learning_rate", "epochs", "hidden_layers", "neurons_per_layer",
                         "activation", "regularization", "reg_lambda", "batch_size"],
        default_config={
            "learning_rate": 0.01,
            "epochs": 200,
            "hidden_layers": 2,
            "neurons_per_layer": 32,
            "activation": "relu",
            "regularization": "none",
            "reg_lambda": 0.0,
            "batch_size": 32,
        },
        challenge=ChallengeDefinition(
            metric="train_test_gap",
            operator="lt",
            threshold=0.05,
            description="Keep the train/test accuracy gap below 5%.",
        ),
        unlock_reward=["imbalanced_dataset", "full_architecture"],
        xp_reward=300,
    ),
    5: LevelConfig(
        level_id=5,
        name="THE ARCHITECT CHAMBER",
        description=(
            "Face the imbalanced dataset where 85% of samples belong to one class. "
            "Accuracy alone will fool you. Design a network that achieves F1 > 0.88."
        ),
        dataset="imbalanced",
        locked_params=[],
        unlocked_params=["learning_rate", "epochs", "hidden_layers", "neurons_per_layer",
                         "activation", "regularization", "reg_lambda", "batch_size"],
        default_config={
            "learning_rate": 0.01,
            "epochs": 100,
            "hidden_layers": 1,
            "neurons_per_layer": 16,
            "activation": "relu",
            "regularization": "none",
            "reg_lambda": 0.0,
            "batch_size": 32,
        },
        challenge=ChallengeDefinition(
            metric="f1_score",
            operator="gt",
            threshold=0.88,
            description="Achieve F1 score > 0.88 on the imbalanced dataset.",
        ),
        unlock_reward=["duel_mode"],
        xp_reward=400,
    ),
    6: LevelConfig(
        level_id=6,
        name="THE DUEL",
        description=(
            "The final forge. You face a deliberately misconfigured opponent model. "
            "Choose your dataset, configure your network, and beat the opponent on "
            "both accuracy AND F1 score to claim the Master Badge."
        ),
        dataset="player_choice",
        locked_params=[],
        unlocked_params=["learning_rate", "epochs", "hidden_layers", "neurons_per_layer",
                         "activation", "regularization", "reg_lambda", "batch_size", "dataset"],
        default_config={
            "opponent": {
                "learning_rate": 10.0,
                "epochs": 5,
                "hidden_layers": 0,
                "neurons_per_layer": 2,
                "activation": "sigmoid",
                "regularization": "none",
                "reg_lambda": 0.0,
                "batch_size": 256,
            },
            "learning_rate": 0.01,
            "epochs": 100,
            "hidden_layers": 2,
            "neurons_per_layer": 16,
            "activation": "relu",
            "regularization": "l2",
            "reg_lambda": 0.001,
            "batch_size": 32,
        },
        challenge=ChallengeDefinition(
            metric="duel_win",
            operator="gt",
            threshold=0.0,
            description="Beat the opponent model on both accuracy AND F1 score.",
        ),
        unlock_reward=["master_badge", "full_forge", "replay_mode"],
        xp_reward=500,
    ),
}


def get_level(level_id: int) -> LevelConfig:
    if level_id not in LEVELS:
        raise ValueError(f"Level {level_id} does not exist. Valid levels: {list(LEVELS.keys())}")
    return LEVELS[level_id]


def get_all_levels() -> List[LevelConfig]:
    return [LEVELS[k] for k in sorted(LEVELS.keys())]
