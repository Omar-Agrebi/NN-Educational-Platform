from pydantic import BaseModel, Field
from typing import Optional, List


class ModelConfig(BaseModel):
    hidden_layers: int = Field(default=0, ge=0, le=10)
    neurons_per_layer: int = Field(default=8, ge=1, le=256)
    activation: str = Field(default="relu")  # sigmoid, relu, tanh, linear
    regularization: str = Field(default="none")  # none, l1, l2
    reg_lambda: float = Field(default=0.001, ge=0.0, le=1.0)


class TrainConfig(BaseModel):
    learning_rate: float = Field(default=0.01, gt=0.0, le=10.0)
    epochs: int = Field(default=100, ge=1, le=2000)
    batch_size: int = Field(default=32, ge=1, le=512)
    slow_mode: bool = Field(default=False)


class TrainRequest(BaseModel):
    session_id: str
    dataset: str = Field(default="linear")  # linear, xor, noisy, imbalanced
    model_cfg: ModelConfig = Field(default_factory=ModelConfig)
    train_cfg: TrainConfig = Field(default_factory=TrainConfig)


class DatasetRequest(BaseModel):
    name: str  # linear, xor, noisy, imbalanced
    n_samples: int = Field(default=200, ge=10, le=5000)
    noise_level: float = Field(default=0.1, ge=0.0, le=1.0)
    random_seed: int = Field(default=42)


class ExperimentRequest(BaseModel):
    experiment_id: str  # PERCEPTRON_VS_MLP_XOR, DEPTH_EXPERIMENT, REGULARIZATION_EXPERIMENT
    session_id: str


class ChallengeSubmitRequest(BaseModel):
    session_id: str
    level_id: int
    run_id: str


class ProgressRequest(BaseModel):
    session_id: str
