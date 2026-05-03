from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class ModelConfig(BaseModel):
    hidden_layers: int = Field(0, ge=0, le=10)
    neurons_per_layer: List[int] = Field(default_factory=lambda: [])
    activations_per_layer: List[str] = Field(default_factory=lambda: [])  # per-layer activation
    activation: str = Field("relu")  # fallback global activation
    regularization: str = Field("none")  # none, l1, l2
    reg_lambda: float = Field(0.0, ge=0.0, le=1.0)
    dropout_rate: float = Field(0.0, ge=0.0, le=0.9)


class TrainConfig(BaseModel):
    learning_rate: float = Field(0.01, gt=0.0, le=10.0)
    epochs: int = Field(100, ge=1, le=2000)
    batch_size: int = Field(32, ge=1, le=512)
    slow_mode: bool = Field(False)
    early_stopping: bool = Field(False)
    patience: int = Field(10, ge=1, le=100)
    momentum: float = Field(0.9, ge=0.0, le=0.99)
    gradient_clip: float = Field(5.0, ge=0.1, le=100.0)


class DatasetRequest(BaseModel):
    name: str
    n_samples: int = Field(200, ge=50, le=2000)
    noise_level: float = Field(0.1, ge=0.0, le=1.0)
    random_seed: int = Field(42)


class TrainRequest(BaseModel):
    session_id: str
    dataset: str
    network_config: ModelConfig
    train_config: TrainConfig
    run_id: Optional[str] = None


class ExperimentRequest(BaseModel):
    experiment_id: str
    session_id: str


class ChallengeSubmitRequest(BaseModel):
    session_id: str
    level_id: int
    run_id: str


class ProgressRequest(BaseModel):
    session_id: str


class SandboxRequest(BaseModel):
    session_id: str
    dataset: str
    network_config: ModelConfig
    train_config: TrainConfig


class LRFinderRequest(BaseModel):
    session_id: str
    dataset: str
    network_config: ModelConfig
    lr_min: float = Field(0.0001)
    lr_max: float = Field(1.0)
    n_trials: int = Field(10)
