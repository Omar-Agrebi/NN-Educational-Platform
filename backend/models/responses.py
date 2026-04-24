from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class TrainingHistory(BaseModel):
    epochs: List[int]
    train_loss: List[float]
    test_loss: List[float]
    train_acc: List[float]
    test_acc: List[float]


class FailureReport(BaseModel):
    flags: List[str]
    severity: float
    is_overfitting: bool
    is_underfitting: bool
    is_diverging: bool
    is_stalled: bool


class ExplanationCard(BaseModel):
    title: str
    body: str
    action_hint: str


class ChallengeResult(BaseModel):
    passed: bool
    is_gate_passed: bool
    score: float
    xp_earned: int
    message: str
    unlocked_features: List[str]


class BoundaryData(BaseModel):
    xx: List[List[float]]
    yy: List[List[float]]
    Z: List[List[float]]
    x_points: List[float]
    y_points: List[float]
    labels: List[int]


class FinalMetrics(BaseModel):
    train_accuracy: float
    test_accuracy: float
    train_loss: float
    test_loss: float
    f1_score: float
    precision: float
    recall: float
    generalization_gap: float
    confusion_matrix: List[List[int]]


class TrainingResult(BaseModel):
    run_id: str
    history: TrainingHistory
    final_metrics: FinalMetrics
    failure_report: FailureReport
    explanation_cards: List[ExplanationCard]
    boundary_data: Optional[BoundaryData]
    challenge_result: Optional[ChallengeResult]


class LevelConfig(BaseModel):
    level_id: int
    name: str
    description: str
    dataset: str
    locked_params: List[str]
    unlocked_params: List[str]
    default_config: Dict[str, Any]
    xp_reward: int
    challenge_description: str
    unlock_reward: List[str]


class ProgressResponse(BaseModel):
    current_level: int
    xp: int
    unlocked_features: List[str]
    level_configs: List[LevelConfig]
    completed_challenges: List[int]


class DatasetInfo(BaseModel):
    name: str
    description: str
    purpose: str
    expected_failure_mode: str
    default_samples: int
    n_features: int
    n_classes: int


class DatasetPreview(BaseModel):
    name: str
    x_train: List[List[float]]
    x_test: List[List[float]]
    y_train: List[int]
    y_test: List[int]
    n_train: int
    n_test: int
    class_distribution: Dict[str, int]
    info: DatasetInfo


class ExperimentComparison(BaseModel):
    experiment_id: str
    model_a_result: TrainingResult
    model_b_result: TrainingResult
    winner: str  # "a", "b", "tie"
    comparison_summary: str


class TrainingStatus(BaseModel):
    session_id: str
    is_training: bool
    current_epoch: int
    total_epochs: int
    current_loss: Optional[float]
    current_acc: Optional[float]
