from pydantic import BaseModel, ConfigDict
from typing import List, Optional, Dict, Any


class TrainingHistory(BaseModel):
    epochs: List[int] = []
    train_loss: List[float] = []
    test_loss: List[float] = []
    train_acc: List[float] = []
    test_acc: List[float] = []


class FailureReport(BaseModel):
    flags: List[str] = []
    severity: float = 0.0
    is_overfitting: bool = False
    is_underfitting: bool = False
    is_diverging: bool = False
    is_stalled: bool = False
    overfit_epoch: Optional[int] = None
    best_test_loss_epoch: Optional[int] = None


class ExplanationCard(BaseModel):
    title: str
    body: str
    action_hint: str
    flag_type: str = "generic"


class ChallengeResult(BaseModel):
    passed: bool
    score: float
    xp_earned: int
    message: str
    unlocked_features: List[str] = []
    is_gate_passed: bool = False
    partial_credit: float = 0.0


class BoundaryData(BaseModel):
    xx: List[List[float]] = []
    yy: List[List[float]] = []
    Z: List[List[float]] = []
    x_points: List[float] = []
    y_points: List[float] = []
    labels: List[int] = []


class WeightSnapshot(BaseModel):
    epoch: int
    weights: List[List[List[float]]] = []
    biases: List[List[List[float]]] = []
    train_loss: float
    test_loss: float
    train_acc: float
    test_acc: float


class InsightEvent(BaseModel):
    epoch: int
    event_type: str
    message: str
    severity: float = 0.0


class TrainingResult(BaseModel):
    run_id: str
    history: TrainingHistory
    final_metrics: Dict[str, float] = {}
    failure_report: FailureReport
    explanation_cards: List[ExplanationCard] = []
    boundary_data: Optional[BoundaryData] = None
    challenge_result: Optional[ChallengeResult] = None
    insight_events: List[InsightEvent] = []
    best_epoch: int = 0
    confusion_matrix: List[List[int]] = []
    weight_snapshots: List[WeightSnapshot] = []


class ProgressResponse(BaseModel):
    current_level: int
    xp: int
    unlocked_features: List[str] = []
    level_configs: List[Dict[str, Any]] = []
    completed_challenges: List[int] = []
    insights_collected: List[str] = []
    total_runs: int = 0
    best_scores: Dict[str, float] = {}


class DatasetInfo(BaseModel):
    name: str
    description: str
    purpose: str
    expected_failure: str
    n_classes: int
    n_features: int
    default_samples: int


class ExperimentResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    experiment_id: str
    model_a_result: TrainingResult
    model_b_result: TrainingResult
    lesson: str
    winner: str
    metric_comparison: Dict[str, Dict[str, float]] = {}


class LRFinderResult(BaseModel):
    learning_rates: List[float]
    losses: List[float]
    recommended_lr: float
    recommended_idx: int


class NeuronActivationData(BaseModel):
    layer_idx: int
    neuron_idx: int
    weights: List[float]
    bias: float
    activation_fn: str
    weight_importance: List[float]
