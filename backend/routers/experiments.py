import uuid
import numpy as np
from fastapi import APIRouter, HTTPException
from typing import List

from models.schemas import ExperimentRequest
from models.responses import (
    TrainingResult, TrainingHistory, FinalMetrics,
    FailureReport, BoundaryData, ExperimentComparison,
)
from core.neural_net import NeuralNetwork
from core.trainer import Trainer
from analysis.metrics import (
    accuracy, f1_score, precision, recall,
    generalization_gap, confusion_matrix,
)
from analysis.failure_detector import analyze
from analysis.explainer import explain
from analysis.boundary_computer import compute_boundary_with_labels
from data.datasets import generate_dataset
from data.splitter import train_test_split
from data.normalizer import normalize_split
from progression.experiment_configs import get_all_experiments, get_experiment, ModelSpec

router = APIRouter(prefix="/experiments", tags=["experiments"])


def _train_model_spec(
    spec: ModelSpec,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_all: np.ndarray,
    y_all: np.ndarray,
) -> TrainingResult:
    """Train a single ModelSpec and return a full TrainingResult."""
    model = NeuralNetwork(
        input_size=2,
        hidden_layers=spec.hidden_layers,
        neurons_per_layer=spec.neurons_per_layer,
        output_size=1,
        activation=spec.activation,
        random_seed=42,
    )
    trainer = Trainer(
        model=model,
        learning_rate=spec.learning_rate,
        batch_size=spec.batch_size,
        regularization=spec.regularization,
        reg_lambda=spec.reg_lambda,
    )
    history = trainer.train(
        X_train, y_train, X_test, y_test,
        epochs=spec.epochs,
    )

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_acc = accuracy(y_train, y_pred_train)
    test_acc = accuracy(y_test, y_pred_test)
    train_loss = history.train_loss[-1] if history.train_loss else 0.0
    test_loss = history.test_loss[-1] if history.test_loss else 0.0
    f1 = f1_score(y_test, y_pred_test)
    prec = precision(y_test, y_pred_test)
    rec = recall(y_test, y_pred_test)
    gap = generalization_gap(train_acc, test_acc)
    cm = confusion_matrix(y_test, y_pred_test).tolist()

    final_metrics = FinalMetrics(
        train_accuracy=round(train_acc, 4),
        test_accuracy=round(test_acc, 4),
        train_loss=round(train_loss, 6),
        test_loss=round(test_loss, 6),
        f1_score=round(f1, 4),
        precision=round(prec, 4),
        recall=round(rec, 4),
        generalization_gap=round(gap, 4),
        confusion_matrix=cm,
    )

    failure_obj = analyze(
        history.train_loss, history.test_loss,
        history.train_acc, history.test_acc,
    )
    failure_report = FailureReport(
        flags=failure_obj.flags,
        severity=failure_obj.severity,
        is_overfitting=failure_obj.is_overfitting,
        is_underfitting=failure_obj.is_underfitting,
        is_diverging=failure_obj.is_diverging,
        is_stalled=failure_obj.is_stalled,
    )

    cards = explain(failure_obj)
    explanation_cards = [
        {"title": c.title, "body": c.body, "action_hint": c.action_hint}
        for c in cards
    ]

    boundary_raw = compute_boundary_with_labels(model, X_all, y_all, resolution=80)
    boundary_data = BoundaryData(
        xx=boundary_raw["xx"],
        yy=boundary_raw["yy"],
        Z=boundary_raw["Z"],
        x_points=boundary_raw["x_points"],
        y_points=boundary_raw["y_points"],
        labels=boundary_raw["labels"],
    )

    training_history = TrainingHistory(
        epochs=history.epochs,
        train_loss=history.train_loss,
        test_loss=history.test_loss,
        train_acc=history.train_acc,
        test_acc=history.test_acc,
    )

    return TrainingResult(
        run_id=str(uuid.uuid4()),
        history=training_history,
        final_metrics=final_metrics,
        failure_report=failure_report,
        explanation_cards=explanation_cards,
        boundary_data=boundary_data,
        challenge_result=None,
    )


@router.get("")
async def list_experiments():
    """List all predefined experiments with metadata."""
    experiments = get_all_experiments()
    return [
        {
            "experiment_id": e.experiment_id,
            "name": e.name,
            "description": e.description,
            "dataset": e.dataset,
            "hypothesis": e.hypothesis,
            "model_a_label": e.model_a.label,
            "model_b_label": e.model_b.label,
            "locked_params": e.locked_params,
        }
        for e in experiments
    ]


@router.post("/run", response_model=ExperimentComparison)
async def run_experiment(request: ExperimentRequest):
    """
    Run both models in an experiment and return comparison result.
    Both models share the same train/test split and normalization.
    """
    try:
        exp = get_experiment(request.experiment_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    try:
        # Shared data pipeline
        X, y = generate_dataset(exp.dataset, random_seed=42)
        stratified = exp.dataset == "imbalanced"
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, stratified=stratified, random_seed=42
        )
        X_train, X_test, _ = normalize_split(X_train, X_test)
        X_all = np.vstack([X_train, X_test])
        y_all = np.hstack([y_train, y_test])

        result_a = _train_model_spec(exp.model_a, X_train, y_train, X_test, y_test, X_all, y_all)
        result_b = _train_model_spec(exp.model_b, X_train, y_train, X_test, y_test, X_all, y_all)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Determine winner by test accuracy
    acc_a = result_a.final_metrics.test_accuracy
    acc_b = result_b.final_metrics.test_accuracy

    if abs(acc_a - acc_b) < 0.005:
        winner = "tie"
        summary = (
            f"Both models performed similarly ({acc_a:.3f} vs {acc_b:.3f}). "
            f"{exp.hypothesis}"
        )
    elif acc_a > acc_b:
        winner = "a"
        summary = (
            f"Model A ({exp.model_a.label}) won with test accuracy {acc_a:.3f} "
            f"vs {acc_b:.3f}. {exp.hypothesis}"
        )
    else:
        winner = "b"
        summary = (
            f"Model B ({exp.model_b.label}) won with test accuracy {acc_b:.3f} "
            f"vs {acc_a:.3f}. {exp.hypothesis}"
        )

    return ExperimentComparison(
        experiment_id=request.experiment_id,
        model_a_result=result_a,
        model_b_result=result_b,
        winner=winner,
        comparison_summary=summary,
    )
