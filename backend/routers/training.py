import uuid
import numpy as np
from fastapi import APIRouter, HTTPException
from typing import Optional

from models.schemas import TrainRequest
from models.responses import (
    TrainingResult, TrainingHistory, FinalMetrics,
    FailureReport, BoundaryData, TrainingStatus,
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
from storage.session_store import session_store
from storage.replay_store import replay_store

router = APIRouter(prefix="/train", tags=["training"])

# Track active trainers for stop support
_active_trainers: dict = {}


def _run_training(request: TrainRequest) -> TrainingResult:
    """Core training pipeline shared by /train and /train/step."""
    session_id = request.session_id
    mc = request.model_cfg
    tc = request.train_cfg

    # ── 1. Generate & split dataset ──────────────────────────────────────────
    X, y = generate_dataset(request.dataset, random_seed=42)
    stratified = request.dataset == "imbalanced"
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratified=stratified, random_seed=42
    )
    X_train, X_test, normalizer = normalize_split(X_train, X_test)

    # Persist split for replay
    session_store.set_dataset_split(session_id, {
        "X_train": X_train.tolist(),
        "X_test": X_test.tolist(),
        "y_train": y_train.tolist(),
        "y_test": y_test.tolist(),
        "dataset": request.dataset,
    })

    # ── 2. Build model ────────────────────────────────────────────────────────
    model = NeuralNetwork(
        input_size=2,
        hidden_layers=mc.hidden_layers,
        neurons_per_layer=mc.neurons_per_layer,
        output_size=1,
        activation=mc.activation,
        random_seed=42,
    )

    # ── 3. Train ──────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        learning_rate=tc.learning_rate,
        batch_size=tc.batch_size,
        regularization=mc.regularization,
        reg_lambda=mc.reg_lambda,
    )
    run_id = str(uuid.uuid4())
    _active_trainers[session_id] = trainer

    session_store.set_training_status(
        session_id, is_training=True,
        total_epochs=tc.epochs, current_epoch=0
    )

    history = trainer.train(
        X_train, y_train, X_test, y_test,
        epochs=tc.epochs,
        slow_mode=tc.slow_mode,
    )

    session_store.set_training_status(session_id, is_training=False)
    _active_trainers.pop(session_id, None)

    # ── 4. Final metrics ──────────────────────────────────────────────────────
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

    # ── 5. Failure analysis ───────────────────────────────────────────────────
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

    # ── 6. Explanations ───────────────────────────────────────────────────────
    cards = explain(failure_obj)
    explanation_cards = [
        {"title": c.title, "body": c.body, "action_hint": c.action_hint}
        for c in cards
    ]

    # ── 7. Decision boundary ──────────────────────────────────────────────────
    X_all = np.vstack([X_train, X_test])
    y_all = np.hstack([y_train, y_test])
    boundary_raw = compute_boundary_with_labels(model, X_all, y_all, resolution=80)
    boundary_data = BoundaryData(
        xx=boundary_raw["xx"],
        yy=boundary_raw["yy"],
        Z=boundary_raw["Z"],
        x_points=boundary_raw["x_points"],
        y_points=boundary_raw["y_points"],
        labels=boundary_raw["labels"],
    )

    # ── 8. Persist weights + snapshots ────────────────────────────────────────
    session_store.set_weights(session_id, model.get_weights())
    session_store.set_last_history(session_id, history)
    replay_store.save_all_snapshots(session_id, run_id, history.snapshots)

    # ── 9. Build response ─────────────────────────────────────────────────────
    training_history = TrainingHistory(
        epochs=history.epochs,
        train_loss=history.train_loss,
        test_loss=history.test_loss,
        train_acc=history.train_acc,
        test_acc=history.test_acc,
    )

    return TrainingResult(
        run_id=run_id,
        history=training_history,
        final_metrics=final_metrics,
        failure_report=failure_report,
        explanation_cards=explanation_cards,
        boundary_data=boundary_data,
        challenge_result=None,
    )


@router.post("", response_model=TrainingResult)
async def train(request: TrainRequest):
    """Full training run. Returns complete TrainingResult."""
    try:
        return _run_training(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/step")
async def train_step(request: TrainRequest):
    """
    Single-epoch step mode.
    Trains for 1 epoch and returns the partial result.
    Useful for slow/interactive mode driven by the frontend.
    """
    original_epochs = request.train_cfg.epochs
    request.train_cfg.epochs = 1
    try:
        result = _run_training(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
async def stop_training(session_id: str):
    """Signal active training session to stop after current epoch."""
    trainer = _active_trainers.get(session_id)
    if trainer:
        trainer.stop()
        session_store.set_training_status(session_id, is_training=False)
        return {"message": f"Stop signal sent to session {session_id}"}
    return {"message": "No active training session found"}


@router.get("/status/{session_id}", response_model=TrainingStatus)
async def get_training_status(session_id: str):
    """Return current training status for a session."""
    status = session_store.get_training_status(session_id)
    return TrainingStatus(
        session_id=session_id,
        is_training=status["is_training"],
        current_epoch=status["current_epoch"],
        total_epochs=status["total_epochs"],
        current_loss=status["current_loss"],
        current_acc=status["current_acc"],
    )
