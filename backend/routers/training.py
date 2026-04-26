from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from models.schemas import TrainRequest, LRFinderRequest
from models.responses import TrainingResult, FailureReport, BoundaryData, LRFinderResult
from core.neural_net import NeuralNetwork
from core.trainer import Trainer
from core.losses import binary_crossentropy
from data.datasets import get_dataset
from data.splitter import train_test_split
from data.normalizer import StandardNormalizer
from analysis.failure_detector import analyze
from analysis.boundary_computer import compute_boundary
from analysis.explainer import explain
from analysis.metrics import compute_all_metrics
from storage.session_store import get_session, update_session, set_stop_flag, clear_stop_flag
from storage.replay_store import save_snapshots
from storage.progress_store import get_progress, update_progress, add_insight
import numpy as np
import uuid
import time

router = APIRouter(prefix="/train", tags=["training"])
_active_trainers = {}


def _run_training(req: TrainRequest):
    session = get_session(req.session_id)
    clear_stop_flag(req.session_id)
    update_session(req.session_id, "training_running", True)

    X, y = get_dataset(req.dataset, n_samples=None, noise=None, seed=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratified=True)
    norm = StandardNormalizer()
    X_train_n = norm.fit_transform(X_train)
    X_test_n = norm.transform(X_test)

    model = NeuralNetwork.from_config(req.nn_config)
    trainer = Trainer()
    _active_trainers[req.session_id] = trainer

    result = trainer.train(
        model=model,
        X_train=X_train_n,
        y_train=y_train,
        X_test=X_test_n,
        y_test=y_test,
        learning_rate=req.train_config.learning_rate,
        epochs=req.train_config.epochs,
        batch_size=req.train_config.batch_size,
        regularization=req.nn_config.regularization,
        reg_lambda=req.nn_config.reg_lambda,
        momentum=req.train_config.momentum,
        gradient_clip=req.train_config.gradient_clip,
        early_stopping=req.train_config.early_stopping,
        patience=req.train_config.patience,
        slow_mode=req.train_config.slow_mode,
    )

    # Failure analysis
    h = result["history"]
    failure_report = analyze(h.train_loss, h.test_loss, h.train_acc, h.test_acc)
    explanation_cards = explain(failure_report)

    # Boundary
    all_X = np.vstack([X_train_n, X_test_n])
    all_y = np.hstack([y_train, y_test])
    boundary_data = compute_boundary(model, all_X, all_y)

    # Metrics
    test_pred = model.forward(X_test_n, training=False)
    final_metrics = compute_all_metrics(y_test, test_pred)
    final_metrics["gen_gap"] = round(abs(h.train_acc[-1] - h.test_acc[-1]), 4)
    final_metrics["best_epoch"] = result["best_epoch"]

    # Confusion matrix
    test_pred_bin = (test_pred >= 0.5).astype(int).flatten()
    from analysis.metrics import confusion_matrix as cm
    conf_mat = cm(y_test.astype(int), test_pred_bin)

    # Insights
    progress = get_progress(req.session_id)
    insights_collected = progress.get("insights_collected", [])
    new_insights = []
    if failure_report.is_overfitting and "overfit_witnessed" not in insights_collected:
        add_insight(req.session_id, "overfit_witnessed")
        new_insights.append("overfit_witnessed")
    if req.nn_config.hidden_layers == 0 and req.dataset == "xor" and "xor_failure_witnessed" not in insights_collected:
        add_insight(req.session_id, "xor_failure_witnessed")
        new_insights.append("xor_failure_witnessed")
    if req.nn_config.regularization != "none" and "regularization_used" not in insights_collected:
        add_insight(req.session_id, "regularization_used")
        new_insights.append("regularization_used")

    # Save replay
    run_id = result["run_id"]
    save_snapshots(req.session_id, run_id, result["weight_snapshots"])

    # Update session
    update_session(req.session_id, "training_running", False)
    update_session(req.session_id, "last_run_id", run_id)
    total_runs = progress.get("total_runs", 0) + 1
    update_progress(req.session_id, {"total_runs": total_runs})

    training_result = TrainingResult(
        run_id=run_id,
        history=h,
        final_metrics=final_metrics,
        failure_report=failure_report,
        explanation_cards=explanation_cards,
        boundary_data=boundary_data,
        insight_events=result["insight_events"],
        best_epoch=result["best_epoch"],
        confusion_matrix=conf_mat,
        weight_snapshots=result["weight_snapshots"][:5],  # limit for response size
    )

    update_session(req.session_id, "last_result", training_result.dict())
    return training_result


@router.post("")
def train(req: TrainRequest):
    try:
        return _run_training(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class StopRequest(BaseModel):
    session_id: str

@router.post("/stop")
def stop_training(req: StopRequest):
    sid = req.session_id
    if sid in _active_trainers:
        _active_trainers[sid].stop()
        set_stop_flag(sid)
    return {"message": "Stop signal sent"}


@router.get("/status/{session_id}")
def get_status(session_id: str):
    session = get_session(session_id)
    return {
        "training_running": session.get("training_running", False),
        "last_run_id": session.get("last_run_id"),
    }


@router.post("/lr-finder")
def lr_finder(req: LRFinderRequest):
    try:
        X, y = get_dataset(req.dataset)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        norm = StandardNormalizer()
        X_train_n = norm.fit_transform(X_train)
        X_test_n = norm.transform(X_test)

        lrs = np.logspace(np.log10(req.lr_min), np.log10(req.lr_max), req.n_trials)
        losses = []
        for lr in lrs:
            model = NeuralNetwork.from_config(req.nn_config)
            trainer = Trainer()
            result = trainer.train(
                model=model, X_train=X_train_n, y_train=y_train,
                X_test=X_test_n, y_test=y_test,
                learning_rate=float(lr), epochs=20, batch_size=32,
            )
            losses.append(result["history"].train_loss[-1] if result["history"].train_loss else 999)

        losses_arr = np.array(losses)
        best_idx = int(np.argmin(losses_arr))
        return LRFinderResult(
            learning_rates=lrs.tolist(),
            losses=losses_arr.tolist(),
            recommended_lr=float(lrs[best_idx]),
            recommended_idx=best_idx,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
