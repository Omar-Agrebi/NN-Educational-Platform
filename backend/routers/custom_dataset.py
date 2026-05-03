from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
import numpy as np

from data.custom_dataset import (
    parse_csv, parse_xlsx, profile_columns, detect_problem_type,
    encode_features, baseline_score
)
from data.normalizer import StandardNormalizer
from data.splitter import train_test_split
from core.custom_trainer import train_custom
from storage.session_store import update_session, get_session

router = APIRouter(prefix="/custom", tags=["custom_dataset"])

# In-memory store for uploaded dataset per session
_dataset_store: Dict[str, Dict[str, Any]] = {}


# ── Upload + Profile ──────────────────────────────────────────────────────────

@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    session_id: str = Form(...),
):
    try:
        content = await file.read()
        filename = file.filename or ""

        if filename.lower().endswith(".csv"):
            headers, rows = parse_csv(content)
        elif filename.lower().endswith((".xlsx", ".xls")):
            headers, rows = parse_xlsx(content)
        else:
            raise HTTPException(status_code=400, detail="Only CSV and XLSX files are supported.")

        if len(rows) < 5:
            raise HTTPException(status_code=400, detail="File must have at least 5 data rows.")
        if len(headers) < 2:
            raise HTTPException(status_code=400, detail="File must have at least 2 columns.")

        profiles = profile_columns(headers, rows)

        # Auto-suggest target column: last non-ID column
        suggested_target = len(headers) - 1
        for i, p in enumerate(profiles):
            if not p["likely_id"] and i == len(headers) - 1:
                suggested_target = i

        _dataset_store[session_id] = {
            "headers": headers,
            "rows": rows,
            "profiles": profiles,
            "filename": filename,
            "n_rows": len(rows),
            "n_cols": len(headers),
        }

        return {
            "filename": filename,
            "n_rows": len(rows),
            "n_cols": len(headers),
            "headers": headers,
            "profiles": profiles,
            "suggested_target_col": suggested_target,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse file: {str(e)}")


# ── Auto-detect problem type ──────────────────────────────────────────────────

@router.post("/detect")
def detect_problem(session_id: str, target_col: int):
    store = _dataset_store.get(session_id)
    if not store:
        raise HTTPException(status_code=404, detail="No dataset uploaded for this session.")
    profiles = store["profiles"]
    if target_col < 0 or target_col >= len(profiles):
        raise HTTPException(status_code=400, detail="Invalid target column index.")
    detection = detect_problem_type(profiles[target_col])
    return detection


# ── Train ─────────────────────────────────────────────────────────────────────

class CustomTrainRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    session_id: str
    target_col: int
    feature_cols: List[int]
    problem_type: str  # binary_classification | multiclass_classification | regression
    # Model config
    model_type: str = Field("mlp")  # mlp | logistic | linear | perceptron
    hidden_layers: int = Field(2, ge=0, le=6)
    neurons_per_layer: List[int] = Field(default_factory=lambda: [64, 32])
    activations_per_layer: List[str] = Field(default_factory=lambda: ["relu", "relu"])
    regularization: str = Field("l2")
    reg_lambda: float = Field(0.001)
    # Train config
    learning_rate: float = Field(0.01)
    epochs: int = Field(100)
    batch_size: int = Field(32)
    test_size: float = Field(0.2)
    random_seed: int = Field(42)


@router.post("/train")
def train_custom_dataset(req: CustomTrainRequest):
    store = _dataset_store.get(req.session_id)
    if not store:
        raise HTTPException(status_code=404, detail="No dataset uploaded. Please upload first.")

    headers = store["headers"]
    rows = store["rows"]

    # Validate columns
    all_cols = set(range(len(headers)))
    if req.target_col not in all_cols:
        raise HTTPException(status_code=400, detail="Invalid target column.")
    feat_cols = [c for c in req.feature_cols if c in all_cols and c != req.target_col]
    if not feat_cols:
        raise HTTPException(status_code=400, detail="No valid feature columns selected.")

    try:
        # Encode
        encoded = encode_features(headers, rows, feat_cols, req.target_col, req.problem_type)
        X = encoded["X"]
        y = encoded["y"]
        n_classes = encoded["target_info"].get("n_classes", 2)
        feature_names = encoded["feature_names"]

        if len(X) < 10:
            raise HTTPException(status_code=400, detail="Not enough valid rows after cleaning.")

        # Subsample to avoid long training timeouts on large datasets
        MAX_ROWS = 2500
        if len(X) > MAX_ROWS:
            idx = np.random.choice(len(X), MAX_ROWS, replace=False)
            X = X[idx]
            y = y[idx]

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=req.test_size,
                                                              stratified=(req.problem_type != "regression"),
                                                              seed=req.random_seed)

        # Normalize features
        norm = StandardNormalizer()
        X_train_n = norm.fit_transform(X_train)
        X_test_n = norm.transform(X_test)

        # Baseline
        baseline = baseline_score(y_train, y_test, req.problem_type)

        # Determine architecture based on model_type
        if req.model_type == "perceptron":
            hl, npl, apl = 0, [], []
        elif req.model_type in ("logistic", "linear"):
            hl, npl, apl = 0, [], []
        else:  # mlp
            hl = req.hidden_layers
            npl = req.neurons_per_layer[:hl] if req.neurons_per_layer else [64] * hl
            apl = req.activations_per_layer[:hl] if req.activations_per_layer else ["relu"] * hl

        result = train_custom(
            X_train_n, y_train, X_test_n, y_test,
            problem_type=req.problem_type,
            n_classes=n_classes,
            hidden_layers=hl,
            neurons_per_layer=npl,
            activations_per_layer=apl,
            learning_rate=req.learning_rate,
            epochs=req.epochs,
            batch_size=req.batch_size,
            regularization=req.regularization,
            reg_lambda=req.reg_lambda,
            seed=req.random_seed,
        )

        # Feature importance mapped back to original feature names
        importance_map = {}
        fi = result["feature_importance"]
        for i, name in enumerate(feature_names):
            if i < len(fi):
                importance_map[name] = round(float(fi[i]), 4)

        # Save predictions for download
        update_session(req.session_id, "custom_predictions", {
            "predictions": result["predictions"],
            "y_test": y_test.tolist(),
            "feature_names": feature_names,
            "problem_type": req.problem_type,
            "classes": encoded["target_info"].get("classes", []),
        })

        return {
            "history": result["history"],
            "final_metrics": result["final_metrics"],
            "baseline": baseline,
            "feature_importance": importance_map,
            "metric_name": result["metric_name"],
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_features": len(feature_names),
            "feature_names": feature_names,
            "n_classes": n_classes,
            "problem_type": req.problem_type,
            "target_classes": encoded["target_info"].get("classes", []),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


# ── Download predictions ──────────────────────────────────────────────────────

@router.get("/predictions/{session_id}")
def get_predictions(session_id: str):
    sess = get_session(session_id)
    preds = sess.get("custom_predictions")
    if not preds:
        raise HTTPException(status_code=404, detail="No predictions found. Train first.")
    return preds


# ── Dataset info ──────────────────────────────────────────────────────────────

@router.get("/info/{session_id}")
def get_dataset_info(session_id: str):
    store = _dataset_store.get(session_id)
    if not store:
        raise HTTPException(status_code=404, detail="No dataset uploaded.")
    return {
        "filename": store["filename"],
        "n_rows": store["n_rows"],
        "n_cols": store["n_cols"],
        "headers": store["headers"],
        "profiles": store["profiles"],
    }
