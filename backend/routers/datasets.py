from fastapi import APIRouter, HTTPException
from typing import List
from collections import Counter

from models.schemas import DatasetRequest
from models.responses import DatasetInfo, DatasetPreview
from data.datasets import generate_dataset, DATASET_INFO
from data.splitter import train_test_split
from data.normalizer import normalize_split

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.get("", response_model=List[DatasetInfo])
async def list_datasets():
    """List all available datasets with metadata."""
    return [DatasetInfo(**info) for info in DATASET_INFO.values()]


@router.post("/generate", response_model=DatasetPreview)
async def generate(request: DatasetRequest):
    """
    Generate a dataset with optional overrides.
    Returns preview: split data, class distribution, and info.
    """
    try:
        X, y = generate_dataset(
            name=request.name,
            n_samples=request.n_samples,
            noise_level=request.noise_level,
            random_seed=request.random_seed,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    stratified = request.name == "imbalanced"
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratified=stratified, random_seed=request.random_seed
    )
    X_train_norm, X_test_norm, _ = normalize_split(X_train, X_test)

    # Class distribution across full dataset
    counts = Counter(y.tolist())
    class_dist = {str(int(k)): int(v) for k, v in sorted(counts.items())}

    info = DatasetInfo(**DATASET_INFO[request.name])

    return DatasetPreview(
        name=request.name,
        x_train=X_train_norm.tolist(),
        x_test=X_test_norm.tolist(),
        y_train=y_train.tolist(),
        y_test=y_test.tolist(),
        n_train=len(y_train),
        n_test=len(y_test),
        class_distribution=class_dist,
        info=info,
    )


@router.get("/{name}/info", response_model=DatasetInfo)
async def dataset_info(name: str):
    """Return description, purpose, and expected failure mode for a dataset."""
    if name not in DATASET_INFO:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{name}' not found. Available: {list(DATASET_INFO.keys())}"
        )
    return DatasetInfo(**DATASET_INFO[name])
