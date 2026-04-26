from fastapi import APIRouter, HTTPException
from models.schemas import DatasetRequest
from data.datasets import DATASET_REGISTRY, get_dataset
from data.splitter import train_test_split
from data.normalizer import StandardNormalizer
from models.responses import DatasetInfo

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.get("")
def list_datasets():
    result = []
    for name, info in DATASET_REGISTRY.items():
        result.append(DatasetInfo(
            name=name,
            description=info["description"],
            purpose=info["purpose"],
            expected_failure=info["expected_failure"],
            n_classes=info["n_classes"],
            n_features=info["n_features"],
            default_samples=info["default_samples"],
        ))
    return result


@router.post("/generate")
def generate_dataset(req: DatasetRequest):
    try:
        X, y = get_dataset(req.name, req.n_samples, req.noise_level, req.random_seed)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        norm = StandardNormalizer()
        X_train_n = norm.fit_transform(X_train)
        X_test_n = norm.transform(X_test)
        return {
            "name": req.name,
            "n_train": len(y_train),
            "n_test": len(y_test),
            "class_balance": {
                "train_0": int((y_train == 0).sum()),
                "train_1": int((y_train == 1).sum()),
                "test_0": int((y_test == 0).sum()),
                "test_1": int((y_test == 1).sum()),
            },
            "x_sample": X_train[:10].tolist(),
            "y_sample": y_train[:10].tolist(),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{name}/info")
def dataset_info(name: str):
    if name not in DATASET_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found")
    info = DATASET_REGISTRY[name]
    return DatasetInfo(
        name=name,
        description=info["description"],
        purpose=info["purpose"],
        expected_failure=info["expected_failure"],
        n_classes=info["n_classes"],
        n_features=info["n_features"],
        default_samples=info["default_samples"],
    )
