import requests
import time
from typing import Optional, Dict, Any

BASE_URL = "http://localhost:8000/api/v1"
TIMEOUT = 120


class APIError(Exception):
    pass


def _get(path: str, retries: int = 3) -> Any:
    for attempt in range(retries):
        try:
            r = requests.get(f"{BASE_URL}{path}", timeout=TIMEOUT)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.ConnectionError:
            if attempt == retries - 1:
                raise APIError("Cannot connect to backend. Is it running on port 8000?")
            time.sleep(0.5 * (attempt + 1))
        except requests.exceptions.HTTPError as e:
            raise APIError(f"HTTP {r.status_code}: {r.text}")


def _post(path: str, payload: Dict, retries: int = 3) -> Any:
    for attempt in range(retries):
        try:
            r = requests.post(f"{BASE_URL}{path}", json=payload, timeout=TIMEOUT)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.ConnectionError:
            if attempt == retries - 1:
                raise APIError("Cannot connect to backend. Is it running on port 8000?")
            time.sleep(0.5 * (attempt + 1))
        except requests.exceptions.HTTPError as e:
            raise APIError(f"HTTP {r.status_code}: {r.text}")


def _delete(path: str) -> Any:
    try:
        r = requests.delete(f"{BASE_URL}{path}", timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise APIError(str(e))


def health_check() -> bool:
    try:
        r = requests.get("http://localhost:8000/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def get_datasets():
    return _get("/datasets")


def generate_dataset(name: str, n_samples: int = 200, noise_level: float = 0.1, seed: int = 42):
    return _post("/datasets/generate", {"name": name, "n_samples": n_samples, "noise_level": noise_level, "random_seed": seed})


def get_dataset_info(name: str):
    return _get(f"/datasets/{name}/info")


def train_model(session_id: str, dataset: str, model_config: dict, train_config: dict):
    return _post("/train", {
        "session_id": session_id,
        "dataset": dataset,
        "network_config": model_config,
        "train_config": train_config,
    })


def get_train_status(session_id: str):
    return _get(f"/train/status/{session_id}")


def stop_training(session_id: str):
    return _post("/train/stop", {"session_id": session_id})  # not used but kept for API parity


def run_lr_finder(session_id: str, dataset: str, model_config: dict, lr_min=0.0001, lr_max=1.0, n_trials=10):
    return _post("/train/lr-finder", {
        "session_id": session_id,
        "dataset": dataset,
        "network_config": model_config,
        "lr_min": lr_min,
        "lr_max": lr_max,
        "n_trials": n_trials,
    })


def get_experiments():
    return _get("/experiments")


def run_experiment(experiment_id: str, session_id: str):
    return _post("/experiments/run", {"experiment_id": experiment_id, "session_id": session_id})


def get_progress(session_id: str):
    return _get(f"/progress/{session_id}")


def validate_challenge(session_id: str, level_id: int, run_id: str):
    return _post("/progress/validate-challenge", {
        "session_id": session_id,
        "level_id": level_id,
        "run_id": run_id,
    })


def reset_progress(session_id: str):
    return _post("/progress/reset", {"session_id": session_id})


def get_replay(session_id: str):
    return _get(f"/replay/{session_id}")


def get_replay_run(session_id: str, run_id: str):
    return _get(f"/replay/{session_id}/{run_id}")


def delete_replay(session_id: str):
    return _delete(f"/replay/{session_id}")
