import time
import requests
import streamlit as st
from typing import Optional

BASE_URL = "http://localhost:8000/api/v1"
MAX_RETRIES = 3


class APIError(Exception):
    def __init__(self, message: str, status_code: int = 0):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


def _request(method: str, path: str, **kwargs) -> dict:
    url = f"{BASE_URL}{path}"
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.request(method, url, timeout=60, **kwargs)
            if resp.status_code >= 500:
                raise APIError(f"Server error {resp.status_code}: {resp.text}", resp.status_code)
            if resp.status_code >= 400:
                try:
                    detail = resp.json().get("detail", resp.text)
                except Exception:
                    detail = resp.text
                raise APIError(f"API error {resp.status_code}: {detail}", resp.status_code)
            return resp.json()
        except APIError:
            raise
        except requests.exceptions.ConnectionError:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                raise APIError("Cannot connect to Neural Forge backend at localhost:8000. Is it running?")
        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                raise APIError("Backend request timed out.")
        except Exception as e:
            raise APIError(str(e))


def _safe(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except APIError as e:
        st.error(f"⚠ API Error: {e.message}")
        return None


def get_datasets():
    return _request("GET", "/datasets")


def generate_dataset(name: str, n_samples: int = 200, noise_level: float = 0.1, seed: int = 42):
    return _request("POST", "/datasets/generate", json={
        "name": name, "n_samples": n_samples, "noise_level": noise_level, "random_seed": seed
    })


def train_model(session_id: str, dataset: str, model_config: dict, train_config: dict):
    return _request("POST", "/train", json={
        "session_id": session_id,
        "dataset": dataset,
        "model_cfg": model_config,
        "train_cfg": train_config,
    })


def get_train_status(session_id: str):
    return _request("GET", f"/train/status/{session_id}")


def stop_training(session_id: str):
    return _request("POST", f"/train/stop", params={"session_id": session_id})


def get_experiments():
    return _request("GET", "/experiments")


def run_experiment(experiment_id: str, session_id: str):
    return _request("POST", "/experiments/run", json={
        "experiment_id": experiment_id,
        "session_id": session_id,
    })


def get_progress(session_id: str):
    return _request("GET", f"/progress/{session_id}")


def validate_challenge(session_id: str, level_id: int, run_id: str):
    return _request("POST", "/progress/validate-challenge", json={
        "session_id": session_id,
        "level_id": level_id,
        "run_id": run_id,
    })


def reset_progress(session_id: str):
    return _request("POST", "/progress/reset", json={"session_id": session_id})


def get_replay(session_id: str):
    return _request("GET", f"/replay/{session_id}")


def get_replay_run(session_id: str, run_id: str):
    return _request("GET", f"/replay/{session_id}/{run_id}")


# Safe wrappers (show st.error instead of raising)
def safe_train(session_id, dataset, model_config, train_config):
    return _safe(train_model, session_id, dataset, model_config, train_config)


def safe_get_progress(session_id):
    return _safe(get_progress, session_id)


def safe_validate(session_id, level_id, run_id):
    return _safe(validate_challenge, session_id, level_id, run_id)


def safe_run_experiment(experiment_id, session_id):
    return _safe(run_experiment, experiment_id, session_id)
