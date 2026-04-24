# Neural Forge — Backend

Pure NumPy neural network platform backend. FastAPI + in-memory storage. No ML frameworks.

---

## Quick Start

### 1. Create a virtual environment
```bash
cd neural_forge/backend
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API is now live at **http://localhost:8000**

---

## Interactive Docs

| URL | Description |
|-----|-------------|
| http://localhost:8000/docs | Swagger UI — try every endpoint |
| http://localhost:8000/redoc | ReDoc documentation |
| http://localhost:8000/health | Health check |

---

## API Reference — `/api/v1`

### Training
| Method | Path | Description |
|--------|------|-------------|
| POST | `/train` | Full training run → TrainingResult |
| POST | `/train/step` | Single epoch step (slow/interactive mode) |
| POST | `/train/stop?session_id=...` | Stop active training |
| GET  | `/train/status/{session_id}` | Poll training status |

**Minimal train request:**
```json
POST /api/v1/train
{
  "session_id": "my-session-1",
  "dataset": "xor",
  "model_config": {
    "hidden_layers": 2,
    "neurons_per_layer": 8,
    "activation": "relu",
    "regularization": "none",
    "reg_lambda": 0.001
  },
  "train_config": {
    "learning_rate": 0.01,
    "epochs": 100,
    "batch_size": 32,
    "slow_mode": false
  }
}
```

### Datasets
| Method | Path | Description |
|--------|------|-------------|
| GET  | `/datasets` | List all datasets |
| POST | `/datasets/generate` | Generate + preview a dataset |
| GET  | `/datasets/{name}/info` | Dataset description + failure mode |

Available datasets: `linear`, `xor`, `noisy`, `imbalanced`

### Experiments
| Method | Path | Description |
|--------|------|-------------|
| GET  | `/experiments` | List predefined experiments |
| POST | `/experiments/run` | Run both models, return comparison |

Available experiment IDs:
- `PERCEPTRON_VS_MLP_XOR`
- `DEPTH_EXPERIMENT`
- `REGULARIZATION_EXPERIMENT`

### Progress
| Method | Path | Description |
|--------|------|-------------|
| GET  | `/progress/{session_id}` | Full progress state |
| POST | `/progress/validate-challenge` | Submit run for challenge |
| POST | `/progress/reset` | Reset progress (dev) |

### Replay
| Method | Path | Description |
|--------|------|-------------|
| GET    | `/replay/{session_id}` | All runs + snapshots |
| GET    | `/replay/{session_id}/{run_id}` | Single run with weights |
| DELETE | `/replay/{session_id}` | Clear replay data |

---

## Architecture

```
neural_forge/backend/
├── main.py                     ← FastAPI app, CORS, routers
├── requirements.txt
├── routers/                    ← HTTP layer (thin, delegates to core)
│   ├── training.py
│   ├── datasets.py
│   ├── experiments.py
│   ├── progress.py
│   └── replay.py
├── core/                       ← Pure NumPy ML engine
│   ├── neural_net.py           ← Variable-depth network, Xavier init
│   ├── activations.py          ← sigmoid, relu, tanh, linear
│   ├── losses.py               ← BCE, MSE, categorical CE
│   ├── regularization.py       ← L1, L2, none
│   ├── optimizer.py            ← SGD + momentum + grad clipping
│   └── trainer.py              ← Full epoch loop, snapshot emission
├── data/                       ← Dataset generation + preprocessing
│   ├── datasets.py             ← linear, xor, noisy, imbalanced
│   ├── splitter.py             ← Train/test split (stratified option)
│   └── normalizer.py           ← Zero-mean unit-variance, no leakage
├── analysis/                   ← Post-training analysis
│   ├── metrics.py              ← accuracy, F1, precision, recall, gap, CM
│   ├── failure_detector.py     ← Overfitting, underfitting, diverge, stall
│   ├── boundary_computer.py    ← 100×100 mesh decision boundary
│   └── explainer.py            ← FailureReport → ExplanationCards
├── progression/                ← Gamification layer
│   ├── level_manager.py        ← 6 levels with challenges + rewards
│   ├── challenge_validator.py  ← Strict gate + partial credit
│   └── experiment_configs.py   ← 3 predefined locked experiments
├── storage/                    ← In-memory stores (thread-safe)
│   ├── session_store.py        ← Model weights + training state
│   ├── replay_store.py         ← Epoch snapshots per run
│   └── progress_store.py       ← XP, level, unlocked features
└── models/                     ← Pydantic schemas
    ├── schemas.py              ← Request models
    └── responses.py            ← Response models
```

---

## Game Levels

| Level | Name | Dataset | Challenge |
|-------|------|---------|-----------|
| 1 | The Perceptron Forge | linear | train_acc > 85% |
| 2 | The XOR Crucible | xor | test_acc > 90% |
| 3 | Depth Trials | xor | accuracy_gain > 10% vs baseline |
| 4 | The Overfit Arena | noisy | train/test gap < 5% |
| 5 | The Architect Chamber | imbalanced | F1 > 0.88 |
| 6 | The Duel | player choice | beat opponent on acc AND F1 |

---

## Notes

- **No database** — all state is in-memory. Restart resets all sessions.
- **No ML frameworks** — NumPy only. All backprop implemented by hand.
- **Session IDs** — pass any string as `session_id`; sessions are auto-created.
- **Boundary data** uses 80×80 resolution (adjustable) for API response size.
