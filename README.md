# ⚙ NEURAL FORGE

A gamified, educational neural network platform. Break models. Fix them. Master them.

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 2. Start the Backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

You should see:
```
INFO: Uvicorn running on http://127.0.0.1:8000
```

Verify it works: http://localhost:8000/health

### 3. Start the Frontend (new terminal)

```bash
cd frontend
streamlit run app.py
```

Opens automatically at: http://localhost:8501

---

## 📁 Structure

```
neural_forge/
├── backend/          FastAPI backend (NumPy-only ML)
│   ├── main.py
│   ├── core/         Neural net, activations, losses, trainer
│   ├── data/         4 datasets, splitter, normalizer
│   ├── analysis/     Failure detector, boundary, metrics, explainer
│   ├── progression/  Level system, challenge validator, experiments
│   ├── storage/      In-memory session, replay, progress stores
│   ├── routers/      REST API endpoints
│   └── models/       Pydantic schemas and responses
├── frontend/         Streamlit frontend
│   ├── app.py        Entry point + World Map
│   ├── pages/        6 level pages
│   ├── components/   Reusable UI components
│   ├── styles/       Dark theme CSS
│   └── utils/        API client, state manager, formatters
└── requirements.txt
```

---

## 🎮 Level Guide

| Level | Name | Dataset | Challenge |
|-------|------|---------|-----------|
| 1 | The Perceptron Forge | Linear | >85% test accuracy |
| 2 | The XOR Crucible | XOR | >90% test accuracy |
| 3 | Depth Trials | XOR | >10% accuracy gain by adding depth |
| 4 | The Overfit Arena | Noisy | Train/test gap <5% |
| 5 | The Architect Chamber | Imbalanced | F1 >0.88 |
| 6 | The Duel | Your choice | Beat opponent on accuracy AND F1 |

---

## ✨ Key Features

- **Per-layer activation functions** — choose ReLU, Sigmoid, Tanh, ELU, Leaky ReLU independently for each hidden layer
- **LR Finder** — sweep 10 learning rates and find the optimal one automatically  
- **Weight Inspector** — click any layer to see weights as a heatmap with importance bars
- **Replay System** — scrub through training history epoch by epoch
- **Diagnosis Mode** — automatic failure detection with fix suggestions
- **Insight Coins** — collectible discovery moments for key learning events
- **XP Bar** — always visible below the Streamlit toolbar (not hidden behind it)
- **Tips Button** — level-specific hints when you're stuck
- **Parameter DNA** — visual fingerprint of your model config
- **Experiment Mode** — controlled side-by-side comparisons
- **The Duel** — final boss with HP bars and cinematic reveal

---

## 🛠 Tech Stack

- **Backend**: Python 3.10+, FastAPI, NumPy only (no sklearn/PyTorch/TensorFlow)
- **Frontend**: Streamlit, Plotly
- **Storage**: In-memory (progress persists to `/tmp/neural_forge_progress.json`)

---

## ⚙ Custom Forge — Bring Your Own Data

Access from the World Map → **Custom Forge** button.

### Supported formats
- CSV (any delimiter, UTF-8)
- XLSX / XLS (first sheet used)

### Supported problem types
| Type | Detection | Output |
|------|-----------|--------|
| Binary Classification | 2 unique target values | Accuracy, confusion matrix |
| Multi-class Classification | 3–20 unique categorical values | Accuracy, confusion matrix |
| Regression | Continuous numeric target | R² score, MSE |

### Features
- **Auto-detection** of problem type with confidence score
- **Column profiler** — missing values, unique counts, type inference
- **Feature selector** — include/exclude columns, auto-excludes ID columns
- **One-hot encoding** for categorical features, mean imputation for missing numerics
- **Baseline comparison** — always shows a "dumb" baseline so you know if your model actually learned
- **Feature importance** — weight magnitude visualization
- **Download predictions** as CSV
- **Confusion matrix** for classification problems
- **Per-layer activation selection** on the MLP

### New endpoint
`POST /api/v1/custom/upload` — upload file  
`POST /api/v1/custom/detect` — auto-detect problem type  
`POST /api/v1/custom/train` — train model  
`GET  /api/v1/custom/predictions/{session_id}` — fetch predictions  
`GET  /api/v1/custom/info/{session_id}` — dataset info  
