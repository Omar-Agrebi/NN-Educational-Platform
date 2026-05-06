import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import io
import json
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from styles.theme import inject_theme, xp_bar, card_title
from utils import state_manager as sm
from utils.api_client import APIError

BASE_URL = "http://localhost:8000/api/v1"

inject_theme()
sm.init_state()

# Track active level to clear results on switch
LEVEL_ID = "custom"
last_active = sm.get("active_level_id")
if last_active != LEVEL_ID:
    sm.set("active_level_id", LEVEL_ID)
    sm.reset_ephemeral_state()

# ── XP Bar ────────────────────────────────────────────────────────────────────
xp_bar(sm.get("current_level", 1), sm.get("xp", 0), sm.get_session_id(), sm.get("insights", []))

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='font-family:var(--display);font-size:12px;color:var(--orange);"
        "letter-spacing:2px;margin-bottom:12px'>CUSTOM FORGE</div>",
        unsafe_allow_html=True,
    )
    if st.button("← World Map", use_container_width=True):
        st.switch_page("app.py")
    st.divider()
    st.markdown("**Upload** your own CSV or XLSX dataset and train a neural network on it.")
    st.markdown("Supports:")
    st.markdown("- Binary classification")
    st.markdown("- Multi-class classification")
    st.markdown("- Regression")

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:24px">
  <div style="display:inline-flex;align-items:center;gap:8px;
    font-family:var(--mono);font-size:10px;color:var(--orange);
    border:1px solid rgba(255,107,43,0.3);border-radius:4px;
    padding:3px 12px;letter-spacing:2px;margin-bottom:10px;
    background:rgba(255,107,43,0.05)">
    ◈ CUSTOM FORGE — BRING YOUR OWN DATA
  </div>
  <div style="font-family:var(--display);font-size:26px;font-weight:900;
    color:#fff;letter-spacing:2px">
    CUSTOM <span style="color:var(--orange)">FORGE</span>
  </div>
  <div style="font-size:13px;color:var(--muted);margin-top:6px;font-weight:300">
    Upload a CSV or XLSX dataset, configure your model, and see real results on your own data.
  </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — UPLOAD
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("### Step 1 — Upload Dataset")

uploaded_file = st.file_uploader(
    "Drop your CSV or XLSX file here",
    type=["csv", "xlsx", "xls"],
    help="Max recommended size: 10,000 rows × 50 columns. First row must be headers.",
    key="custom_upload",
)

if uploaded_file is not None:
    # Upload to backend
    if sm.get("custom_upload_filename") != uploaded_file.name:
        with st.spinner("Parsing and profiling your dataset…"):
            try:
                response = requests.post(
                    f"{BASE_URL}/custom/upload",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                    data={"session_id": sm.get_session_id()},
                    timeout=60,
                )
                if response.status_code == 200:
                    upload_result = response.json()
                    sm.set("custom_upload_result", upload_result)
                    sm.set("custom_upload_filename", uploaded_file.name)
                    sm.set("custom_train_result", None)  # reset on new upload
                else:
                    st.error(f"Upload failed: {response.json().get('detail', 'Unknown error')}")
                    st.stop()
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to backend. Start it with: `uvicorn main:app --reload --port 8000`")
                st.stop()

upload_result = sm.get("custom_upload_result")

if not upload_result:
    st.markdown(
        "<div style='background:var(--card2);border:1px dashed var(--border);border-radius:10px;"
        "padding:32px;text-align:center;color:var(--muted);font-size:13px'>"
        "Upload a CSV or XLSX file to get started."
        "</div>",
        unsafe_allow_html=True,
    )
    st.stop()

# ── Dataset Summary ───────────────────────────────────────────────────────────
profiles = upload_result.get("profiles", [])
headers = upload_result.get("headers", [])
n_rows = upload_result.get("n_rows", 0)
n_cols = upload_result.get("n_cols", 0)

col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
with col_sum1:
    st.metric("Rows", f"{n_rows:,}")
with col_sum2:
    st.metric("Columns", n_cols)
with col_sum3:
    numeric_cols = sum(1 for p in profiles if p["is_numeric"])
    st.metric("Numeric cols", numeric_cols)
with col_sum4:
    missing_cols = sum(1 for p in profiles if p["missing_pct"] > 0)
    st.metric("Cols with missing", missing_cols)

# ── Column Profiler ───────────────────────────────────────────────────────────
with st.expander("📊 Column Profiler", expanded=True):
    card_title("Column Overview")

    # Missing value heatmap
    fig_miss = go.Figure()
    col_names = [p["name"][:20] for p in profiles]
    miss_pcts = [p["missing_pct"] for p in profiles]

    fig_miss.add_trace(go.Bar(
        x=col_names,
        y=miss_pcts,
        marker_color=["#ff3b5c" if v > 20 else "#ff6b2b" if v > 5 else "#00ff88" for v in miss_pcts],
        name="Missing %",
    ))
    fig_miss.update_layout(
        paper_bgcolor="#111827", plot_bgcolor="#0a0f1e",
        height=160, margin=dict(l=8, r=8, t=8, b=40),
        xaxis=dict(tickfont=dict(size=8, color="#4a5568", family="Share Tech Mono"),
                   gridcolor="#1f2d48"),
        yaxis=dict(title="Missing %", gridcolor="#1f2d48",
                   tickfont=dict(size=8, color="#4a5568", family="Share Tech Mono")),
        showlegend=False,
    )
    st.plotly_chart(fig_miss, use_container_width=True, key="miss_chart")

    # Column detail table
    rows_table = []
    for p in profiles:
        kind = "numeric" if p["is_numeric"] else "categorical"
        missing_str = f"{p['missing_pct']}%" if p["missing_pct"] > 0 else "✓ none"
        uniq = p["n_unique"]
        sample = ", ".join(str(v) for v in p["unique_values"][:4])
        rows_table.append([p["name"], kind, missing_str, uniq, sample])

    fig_table = go.Figure(data=[go.Table(
        header=dict(
            values=["<b>Column</b>", "<b>Type</b>", "<b>Missing</b>", "<b>Unique</b>", "<b>Sample values</b>"],
            fill_color="#1f2d48", font=dict(color="#c9d4e8", size=10, family="Share Tech Mono"),
            align="left", height=28,
        ),
        cells=dict(
            values=list(zip(*rows_table)) if rows_table else [[]]*5,
            fill_color=["#111827", "#0f1d2f"],
            font=dict(color="#c9d4e8", size=9, family="Share Tech Mono"),
            align="left", height=24,
        ),
    )])
    fig_table.update_layout(
        paper_bgcolor="#111827", margin=dict(l=0, r=0, t=0, b=0), height=min(40 + len(profiles) * 24, 320),
    )
    st.plotly_chart(fig_table, use_container_width=True, key="col_table")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — CONFIGURE
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### Step 2 — Configure")

col_cfg1, col_cfg2 = st.columns([1, 1])

with col_cfg1:
    # Target column
    card_title("Target Column")
    suggested = upload_result.get("suggested_target_col", len(headers) - 1)
    target_col = st.selectbox(
        "Which column are you predicting?",
        options=list(range(len(headers))),
        format_func=lambda i: f"{headers[i]} ({profiles[i]['n_unique']} unique)",
        index=suggested,
        key="target_col_select",
    )

    # Problem type
    card_title("Problem Type")

    # Auto-detect
    try:
        det_resp = requests.post(
            f"{BASE_URL}/custom/detect",
            params={"session_id": sm.get_session_id(), "target_col": target_col},
            timeout=10,
        )
        if det_resp.status_code == 200:
            detection = det_resp.json()
            det_type = detection.get("type", "binary_classification")
            det_conf = detection.get("confidence", 0.5)
            det_reason = detection.get("reason", "")
        else:
            det_type = "binary_classification"; det_conf = 0.5; det_reason = ""
    except Exception:
        det_type = "binary_classification"; det_conf = 0.5; det_reason = ""

    conf_color = "var(--green)" if det_conf > 0.85 else "var(--orange)" if det_conf > 0.6 else "var(--muted)"
    st.markdown(
        f"<div style='background:rgba(0,0,0,0.2);border:1px solid var(--border);border-radius:6px;"
        f"padding:10px 14px;margin-bottom:10px'>"
        f"<div style='font-family:var(--mono);font-size:9px;color:var(--muted);letter-spacing:1px'>AUTO-DETECTED</div>"
        f"<div style='font-size:13px;color:{conf_color};font-weight:700;margin:3px 0'>"
        f"{det_type.replace('_',' ').title()} ({int(det_conf*100)}% confident)</div>"
        f"<div style='font-size:10px;color:var(--muted)'>{det_reason}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    PROBLEM_TYPES = {
        "binary_classification": "Binary Classification",
        "multiclass_classification": "Multi-class Classification",
        "regression": "Regression",
    }
    problem_type = st.selectbox(
        "Override problem type",
        options=list(PROBLEM_TYPES.keys()),
        format_func=lambda x: PROBLEM_TYPES[x],
        index=list(PROBLEM_TYPES.keys()).index(det_type) if det_type in PROBLEM_TYPES else 0,
        key="problem_type_select",
    )

    # Feature columns
    card_title("Feature Columns")
    st.caption("Uncheck ID columns, free-text fields, or anything irrelevant.")
    all_feature_indices = [i for i in range(len(headers)) if i != target_col]
    default_features = [i for i in all_feature_indices if not profiles[i]["likely_id"] and profiles[i]["missing_pct"] < 50]

    feature_cols = []
    for i in all_feature_indices:
        p = profiles[i]
        label = f"{p['name']} — {'numeric' if p['is_numeric'] else 'categorical'}, {p['n_unique']} unique"
        default_on = i in default_features
        checked = st.checkbox(label, value=default_on, key=f"feat_{i}")
        if checked:
            feature_cols.append(i)

with col_cfg2:
    # Model selection
    card_title("Model")
    MODEL_OPTIONS = {
        "mlp": "MLP (Multi-Layer Perceptron)",
        "logistic": "Logistic Regression (single layer)",
        "linear": "Linear Regression (single layer)",
        "perceptron": "Perceptron (single neuron, no activation)",
    }
    # Filter by problem type
    valid_models = ["mlp"]
    if problem_type == "binary_classification":
        valid_models = ["mlp", "logistic", "perceptron"]
    elif problem_type == "multiclass_classification":
        valid_models = ["mlp", "logistic"]
    elif problem_type == "regression":
        valid_models = ["mlp", "linear"]

    model_type = st.selectbox(
        "Model architecture",
        options=valid_models,
        format_func=lambda x: MODEL_OPTIONS[x],
        key="model_type_select",
    )

    # Architecture (shown only for MLP)
    hidden_layers = 0
    neurons_list = []
    activations_list = []

    if model_type == "mlp":
        hidden_layers = st.slider("Hidden layers", 0, 5, 2, key="hl_custom")
        for i in range(hidden_layers):
            c1, c2 = st.columns(2)
            with c1:
                n = st.slider(f"Layer {i+1} neurons", 4, 256, 64, key=f"n_custom_{i}")
                neurons_list.append(n)
            with c2:
                a = st.selectbox(f"Layer {i+1} activation",
                                 ["relu", "tanh", "sigmoid", "leaky_relu", "elu"],
                                 key=f"a_custom_{i}")
                activations_list.append(a)

    card_title("Training Config")
    lr = st.slider("Learning rate", 0.0001, 1.0, 0.01, step=0.0001, format="%.4f", key="lr_custom")
    epochs = st.slider("Epochs", 10, 500, 100, step=10, key="ep_custom")
    batch_size = st.select_slider("Batch size", [8, 16, 32, 64, 128, 256], value=32, key="bs_custom")
    test_size = st.slider("Test split", 0.1, 0.4, 0.2, step=0.05, key="ts_custom")
    reg_type = st.selectbox("Regularization", ["none", "l1", "l2"], key="reg_custom")
    reg_lambda = 0.0
    if reg_type != "none":
        reg_lambda = st.slider("λ", 0.0001, 0.5, 0.001, step=0.0001, format="%.4f", key="rl_custom")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — TRAIN
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### Step 3 — Train & Evaluate")

if not feature_cols:
    st.warning("Select at least one feature column above.")
    st.stop()

train_clicked = st.button("⚙ TRAIN MODEL", type="primary", use_container_width=True, key="train_custom")

if train_clicked:
    payload = {
        "session_id": sm.get_session_id(),
        "target_col": target_col,
        "feature_cols": feature_cols,
        "problem_type": problem_type,
        "model_type": model_type,
        "hidden_layers": hidden_layers,
        "neurons_per_layer": neurons_list,
        "activations_per_layer": activations_list,
        "regularization": reg_type,
        "reg_lambda": reg_lambda,
        "learning_rate": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "test_size": test_size,
        "random_seed": 42,
    }
    with st.spinner("Training your model…"):
        try:
            resp = requests.post(f"{BASE_URL}/custom/train", json=payload, timeout=300)
            if resp.status_code == 200:
                sm.set("custom_train_result", resp.json())
            else:
                st.error(f"Training failed: {resp.json().get('detail', 'Unknown error')}")
        except requests.exceptions.ConnectionError:
            st.error("Cannot reach backend.")
        except Exception as e:
            st.error(f"Error: {e}")

# ── Results ───────────────────────────────────────────────────────────────────
result = sm.get("custom_train_result")

if not result:
    st.markdown(
        "<div style='background:var(--card2);border:1px dashed var(--border);border-radius:10px;"
        "padding:28px;text-align:center;color:var(--muted);font-size:13px'>"
        "Configure your model above and click Train to see results here."
        "</div>",
        unsafe_allow_html=True,
    )
    st.stop()

# ── Metric cards ──────────────────────────────────────────────────────────────
fm = result.get("final_metrics", {})
baseline = result.get("baseline", {})
metric_name = result.get("metric_name", "accuracy")
problem = result.get("problem_type", "binary_classification")
is_regression = problem == "regression"

# Primary metric display
train_m = fm.get("train_metric", 0)
test_m  = fm.get("test_metric", 0)
bl_acc  = baseline.get("accuracy", baseline.get("r2", 0))
bl_strategy = baseline.get("strategy", "")

# Color: green if test > baseline by >5%, orange if close, red if worse
beat_baseline = test_m > bl_acc + 0.05 if not is_regression else test_m > bl_acc + 0.05

mc1, mc2, mc3, mc4 = st.columns(4)
with mc1:
    metric_label = "Train " + ("Accuracy" if not is_regression else "R²")
    color = "var(--cyan)"
    st.markdown(
        f"<div style='background:var(--card);border:1px solid var(--border);border-radius:10px;padding:16px;text-align:center'>"
        f"<div style='font-family:var(--mono);font-size:9px;color:var(--muted);letter-spacing:1px'>{metric_label}</div>"
        f"<div style='font-family:var(--display);font-size:32px;font-weight:900;color:{color}'>"
        f"{train_m*100:.1f}{'%' if not is_regression else ''}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
with mc2:
    metric_label = "Test " + ("Accuracy" if not is_regression else "R²")
    color = "var(--green)" if beat_baseline else "var(--orange)"
    st.markdown(
        f"<div style='background:var(--card);border:2px solid {color};border-radius:10px;padding:16px;text-align:center;"
        f"box-shadow:0 0 16px {color}22'>"
        f"<div style='font-family:var(--mono);font-size:9px;color:var(--muted);letter-spacing:1px'>{metric_label}</div>"
        f"<div style='font-family:var(--display);font-size:32px;font-weight:900;color:{color}'>"
        f"{test_m*100:.1f}{'%' if not is_regression else ''}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
with mc3:
    bl_label = "Baseline " + ("Accuracy" if not is_regression else "R²")
    st.markdown(
        f"<div style='background:var(--card);border:1px solid var(--border);border-radius:10px;padding:16px;text-align:center'>"
        f"<div style='font-family:var(--mono);font-size:9px;color:var(--muted);letter-spacing:1px'>{bl_label}</div>"
        f"<div style='font-family:var(--display);font-size:32px;font-weight:900;color:var(--muted)'>"
        f"{bl_acc*100:.1f}{'%' if not is_regression else ''}</div>"
        f"<div style='font-size:9px;color:var(--muted);margin-top:4px'>{bl_strategy}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
with mc4:
    lift = test_m - bl_acc
    lift_color = "var(--green)" if lift > 0.05 else "var(--orange)" if lift > 0 else "var(--red)"
    lift_label = "Lift vs Baseline"
    st.markdown(
        f"<div style='background:var(--card);border:1px solid var(--border);border-radius:10px;padding:16px;text-align:center'>"
        f"<div style='font-family:var(--mono);font-size:9px;color:var(--muted);letter-spacing:1px'>{lift_label}</div>"
        f"<div style='font-family:var(--display);font-size:32px;font-weight:900;color:{lift_color}'>"
        f"{'+' if lift>=0 else ''}{lift*100:.1f}{'%' if not is_regression else ''}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

# ── Loss + Metric curves ──────────────────────────────────────────────────────
st.markdown("---")
col_l, col_r = st.columns(2)

history = result.get("history", {})
eps = history.get("epochs", [])

with col_l:
    card_title("Loss Curves")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eps, y=history.get("train_loss", []),
                             mode="lines", name="Train Loss",
                             line=dict(color="#00f5ff", width=2)))
    fig.add_trace(go.Scatter(x=eps, y=history.get("test_loss", []),
                             mode="lines", name="Test Loss",
                             line=dict(color="#ff6b2b", width=2),
                             fill="tonexty", fillcolor="rgba(255,107,43,0.06)"))
    fig.update_layout(
        paper_bgcolor="#111827", plot_bgcolor="#0a0f1e",
        height=220, margin=dict(l=48, r=12, t=12, b=36),
        xaxis=dict(title="Epoch", gridcolor="#1f2d48",
                   tickfont=dict(size=9, color="#4a5568", family="Share Tech Mono")),
        yaxis=dict(title="Loss", gridcolor="#1f2d48",
                   tickfont=dict(size=9, color="#4a5568", family="Share Tech Mono")),
        legend=dict(font=dict(size=9, color="#c9d4e8"), bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig, use_container_width=True, key="custom_loss")

with col_r:
    met_label = "Accuracy" if not is_regression else "R² Score"
    card_title(f"{met_label} Curves")
    fig2 = go.Figure()
    train_met = [v * 100 for v in history.get("train_metric", [])] if not is_regression else history.get("train_metric", [])
    test_met  = [v * 100 for v in history.get("test_metric", [])] if not is_regression else history.get("test_metric", [])
    fig2.add_trace(go.Scatter(x=eps, y=train_met, mode="lines", name=f"Train {met_label}",
                              line=dict(color="#00f5ff", width=2)))
    fig2.add_trace(go.Scatter(x=eps, y=test_met, mode="lines", name=f"Test {met_label}",
                              line=dict(color="#ff6b2b", width=2),
                              fill="tonexty", fillcolor="rgba(255,107,43,0.06)"))
    # Baseline reference line
    bl_ref = bl_acc * 100 if not is_regression else bl_acc
    fig2.add_hline(y=bl_ref, line_dash="dash", line_color="rgba(139,92,246,0.5)",
                   annotation_text=f"Baseline: {bl_ref:.1f}",
                   annotation_font_color="#8b5cf6", annotation_font_size=9)
    fig2.update_layout(
        paper_bgcolor="#111827", plot_bgcolor="#0a0f1e",
        height=220, margin=dict(l=48, r=12, t=12, b=36),
        xaxis=dict(title="Epoch", gridcolor="#1f2d48",
                   tickfont=dict(size=9, color="#4a5568", family="Share Tech Mono")),
        yaxis=dict(title=met_label + (" %" if not is_regression else ""),
                   gridcolor="#1f2d48",
                   tickfont=dict(size=9, color="#4a5568", family="Share Tech Mono")),
        legend=dict(font=dict(size=9, color="#c9d4e8"), bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig2, use_container_width=True, key="custom_metric")

# ── Feature Importance ────────────────────────────────────────────────────────
st.markdown("---")
col_fi, col_dl = st.columns(2)

with col_fi:
    card_title("Feature Importance (Weight Magnitude)", color="var(--purple)")
    fi = result.get("feature_importance", {})
    if fi:
        sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)
        names = [x[0][:25] for x in sorted_fi[:15]]
        vals  = [x[1] for x in sorted_fi[:15]]
        colors = [f"rgba(139,92,246,{0.4 + 0.6*v})" for v in vals]

        fig3 = go.Figure(go.Bar(
            x=vals, y=names, orientation="h",
            marker_color=colors,
        ))
        fig3.update_layout(
            paper_bgcolor="#111827", plot_bgcolor="#0a0f1e",
            height=max(180, len(names) * 24 + 40),
            margin=dict(l=8, r=12, t=8, b=8),
            xaxis=dict(title="Relative importance", gridcolor="#1f2d48",
                       tickfont=dict(size=8, color="#4a5568", family="Share Tech Mono")),
            yaxis=dict(tickfont=dict(size=9, color="#c9d4e8", family="Share Tech Mono"),
                       autorange="reversed"),
            showlegend=False,
        )
        st.plotly_chart(fig3, use_container_width=True, key="feat_imp")
    else:
        st.markdown("<div style='color:var(--muted);font-size:11px'>No feature importance data.</div>",
                    unsafe_allow_html=True)

# ── Download Predictions ──────────────────────────────────────────────────────
with col_dl:
    card_title("Download Predictions", color="var(--orange)")

    try:
        pred_resp = requests.get(
            f"{BASE_URL}/custom/predictions/{sm.get_session_id()}", timeout=10
        )
        if pred_resp.status_code == 200:
            pred_data = pred_resp.json()
            predictions = pred_data.get("predictions", [])
            y_test_vals = pred_data.get("y_test", [])
            classes = pred_data.get("classes", [])

            # Build CSV content
            lines = ["index,actual,predicted"]
            for i, (actual, predicted) in enumerate(zip(y_test_vals, predictions)):
                if classes:
                    try:
                        actual_lbl = classes[int(actual)] if int(actual) < len(classes) else actual
                        pred_lbl = classes[int(predicted)] if int(predicted) < len(classes) else predicted
                    except Exception:
                        actual_lbl, pred_lbl = actual, predicted
                else:
                    actual_lbl, pred_lbl = actual, predicted
                lines.append(f"{i},{actual_lbl},{pred_lbl}")
            csv_content = "\n".join(lines)

            st.download_button(
                label="⬇ Download Predictions (CSV)",
                data=csv_content,
                file_name="neural_forge_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )

            # Quick accuracy summary
            if not is_regression and y_test_vals and predictions:
                correct = sum(int(a) == int(p) for a, p in zip(y_test_vals, predictions))
                total = len(y_test_vals)
                st.markdown(
                    f"<div style='background:var(--card2);border:1px solid var(--border);border-radius:6px;"
                    f"padding:10px 14px;margin-top:10px'>"
                    f"<div style='font-family:var(--mono);font-size:10px;color:var(--muted)'>TEST SET SUMMARY</div>"
                    f"<div style='font-family:var(--mono);font-size:14px;color:var(--cyan);margin-top:4px'>"
                    f"{correct}/{total} correct ({correct/total*100:.1f}%)</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            # Class distribution bar chart
            if not is_regression and classes:
                from collections import Counter
                pred_counts = Counter(int(p) for p in predictions)
                fig_dist = go.Figure(go.Bar(
                    x=[classes[i] if i < len(classes) else str(i) for i in sorted(pred_counts)],
                    y=[pred_counts[i] for i in sorted(pred_counts)],
                    marker_color="#8b5cf6",
                ))
                fig_dist.update_layout(
                    paper_bgcolor="#111827", plot_bgcolor="#0a0f1e",
                    height=160, margin=dict(l=8, r=8, t=24, b=36),
                    title=dict(text="Prediction distribution", font=dict(size=10, color="#4a5568")),
                    xaxis=dict(tickfont=dict(size=9, color="#c9d4e8", family="Share Tech Mono"),
                               gridcolor="#1f2d48"),
                    yaxis=dict(gridcolor="#1f2d48",
                               tickfont=dict(size=9, color="#4a5568", family="Share Tech Mono")),
                )
                st.plotly_chart(fig_dist, use_container_width=True, key="pred_dist")
        else:
            st.info("Train a model to download predictions.")
    except Exception:
        st.info("Train a model to download predictions.")

# ── Confusion matrix for classification ──────────────────────────────────────
if not is_regression and result:
    try:
        pred_resp = requests.get(
            f"{BASE_URL}/custom/predictions/{sm.get_session_id()}", timeout=10
        )
        if pred_resp.status_code == 200:
            pred_data = pred_resp.json()
            predictions = pred_data.get("predictions", [])
            y_test_vals = pred_data.get("y_test", [])
            classes = pred_data.get("classes", [])

            if predictions and y_test_vals:
                st.markdown("---")
                card_title("Confusion Matrix", color="var(--purple)")

                n_cls = len(classes) if classes else max(int(max(y_test_vals))+1, int(max(predictions))+1)
                cm = np.zeros((n_cls, n_cls), dtype=int)
                for actual, pred in zip(y_test_vals, predictions):
                    try:
                        cm[int(actual)][int(pred)] += 1
                    except Exception:
                        pass

                cls_labels = [str(c) for c in classes] if classes else [str(i) for i in range(n_cls)]
                fig_cm = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=cls_labels,
                    y=cls_labels,
                    colorscale=[[0, "#111827"], [1, "#8b5cf6"]],
                    showscale=False,
                    text=cm.tolist(),
                    texttemplate="%{text}",
                    textfont=dict(size=14, color="white", family="Share Tech Mono"),
                ))
                fig_cm.update_layout(
                    paper_bgcolor="#111827", plot_bgcolor="#0a0f1e",
                    height=min(400, 100 + n_cls * 60),
                    margin=dict(l=8, r=8, t=8, b=36),
                    xaxis=dict(title="Predicted", gridcolor="#1f2d48",
                               tickfont=dict(size=10, color="#c9d4e8", family="Share Tech Mono")),
                    yaxis=dict(title="Actual", gridcolor="#1f2d48",
                               tickfont=dict(size=10, color="#c9d4e8", family="Share Tech Mono"),
                               autorange="reversed"),
                )
                st.plotly_chart(fig_cm, use_container_width=True, key="custom_cm")
    except Exception:
        pass

# ── Dataset info bar at bottom ────────────────────────────────────────────────
st.markdown("---")
n_train = result.get("n_train", 0)
n_test  = result.get("n_test", 0)
n_feat  = result.get("n_features", 0)
n_cls   = result.get("n_classes", 2)

st.markdown(
    f"<div style='font-family:var(--mono);font-size:10px;color:var(--muted);text-align:center'>"
    f"Trained on <span style='color:var(--cyan)'>{n_train}</span> samples · "
    f"Tested on <span style='color:var(--orange)'>{n_test}</span> samples · "
    f"<span style='color:var(--purple)'>{n_feat}</span> features · "
    f"<span style='color:var(--green)'>{n_cls}</span> classes · "
    f"Best epoch: <span style='color:var(--cyan)'>{fm.get('best_epoch','—')}</span>"
    f"</div>",
    unsafe_allow_html=True,
)
