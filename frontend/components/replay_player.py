import streamlit as st
import time
from components.boundary_viz import render_boundary
from components.loss_curve import render_loss_curve


def _build_history_up_to(snapshots: list, epoch_idx: int) -> dict:
    sliced = snapshots[:epoch_idx + 1]
    return {
        "epochs": [s["epoch"] for s in sliced],
        "train_loss": [s["train_loss"] for s in sliced],
        "test_loss": [s["test_loss"] for s in sliced],
        "train_acc": [s["train_acc"] for s in sliced],
        "test_acc": [s["test_acc"] for s in sliced],
    }


def _detect_events(snapshots: list) -> dict:
    """Annotate key events: first overfit, minimum loss epoch."""
    events = {}
    min_test = float("inf")
    min_epoch = 0
    for i, s in enumerate(snapshots):
        gap = s["train_acc"] - s["test_acc"]
        if gap > 0.10 and "overfit_start" not in events:
            events["overfit_start"] = i
        if s["test_loss"] < min_test:
            min_test = s["test_loss"]
            min_epoch = i
    events["best_epoch"] = min_epoch
    return events


def render_replay_player(replay_data: dict):
    if not replay_data:
        st.info("No replay data available.")
        return

    runs = replay_data.get("runs", {})
    if not runs:
        st.info("No runs recorded.")
        return

    run_ids = list(runs.keys())
    selected_run = st.selectbox("Select Run", run_ids,
                                 format_func=lambda r: f"Run {r[:8]}…",
                                 key="replay_run_select")
    if not selected_run:
        return

    run_data = runs[selected_run]
    snapshots = run_data.get("snapshots", [])
    if not snapshots:
        st.info("No snapshots for this run.")
        return

    n = len(snapshots)
    events = _detect_events(snapshots)

    # Speed selector
    speed_col, _, play_col = st.columns([2, 4, 2])
    with speed_col:
        speed = st.select_slider("Speed", [0.5, 1.0, 2.0], value=1.0, key="replay_speed")
    with play_col:
        play = st.button("▶ Play", key="replay_play")

    epoch_idx = st.slider("Epoch", 0, n - 1, value=0, key="replay_epoch_slider")

    # Event annotations
    event_html = ""
    if "overfit_start" in events and events["overfit_start"] == epoch_idx:
        event_html = '<span style="color:#ff3b5c;font-size:0.78rem;">⚠ Overfitting begins here</span>'
    if events.get("best_epoch") == epoch_idx:
        event_html += '<span style="color:#00ff88;font-size:0.78rem;margin-left:8px;">★ Best epoch</span>'
    if event_html:
        st.markdown(event_html, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    boundary_ph = col1.empty()
    loss_ph = col2.empty()

    def _render_epoch(idx):
        snap = snapshots[idx]
        history = _build_history_up_to(snapshots, idx)
        with boundary_ph.container():
            bd = snap.get("boundary_data")
            if bd:
                render_boundary(bd, epoch=snap["epoch"], key=f"replay_bd_{idx}")
            else:
                st.caption("No boundary snapshot at this epoch.")
        with loss_ph.container():
            render_loss_curve(history, key=f"replay_loss_{idx}")
            st.metric("Epoch", snap["epoch"])
            c1, c2 = st.columns(2)
            c1.metric("Train Loss", f"{snap['train_loss']:.4f}")
            c2.metric("Test Loss", f"{snap['test_loss']:.4f}")

    _render_epoch(epoch_idx)

    if play:
        delay = 1.0 / speed
        for i in range(epoch_idx, n):
            st.session_state["replay_epoch_slider"] = i
            _render_epoch(i)
            time.sleep(delay)
