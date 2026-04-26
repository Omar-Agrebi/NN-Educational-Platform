import streamlit as st
import time
from components.boundary_viz import render_boundary
from components.loss_curve import render_loss_curve
from styles.theme import card_title


def render_replay_player(replay_data: dict):
    card_title("Training Replay", color="var(--purple)")

    if not replay_data:
        st.markdown(
            "<div style='color:var(--muted);font-family:var(--mono);font-size:11px'>"
            "No replay data. Complete a training run first.</div>",
            unsafe_allow_html=True,
        )
        return

    # Flatten all runs into a list with labels
    all_runs = []
    for run_id, snaps in replay_data.items():
        if snaps:
            all_runs.append((run_id, snaps))

    if not all_runs:
        st.markdown("<div style='color:var(--muted);font-family:var(--mono);font-size:11px'>No snapshots available.</div>", unsafe_allow_html=True)
        return

    run_labels = [f"Run {r[:6]}… ({len(s)} snapshots)" for r, s in all_runs]
    selected_run_label = st.selectbox("Select run", options=run_labels, key="replay_run_select")
    run_idx = run_labels.index(selected_run_label)
    _, snapshots = all_runs[run_idx]

    n = len(snapshots)
    if n == 0:
        return

    # Epoch scrubber
    epoch_idx = st.slider(
        "Epoch scrubber",
        min_value=0,
        max_value=n - 1,
        value=0,
        key="replay_epoch_slider",
    )

    snap = snapshots[epoch_idx]
    ep   = snap.get("epoch", epoch_idx + 1)

    # Timeline with markers
    st.markdown(_build_timeline(snapshots, epoch_idx, n), unsafe_allow_html=True)

    # Play controls
    col_a, col_b, col_c, col_d, col_e = st.columns(5)
    with col_a:
        if st.button("⏮ Start", key="rep_start"):
            st.session_state["replay_epoch_slider"] = 0
            st.rerun()
    with col_b:
        play = st.button("▶ Play", key="rep_play")
    with col_c:
        if st.button("⏭ End", key="rep_end"):
            st.session_state["replay_epoch_slider"] = n - 1
            st.rerun()
    with col_d:
        speed = st.select_slider("Speed", options=["0.5×", "1×", "2×"], value="1×", key="rep_speed")
    with col_e:
        st.markdown(
            f"<div style='font-family:var(--mono);font-size:10px;color:var(--muted);padding-top:28px'>"
            f"Ep {ep} / {snapshots[-1].get('epoch', n)}</div>",
            unsafe_allow_html=True,
        )

    # Current snapshot metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Train Loss", f"{snap.get('train_loss', 0):.4f}")
    with c2:
        st.metric("Test Loss",  f"{snap.get('test_loss', 0):.4f}")
    with c3:
        st.metric("Train Acc",  f"{snap.get('train_acc', 0):.1%}")
    with c4:
        st.metric("Test Acc",   f"{snap.get('test_acc', 0):.1%}")

    # Build partial history up to this snapshot
    partial_history = {
        "epochs":     [s.get("epoch", i+1) for i, s in enumerate(snapshots[:epoch_idx+1])],
        "train_loss": [s.get("train_loss", 0) for s in snapshots[:epoch_idx+1]],
        "test_loss":  [s.get("test_loss", 0) for s in snapshots[:epoch_idx+1]],
        "train_acc":  [s.get("train_acc", 0) for s in snapshots[:epoch_idx+1]],
        "test_acc":   [s.get("test_acc", 0) for s in snapshots[:epoch_idx+1]],
    }
    render_loss_curve(partial_history, title="Loss at this epoch", show_accuracy=False,
                      height=160, key=f"replay_loss_{epoch_idx}")

    # Auto-play
    if play:
        delay = {"0.5×": 0.3, "1×": 0.15, "2×": 0.06}.get(speed, 0.15)
        for i in range(epoch_idx, n):
            st.session_state["replay_epoch_slider"] = i
            time.sleep(delay)
            st.rerun()


def _build_timeline(snapshots, current_idx, n):
    pct = int(current_idx / max(n - 1, 1) * 100)

    # Detect key events from loss trajectory
    markers_html = ""
    train_losses = [s.get("train_loss", 0) for s in snapshots]
    test_losses  = [s.get("test_loss", 0) for s in snapshots]

    # Best test loss
    if test_losses:
        best_idx = test_losses.index(min(test_losses))
        bpct = int(best_idx / max(n - 1, 1) * 100)
        markers_html += (
            f"<div style='position:absolute;left:{bpct}%;transform:translateX(-50%);bottom:-16px;"
            f"font-family:var(--mono);font-size:8px;color:var(--green)'>✓best</div>"
            f"<div style='position:absolute;left:{bpct}%;top:-5px;transform:translateX(-50%);"
            f"width:1px;height:14px;background:var(--green)'></div>"
        )

    # Overfit start: when test starts increasing while train still decreasing
    for i in range(1, len(test_losses)):
        if test_losses[i] > test_losses[i-1] and (i > 2) and train_losses[i] < train_losses[i-1]:
            opct = int(i / max(n - 1, 1) * 100)
            markers_html += (
                f"<div style='position:absolute;left:{opct}%;transform:translateX(-50%);bottom:-16px;"
                f"font-family:var(--mono);font-size:8px;color:var(--red)'>⚠ overfit</div>"
                f"<div style='position:absolute;left:{opct}%;top:-5px;transform:translateX(-50%);"
                f"width:1px;height:14px;background:var(--red)'></div>"
            )
            break

    return f"""
<div style="position:relative;margin:24px 0 32px">
  <div style="height:4px;background:var(--border);border-radius:2px;position:relative">
    <div style="height:100%;width:{pct}%;background:linear-gradient(90deg,var(--cyan),var(--purple));border-radius:2px"></div>
    <div style="position:absolute;top:50%;left:{pct}%;transform:translate(-50%,-50%);
      width:12px;height:12px;border-radius:50%;background:var(--cyan);
      border:2px solid var(--navy);box-shadow:0 0 8px rgba(0,245,255,0.5)"></div>
    {markers_html}
  </div>
</div>
"""
