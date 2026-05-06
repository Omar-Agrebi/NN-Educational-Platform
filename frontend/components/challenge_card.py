import streamlit as st
from utils.api_client import validate_challenge, APIError
from utils import state_manager as sm
from utils.formatters import fmt_pct, level_color


def render_challenge_card(level_config: dict, last_result: dict = None):
    level_id  = level_config.get("level_id", 1)
    challenge = level_config.get("challenge", {})
    xp_reward = level_config.get("xp_reward", 100)
    color     = level_color(level_id)

    target_metric = challenge.get("target_metric", "test_acc")
    target_value  = challenge.get("target_value", 0.85)
    description   = challenge.get("description", "Complete the challenge")

    # Compute current score from last result
    current_score = 0.0
    if last_result:
        history = last_result.get("history", {})
        metrics = last_result.get("final_metrics", {})
        if target_metric == "test_acc":
            accs = history.get("test_acc", [])
            current_score = accs[-1] if accs else 0.0
        elif target_metric == "train_accuracy":
            accs = history.get("train_acc", [])
            current_score = accs[-1] if accs else 0.0
        elif target_metric == "f1_score":
            current_score = metrics.get("f1_score", 0.0)
        elif target_metric == "gen_gap":
            train_accs = history.get("train_acc", [])
            test_accs  = history.get("test_acc", [])
            if train_accs and test_accs:
                gap = abs(train_accs[-1] - test_accs[-1])
                current_score = gap
                # For gap, lower is better — invert progress
                progress_pct = max(0, min(100, int((1 - gap / max(target_value * 2, 0.01)) * 100)))
            else:
                progress_pct = 0
        elif target_metric == "accuracy_gain":
            current_score = metrics.get("accuracy", 0.0)

    is_completed = sm.is_challenge_completed(level_id)

    if target_metric == "gen_gap":
        if last_result:
            h = last_result.get("history", {})
            ta = h.get("train_acc", [])
            va = h.get("test_acc", [])
            gap = abs(ta[-1] - va[-1]) if ta and va else 1.0
            progress_pct = max(0, min(100, int((1 - gap / max(target_value * 4, 0.01)) * 100)))
        else:
            progress_pct = 0
    else:
        progress_pct = min(100, int(current_score / max(target_value, 0.001) * 100))

    border_color = "#00ff88" if is_completed else color

    st.markdown(f"""
<div class="quest-card" style="border-color:{border_color}">
  <div style="font-family:var(--display);font-size:10px;color:{color};
    letter-spacing:2px;margin-bottom:8px">CHALLENGE — LEVEL {level_id:02d}</div>
  <div style="font-size:13px;color:var(--text);margin-bottom:10px;line-height:1.5">
    {description}
  </div>
  <div style="font-family:var(--mono);font-size:11px;color:var(--cyan);
    background:rgba(0,245,255,0.06);border-radius:4px;
    padding:4px 10px;display:inline-block;margin-bottom:12px">
    Target: {_format_target(target_metric, target_value)}
  </div>
  <div>
    <div style="display:flex;justify-content:space-between;
      font-family:var(--mono);font-size:9px;color:var(--muted);margin-bottom:4px">
      <span>Progress</span>
      <span>{_format_score(target_metric, current_score)}</span>
    </div>
    <div style="height:5px;background:var(--border);border-radius:3px;overflow:hidden">
      <div style="height:100%;width:{progress_pct}%;
        background:linear-gradient(90deg,{color},{color}cc);
        border-radius:3px;transition:width 0.5s ease"></div>
    </div>
  </div>
  <div style="margin-top:12px;display:flex;align-items:center;justify-content:space-between">
    <div style="display:flex;align-items:center;gap:6px;
      font-family:var(--mono);font-size:10px;color:var(--orange)">
      <span style="width:6px;height:6px;border-radius:50%;background:var(--orange);display:inline-block"></span>
      +{xp_reward} XP on completion
    </div>
    {"<div style='font-family:var(--mono);font-size:10px;color:var(--green)'>✓ COMPLETE</div>" if is_completed else ""}
  </div>
</div>
""", unsafe_allow_html=True)

    if last_result and not is_completed:
        run_id = last_result.get("run_id", "")
        if run_id and st.button("⚡ Submit Run for Validation", key=f"submit_challenge_{level_id}", type="primary"):
            try:
                result = validate_challenge(sm.get_session_id(), level_id, run_id)
                if result.get("passed"):
                    st.success(result.get("message", "Challenge passed!"))
                    sm.sync_progress_from_api()
                    st.balloons()
                else:
                    st.warning(result.get("message", "Not yet — keep iterating."))
                    score = result.get("score", 0)
                    st.markdown(
                        f"<div style='font-family:var(--mono);font-size:11px;color:var(--muted)'>"
                        f"Partial XP earned: +{result.get('xp_earned',0)}</div>",
                        unsafe_allow_html=True,
                    )
            except APIError as e:
                st.error(f"API error: {e}")


def _format_target(metric: str, value: float) -> str:
    if metric == "gen_gap":
        return f"train/test gap < {value:.0%}"
    if metric in ("test_acc", "train_accuracy"):
        return f"accuracy > {value:.0%}"
    if metric == "f1_score":
        return f"F1 > {value:.2f}"
    if metric == "accuracy_gain":
        return f"accuracy gain > {value:.0%}"
    return f"{value}"


def _format_score(metric: str, score: float) -> str:
    if metric == "gen_gap":
        return f"gap: {score:.1%}"
    if metric in ("test_acc", "train_accuracy", "accuracy_gain"):
        return f"{score:.1%}"
    if metric == "f1_score":
        return f"F1: {score:.3f}"
    return f"{score:.3f}"
