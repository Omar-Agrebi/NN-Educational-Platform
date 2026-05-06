import streamlit as st
from components.loss_curve import render_loss_curve
from components.boundary_viz import render_boundary
from styles.theme import card_title


def render_comparison_panel(result_a: dict, result_b: dict, label_a: str = "Model A",
                             label_b: str = "Model B", archetype_a: str = "", archetype_b: str = ""):
    if not result_a or not result_b:
        st.info("Run both models to see comparison.")
        return

    metrics_a = result_a.get("final_metrics", {})
    metrics_b = result_b.get("final_metrics", {})
    hist_a    = result_a.get("history", {})
    hist_b    = result_b.get("history", {})

    acc_a  = metrics_a.get("accuracy", 0)
    acc_b  = metrics_b.get("accuracy", 0)
    f1_a   = metrics_a.get("f1_score", 0)
    f1_b   = metrics_b.get("f1_score", 0)
    prec_a = metrics_a.get("precision", 0)
    prec_b = metrics_b.get("precision", 0)
    rec_a  = metrics_a.get("recall", 0)
    rec_b  = metrics_b.get("recall", 0)

    wins_a = sum([acc_a > acc_b, f1_a > f1_b, prec_a > prec_b, rec_a > rec_b])
    wins_b = 4 - wins_a
    winner = "a" if wins_a > wins_b else "b"

    # Winner banner
    if winner == "a":
        st.markdown(
            f"<div class='success-banner'><span style='color:var(--green);font-family:var(--display);font-size:14px;letter-spacing:2px'>"
            f"🏆 {label_a} WINS</span> — {wins_a}/4 metrics</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='success-banner'><span style='color:var(--cyan);font-family:var(--display);font-size:14px;letter-spacing:2px'>"
            f"🏆 {label_b} WINS</span> — {wins_b}/4 metrics</div>",
            unsafe_allow_html=True,
        )

    # HP bars
    col_l, col_vs, col_r = st.columns([5, 1, 5])
    with col_l:
        st.markdown(
            f"<div style='font-family:var(--display);font-size:11px;font-weight:700;"
            f"color:var(--cyan);letter-spacing:2px;margin-bottom:8px'>{label_a}"
            f"{'<br><span style=\"font-size:9px;color:var(--muted)\">' + archetype_a + '</span>' if archetype_a else ''}</div>",
            unsafe_allow_html=True,
        )
        _hp_bar(acc_a, "Accuracy", "#00f5ff")
        _hp_bar(f1_a,  "F1 Score", "#8b5cf6")

    with col_vs:
        st.markdown(
            "<div style='font-family:var(--display);font-size:20px;font-weight:900;"
            "color:var(--orange);text-align:center;padding-top:20px;"
            "animation:vs-pulse 1.5s infinite'>VS</div>",
            unsafe_allow_html=True,
        )

    with col_r:
        st.markdown(
            f"<div style='font-family:var(--display);font-size:11px;font-weight:700;"
            f"color:var(--red);letter-spacing:2px;margin-bottom:8px;text-align:right'>{label_b}"
            f"{'<br><span style=\"font-size:9px;color:var(--muted)\">' + archetype_b + '</span>' if archetype_b else ''}</div>",
            unsafe_allow_html=True,
        )
        _hp_bar(acc_b, "Accuracy", "#ff3b5c")
        _hp_bar(f1_b,  "F1 Score", "#ff6b2b")

    st.markdown("<hr style='border-color:var(--border);margin:16px 0'>", unsafe_allow_html=True)

    # Metric table
    card_title("Metric Comparison")
    metrics_rows = [
        ("Accuracy",  acc_a,  acc_b),
        ("F1 Score",  f1_a,   f1_b),
        ("Precision", prec_a, prec_b),
        ("Recall",    rec_a,  rec_b),
    ]
    for metric, va, vb in metrics_rows:
        better_a = va > vb
        color_a  = "var(--green)" if better_a else "var(--muted)"
        color_b  = "var(--green)" if not better_a else "var(--muted)"
        st.markdown(f"""
<div style="display:grid;grid-template-columns:1fr 2fr 1fr;
  align-items:center;padding:6px 0;border-bottom:1px solid var(--border)">
  <div style="font-family:var(--mono);font-size:13px;color:{color_a};font-weight:700">
    {'▶ ' if better_a else ''}{va:.3f}
  </div>
  <div style="font-family:var(--mono);font-size:9px;color:var(--muted);
    text-align:center;letter-spacing:1px">{metric.upper()}</div>
  <div style="font-family:var(--mono);font-size:13px;color:{color_b};
    font-weight:700;text-align:right">
    {vb:.3f}{'  ◀' if not better_a else ''}
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:var(--border);margin:16px 0'>", unsafe_allow_html=True)

    # Loss curves side by side
    col1, col2 = st.columns(2)
    with col1:
        render_loss_curve(hist_a, title=f"{label_a} — Loss", show_accuracy=False, height=180, key="comp_loss_a")
    with col2:
        render_loss_curve(hist_b, title=f"{label_b} — Loss", show_accuracy=False, height=180, key="comp_loss_b")

    # Boundaries side by side
    bd_a = result_a.get("boundary_data")
    bd_b = result_b.get("boundary_data")
    if bd_a or bd_b:
        col1, col2 = st.columns(2)
        with col1:
            render_boundary(bd_a, title=f"{label_a} — Boundary", height=260, key="comp_bd_a")
        with col2:
            render_boundary(bd_b, title=f"{label_b} — Boundary", height=260, key="comp_bd_b")


def _hp_bar(val: float, label: str, color: str):
    pct = min(100, int(val * 100))
    st.markdown(f"""
<div class="hp-bar-wrap">
  <div style="display:flex;justify-content:space-between;
    font-family:var(--mono);font-size:9px;color:var(--muted);margin-bottom:3px">
    <span>{label}</span><span>{val:.3f}</span>
  </div>
  <div class="hp-track">
    <div style="height:100%;width:{pct}%;background:{color};border-radius:5px;transition:width 0.5s"></div>
  </div>
</div>
""", unsafe_allow_html=True)
