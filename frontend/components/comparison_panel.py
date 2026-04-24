import streamlit as st
import plotly.graph_objects as go
from components.loss_curve import render_loss_curve
from components.boundary_viz import render_boundary
from utils.formatters import fmt_pct, fmt_loss, level_color


def _metric_row(label, val_a, val_b, higher_is_better=True):
    a_wins = (val_a > val_b) if higher_is_better else (val_a < val_b)
    color_a = "#00ff88" if a_wins else "#ff3b5c"
    color_b = "#00ff88" if not a_wins else "#ff3b5c"
    if abs(val_a - val_b) < 0.005:
        color_a = color_b = "#ff6b2b"
    return (
        f"<tr>"
        f"<td style='color:{color_a};font-family:monospace;text-align:right;padding:3px 12px;'>{val_a:.4f}</td>"
        f"<td style='color:#4a5568;text-align:center;padding:3px 8px;font-size:0.8rem;'>{label}</td>"
        f"<td style='color:{color_b};font-family:monospace;text-align:left;padding:3px 12px;'>{val_b:.4f}</td>"
        f"</tr>"
    )


def render_comparison_panel(result_a: dict, result_b: dict,
                             label_a: str = "Model A", label_b: str = "Model B",
                             is_duel: bool = False):
    if not result_a or not result_b:
        st.info("Run both models to see the comparison.")
        return

    metrics_a = result_a.get("final_metrics", {})
    metrics_b = result_b.get("final_metrics", {})

    acc_a = metrics_a.get("test_accuracy", 0)
    acc_b = metrics_b.get("test_accuracy", 0)
    f1_a = metrics_a.get("f1_score", 0)
    f1_b = metrics_b.get("f1_score", 0)

    a_wins_acc = acc_a > acc_b + 0.005
    b_wins_acc = acc_b > acc_a + 0.005
    a_wins_f1  = f1_a > f1_b + 0.005
    b_wins_f1  = f1_b > f1_a + 0.005
    a_wins_both = a_wins_acc and a_wins_f1
    b_wins_both = b_wins_acc and b_wins_f1

    col_a, col_vs, col_b = st.columns([5, 1, 5])

    with col_a:
        badge = "🧑 YOU" if is_duel else "A"
        color_a = "#00f5ff"
        st.markdown(f"""
        <div style="text-align:center;margin-bottom:10px;">
          <span class="level-badge" style="color:{color_a};border-color:{color_a};">{badge}</span>
          <div style="font-weight:700;color:{color_a};margin-top:6px;">{label_a}</div>
        </div>
        """, unsafe_allow_html=True)

        if is_duel:
            hp = int(acc_a * 100)
            st.markdown(f"""
            <div class="forge-card" style="padding:10px;">
              <div class="section-title">ACCURACY</div>
              <div class="hp-bar-track">
                <div class="hp-bar-fill" style="width:{hp}%;background:#00f5ff;--hp:{hp}%;"></div>
              </div>
              <div style="font-family:monospace;color:#00f5ff;text-align:right;">{fmt_pct(acc_a)}</div>
              <div class="section-title" style="margin-top:6px;">F1 SCORE</div>
              <div class="hp-bar-track">
                <div class="hp-bar-fill" style="width:{int(f1_a*100)}%;background:#8b5cf6;"></div>
              </div>
              <div style="font-family:monospace;color:#8b5cf6;text-align:right;">{fmt_pct(f1_a)}</div>
            </div>
            """, unsafe_allow_html=True)

        render_loss_curve(result_a.get("history", {}), key="loss_a", show_accuracy=False)
        render_boundary(result_a.get("boundary_data"), title=f"{label_a} Boundary", key="boundary_a")

        mcols = st.columns(2)
        with mcols[0]:
            st.metric("Test Accuracy", fmt_pct(acc_a))
            st.metric("Precision", fmt_pct(metrics_a.get("precision", 0)))
        with mcols[1]:
            st.metric("F1 Score", fmt_pct(f1_a))
            st.metric("Recall", fmt_pct(metrics_a.get("recall", 0)))

    with col_vs:
        if is_duel:
            st.markdown('<div class="vs-text" style="margin-top:80px;">VS</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="text-align:center;color:#4a5568;margin-top:80px;font-size:1.5rem;">vs</div>', unsafe_allow_html=True)

    with col_b:
        badge = "🤖 OPPONENT" if is_duel else "B"
        color_b = "#ff6b2b"
        st.markdown(f"""
        <div style="text-align:center;margin-bottom:10px;">
          <span class="level-badge" style="color:{color_b};border-color:{color_b};">{badge}</span>
          <div style="font-weight:700;color:{color_b};margin-top:6px;">{label_b}</div>
        </div>
        """, unsafe_allow_html=True)

        if is_duel:
            hp = int(acc_b * 100)
            st.markdown(f"""
            <div class="forge-card" style="padding:10px;">
              <div class="section-title">ACCURACY</div>
              <div class="hp-bar-track">
                <div class="hp-bar-fill" style="width:{hp}%;background:#ff6b2b;--hp:{hp}%;"></div>
              </div>
              <div style="font-family:monospace;color:#ff6b2b;text-align:right;">{fmt_pct(acc_b)}</div>
              <div class="section-title" style="margin-top:6px;">F1 SCORE</div>
              <div class="hp-bar-track">
                <div class="hp-bar-fill" style="width:{int(f1_b*100)}%;background:#ff3b5c;"></div>
              </div>
              <div style="font-family:monospace;color:#ff3b5c;text-align:right;">{fmt_pct(f1_b)}</div>
            </div>
            """, unsafe_allow_html=True)

        render_loss_curve(result_b.get("history", {}), key="loss_b", show_accuracy=False)
        render_boundary(result_b.get("boundary_data"), title=f"{label_b} Boundary", key="boundary_b")

        mcols = st.columns(2)
        with mcols[0]:
            st.metric("Test Accuracy", fmt_pct(acc_b))
            st.metric("Precision", fmt_pct(metrics_b.get("precision", 0)))
        with mcols[1]:
            st.metric("F1 Score", fmt_pct(f1_b))
            st.metric("Recall", fmt_pct(metrics_b.get("recall", 0)))

    # Winner declaration
    st.markdown("---")
    if is_duel:
        if a_wins_both:
            st.markdown("""
            <div style="text-align:center;padding:20px;background:rgba(0,255,136,0.08);
                 border:1px solid #00ff88;border-radius:12px;animation:level-up-flash 0.6s ease-out;">
              <div style="font-size:2rem;font-weight:900;color:#00ff88;letter-spacing:0.1em;">
                ⚔️ VICTORY
              </div>
              <div style="color:#e2e8f0;margin-top:8px;">You outperformed the opponent on both Accuracy and F1.</div>
            </div>
            """, unsafe_allow_html=True)
        elif b_wins_both:
            st.markdown("""
            <div style="text-align:center;padding:20px;background:rgba(255,59,92,0.08);
                 border:1px solid #ff3b5c;border-radius:12px;">
              <div style="font-size:2rem;font-weight:900;color:#ff3b5c;letter-spacing:0.1em;">
                ☠ DEFEATED
              </div>
              <div style="color:#e2e8f0;margin-top:8px;">The opponent won. Adjust your config and try again.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align:center;padding:20px;background:rgba(255,107,43,0.08);
                 border:1px solid #ff6b2b;border-radius:12px;">
              <div style="font-size:1.5rem;font-weight:900;color:#ff6b2b;">⚖ SPLIT DECISION</div>
              <div style="color:#e2e8f0;margin-top:8px;">You need to win on BOTH accuracy AND F1 to claim victory.</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        rows = _metric_row("Test Accuracy", acc_a, acc_b) + _metric_row("F1 Score", f1_a, f1_b)
        st.markdown(f"""
        <div style="text-align:center;">
          <table style="margin:auto;border-collapse:collapse;">
            <thead>
              <tr>
                <th style="color:#00f5ff;padding:4px 12px;">{label_a}</th>
                <th style="color:#4a5568;padding:4px 8px;">Metric</th>
                <th style="color:#ff6b2b;padding:4px 12px;">{label_b}</th>
              </tr>
            </thead>
            <tbody>{rows}</tbody>
          </table>
        </div>
        """, unsafe_allow_html=True)
