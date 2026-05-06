import streamlit as st
import plotly.graph_objects as go
import numpy as np
from styles.theme import card_title


def render_confusion_matrix(cm: list, labels=("Class 0", "Class 1")):
    card_title("Confusion Matrix", color="var(--purple)")

    if not cm or len(cm) != 2:
        st.markdown("<div style='color:var(--muted);font-size:11px;font-family:var(--mono)'>No data yet.</div>", unsafe_allow_html=True)
        return

    Z = np.array(cm)
    total = Z.sum() or 1

    text = [
        [f"{Z[0,0]}<br><span style='font-size:9px'>TN</span>", f"{Z[0,1]}<br><span style='font-size:9px'>FP</span>"],
        [f"{Z[1,0]}<br><span style='font-size:9px'>FN</span>", f"{Z[1,1]}<br><span style='font-size:9px'>TP</span>"],
    ]

    fig = go.Figure(data=go.Heatmap(
        z=Z,
        x=list(labels),
        y=list(labels),
        colorscale=[
            [0.0, "#111827"],
            [1.0, "#8b5cf6"],
        ],
        showscale=False,
        text=[[str(v) for v in row] for row in Z.tolist()],
        texttemplate="%{text}",
        textfont=dict(size=18, color="white", family="Share Tech Mono"),
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
    ))

    # Overlay TN/FP/FN/TP labels
    annotations = []
    names = [["TN", "FP"], ["FN", "TP"]]
    colors_map = {"TP": "#00ff88", "TN": "#00f5ff", "FP": "#ff3b5c", "FN": "#ff6b2b"}
    for i in range(2):
        for j in range(2):
            annotations.append(dict(
                x=j, y=i,
                text=f"<b>{Z[i,j]}</b><br><span style='font-size:8px'>{names[i][j]}</span>",
                showarrow=False,
                font=dict(color=colors_map[names[i][j]], size=14, family="Share Tech Mono"),
            ))

    fig.update_layout(
        paper_bgcolor="#111827",
        plot_bgcolor="#0a0f1e",
        height=200,
        margin=dict(l=48, r=12, t=12, b=36),
        xaxis=dict(
            title="Predicted",
            title_font=dict(size=9, color="#4a5568"),
            tickfont=dict(size=9, color="#c9d4e8", family="Share Tech Mono"),
        ),
        yaxis=dict(
            title="Actual",
            title_font=dict(size=9, color="#4a5568"),
            tickfont=dict(size=9, color="#c9d4e8", family="Share Tech Mono"),
            autorange="reversed",
        ),
        annotations=annotations,
    )
    st.plotly_chart(fig, use_container_width=True, key="conf_matrix")

    # Precision-recall curve hint
    tn, fp = Z[0, 0], Z[0, 1]
    fn, tp = Z[1, 0], Z[1, 1]
    prec   = tp / (tp + fp + 1e-8)
    rec    = tp / (tp + fn + 1e-8)
    f1     = 2 * prec * rec / (prec + rec + 1e-8)

    st.markdown(f"""
<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-top:8px">
  <div style="background:var(--card2);border:1px solid var(--border);border-radius:6px;padding:8px;text-align:center">
    <div style="font-family:var(--mono);font-size:9px;color:var(--muted)">PRECISION</div>
    <div style="font-family:var(--mono);font-size:16px;color:var(--cyan)">{prec:.3f}</div>
  </div>
  <div style="background:var(--card2);border:1px solid var(--border);border-radius:6px;padding:8px;text-align:center">
    <div style="font-family:var(--mono);font-size:9px;color:var(--muted)">RECALL</div>
    <div style="font-family:var(--mono);font-size:16px;color:var(--orange)">{rec:.3f}</div>
  </div>
  <div style="background:var(--card2);border:1px solid var(--border);border-radius:6px;padding:8px;text-align:center">
    <div style="font-family:var(--mono);font-size:9px;color:var(--muted)">F1 SCORE</div>
    <div style="font-family:var(--mono);font-size:16px;color:var(--green)">{f1:.3f}</div>
  </div>
</div>
""", unsafe_allow_html=True)
