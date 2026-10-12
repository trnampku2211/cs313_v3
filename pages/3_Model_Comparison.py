import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.ui import benchmark_required, cached_benchmark_results, configure_page, protocol_badge


configure_page("Model Comparison", "🏁")
st.title("Model Comparison")
st.caption("Official held-out test benchmark")
protocol_badge()

if not benchmark_required():
    st.stop()

payload = cached_benchmark_results()
rows = []
for result in payload["results"]:
    metrics = result["test_metrics"]
    rows.append(
        {
            "Model": result["model"],
            "Learning type": result["learning_type"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1": metrics["f1"],
            "Average Precision": metrics["average_precision"],
            "False positives": metrics["false_positive"],
            "False negatives": metrics["false_negative"],
            "Fit time (s)": result["fit_time_seconds"],
            "Inference time (s)": result["inference_time_seconds"],
            "Score source": result["score_source"],
        }
    )
comparison = pd.DataFrame(rows)
st.dataframe(
    comparison.style.format({"Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}", "Average Precision": lambda x: "N/A" if pd.isna(x) else f"{x:.3f}"}),
    hide_index=True,
    use_container_width=True,
)

metric = st.selectbox("Comparison metric", ["F1", "Recall", "Precision", "Average Precision"])
chart_data = comparison.dropna(subset=[metric]).sort_values(metric, ascending=False)
fig = px.bar(chart_data, x="Model", y=metric, color="Learning type", text_auto=".3f", range_y=[0, 1])
st.plotly_chart(fig, use_container_width=True)

st.subheader("Model details")
selected = st.selectbox("Model", comparison["Model"].tolist())
result = next(item for item in payload["results"] if item["model"] == selected)
left, right = st.columns([1, 1.2])
with left:
    metrics = result["test_metrics"]
    matrix = [[metrics["true_negative"], metrics["false_positive"]], [metrics["false_negative"], metrics["true_positive"]]]
    fig = go.Figure(data=go.Heatmap(z=matrix, x=["Predicted normal", "Predicted anomaly"],
                                    y=["Actual normal", "Actual failure"], text=matrix, texttemplate="%{text}", colorscale="Blues"))
    fig.update_layout(title="Confusion matrix", yaxis_autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)
with right:
    st.markdown(f"**Learning type:** {result['learning_type']}")
    st.markdown(f"**Score source:** {result['score_source']}")
    st.markdown(f"**Selected threshold:** {metrics['threshold']}")
    st.markdown("**Validation-selected parameters**")
    st.json(result["parameters"])

curve = result.get("test_pr_curve")
if curve:
    curve_frame = pd.DataFrame({"Recall": curve["recall"], "Precision": curve["precision"]})
    fig = px.line(curve_frame, x="Recall", y="Precision", title=f"Precision–Recall curve — {selected}")
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("A continuous anomaly score is not available for this model; its PR curve is N/A.")

st.info(
    "Random Forest is supervised and learned failure labels during fitting. DBSCAN, Isolation Forest, "
    "and LOF were fitted without labels, but their configurations were selected using validation labels."
)

