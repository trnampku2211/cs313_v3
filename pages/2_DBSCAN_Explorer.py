from time import perf_counter

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.data import FEATURE_COLUMNS, NUMERIC_FEATURES, TARGET_COLUMN
from src.models.dbscan_model import fit_dbscan, k_distances
from src.preprocessing import fit_preprocessor
from src.splitting import create_stratified_split
from src.ui import cached_benchmark_results, cached_dataset, configure_page, protocol_badge


configure_page("DBSCAN Explorer", "🧭")
st.title("DBSCAN Explorer")
st.caption("Primary algorithm • interactive density-based anomaly demonstration")
protocol_badge()
st.warning("Interactive exploration — results on this page do not modify the official benchmark.")

frame = cached_dataset()
split = create_stratified_split(frame[TARGET_COLUMN].astype(int), random_state=42)
train_frame = frame.iloc[split.train].copy()

official = cached_benchmark_results()
if official:
    db_result = next(item for item in official["results"] if item["model"] == "DBSCAN")
    with st.expander("Official DBSCAN configuration", expanded=False):
        st.json(db_result["parameters"])
        sensitivity = official.get("sensitivity_results", [])
        if sensitivity:
            st.markdown("**Numeric-only sensitivity experiment**")
            st.json(
                {
                    "parameters": sensitivity[0]["parameters"],
                    "test_metrics": sensitivity[0]["test_metrics"],
                }
            )

with st.form("dbscan_controls"):
    mode = st.radio("Demonstration mode", ["Full processed feature space", "Two-feature teaching mode"], horizontal=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        eps = st.number_input("eps", min_value=0.01, max_value=10.0, value=0.75, step=0.05)
    with c2:
        min_samples = st.number_input("min_samples", min_value=2, max_value=200, value=15, step=1)
    with c3:
        color_mode = st.selectbox("Color points by", ["DBSCAN role", "Cluster label", "Ground truth"])
    selected_features = NUMERIC_FEATURES[:2]
    if mode == "Two-feature teaching mode":
        selected_features = st.multiselect(
            "Select exactly two numerical features", NUMERIC_FEATURES, default=NUMERIC_FEATURES[:2], max_selections=2
        )
    run = st.form_submit_button("Run interactive DBSCAN", type="primary", use_container_width=True)

if run:
    if mode == "Two-feature teaching mode" and len(selected_features) != 2:
        st.error("Select exactly two numerical features.")
        st.stop()
    if mode == "Full processed feature space":
        preprocessor = fit_preprocessor(train_frame[FEATURE_COLUMNS])
        values = preprocessor.transform(train_frame[FEATURE_COLUMNS])
        display_values = PCA(n_components=2, random_state=42).fit_transform(values)
        x_label, y_label = "PCA component 1", "PCA component 2"
        subtitle = "DBSCAN fitted in the full processed feature space; PCA is visualization only."
    else:
        values = StandardScaler().fit_transform(train_frame[selected_features])
        display_values = values
        x_label, y_label = selected_features
        subtitle = "DBSCAN fitted directly on the two standardized features shown."

    started = perf_counter()
    fitted = fit_dbscan(values, eps=eps, min_samples=int(min_samples))
    elapsed = perf_counter() - started
    labels = fitted.model.labels_.astype(str)
    plot_data = pd.DataFrame({x_label: display_values[:, 0], y_label: display_values[:, 1]})
    plot_data["Role"] = fitted.point_roles
    plot_data["Cluster"] = np.where(fitted.model.labels_ == -1, "Noise", labels)
    plot_data["Ground truth"] = train_frame[TARGET_COLUMN].map({0: "Normal", 1: "Failure"}).to_numpy()
    plot_data["UDI"] = train_frame["UDI"].to_numpy()
    for column in NUMERIC_FEATURES:
        plot_data[column] = train_frame[column].to_numpy()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clusters", fitted.cluster_count)
    c2.metric("Core points", int(np.sum(fitted.point_roles == "core")))
    c3.metric("Border points", int(np.sum(fitted.point_roles == "border")))
    c4.metric("Noise points", f"{fitted.noise_count:,} ({fitted.noise_count / len(values):.2%})")
    st.caption(f"Interactive fit time: {elapsed:.3f} seconds. {subtitle}")

    color_column = {"DBSCAN role": "Role", "Cluster label": "Cluster", "Ground truth": "Ground truth"}[color_mode]
    symbol_column = "Role" if color_mode != "DBSCAN role" else None
    fig = px.scatter(
        plot_data,
        x=x_label,
        y=y_label,
        color=color_column,
        symbol=symbol_column,
        hover_data=["UDI"] + NUMERIC_FEATURES,
        opacity=0.72,
        render_mode="webgl",
        title=color_mode,
    )
    fig.update_traces(marker={"size": 6})
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("K-distance diagnostic", expanded=True):
        distances = k_distances(values, int(min_samples))
        distance_frame = pd.DataFrame({"Sorted training point": np.arange(len(distances)), "k-distance": distances})
        fig = px.line(distance_frame, x="Sorted training point", y="k-distance")
        fig.add_hline(y=eps, line_dash="dash", line_color="#ef4444", annotation_text="Selected eps")
        st.plotly_chart(fig, use_container_width=True)

    cluster_table = (
        plot_data.groupby("Cluster", observed=True)
        .agg(Records=("UDI", "count"), Failures=("Ground truth", lambda s: int((s == "Failure").sum())))
        .reset_index()
    )
    cluster_table["Failure rate"] = cluster_table["Failures"] / cluster_table["Records"]
    st.subheader("Cluster statistics")
    st.dataframe(cluster_table.sort_values("Records", ascending=False), hide_index=True, use_container_width=True)

    if fitted.cluster_count == 0:
        st.error("Degenerate result: DBSCAN found no clusters. Increase eps or reduce min_samples.")
    elif fitted.noise_count == 0:
        st.info("DBSCAN found no noise points. Consider a smaller eps for anomaly exploration.")
else:
    st.info("Choose parameters and run DBSCAN. The fixed official benchmark remains unchanged.")

