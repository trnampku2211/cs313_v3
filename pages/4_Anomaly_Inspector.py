import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.data import FAILURE_MODE_COLUMNS, FEATURE_COLUMNS, NUMERIC_FEATURES, TARGET_COLUMN
from src.models.isolation_forest_model import anomaly_scores as if_scores
from src.models.lof_model import anomaly_scores as lof_scores
from src.models.random_forest_model import failure_probabilities
from src.ui import benchmark_required, cached_benchmark_bundle, cached_dataset, configure_page, protocol_badge


configure_page("Anomaly Inspector", "🔎")
st.title("Anomaly Inspector")
st.caption("Record-level comparison on the fixed official test set")
protocol_badge()

if not benchmark_required():
    st.stop()

frame = cached_dataset()
bundle = cached_benchmark_bundle()
split = bundle["split"]
test_frame = frame.iloc[split.test].copy()
preprocessor = bundle["preprocessor"]
models = bundle["models"]

udi = st.selectbox("Test record (UDI)", test_frame["UDI"].tolist())
row = test_frame.loc[test_frame["UDI"] == udi].iloc[0]
x = preprocessor.transform(pd.DataFrame([row[FEATURE_COLUMNS]]))
actual = int(row[TARGET_COLUMN])

db_bundle = models["DBSCAN"]
db_score = float(db_bundle["fit"].predictor.anomaly_score(x)[0])
db_prediction = int(db_bundle["fit"].predictor.predict_anomalies(x)[0])
if_bundle = models["Isolation Forest"]
if_score = float(if_scores(if_bundle["model"], x)[0])
if_prediction = int(if_score >= if_bundle["threshold"])
lof_bundle = models["Local Outlier Factor"]
lof_score = float(lof_scores(lof_bundle["model"], x)[0])
lof_prediction = int(lof_score >= lof_bundle["threshold"])
rf_bundle = models["Random Forest"]
rf_score = float(failure_probabilities(rf_bundle["model"], x)[0])
rf_prediction = int(rf_score >= rf_bundle["threshold"])

c1, c2, c3, c4 = st.columns(4)
c1.metric("UDI", int(udi))
c2.metric("Product type", row["Type"])
c3.metric("Ground truth", "Failure" if actual else "Normal")
c4.metric("Model agreement", f"{sum([db_prediction, if_prediction, lof_prediction, rf_prediction])}/4 anomaly")

prediction_table = pd.DataFrame(
    [
        ["DBSCAN", db_prediction, db_score, 1.0],
        ["Isolation Forest", if_prediction, if_score, if_bundle["threshold"]],
        ["Local Outlier Factor", lof_prediction, lof_score, lof_bundle["threshold"]],
        ["Random Forest", rf_prediction, rf_score, rf_bundle["threshold"]],
    ],
    columns=["Model", "Prediction", "Score", "Threshold"],
)
prediction_table["Prediction"] = prediction_table["Prediction"].map({0: "Normal", 1: "Anomaly"})
prediction_table["Correct"] = prediction_table["Prediction"].eq("Anomaly" if actual else "Normal")
st.subheader("Model decisions")
st.dataframe(prediction_table, hide_index=True, use_container_width=True)

train_numeric = frame.iloc[split.train][NUMERIC_FEATURES]
z_values = (row[NUMERIC_FEATURES].astype(float) - train_numeric.mean()) / train_numeric.std(ddof=0)
sensor_table = pd.DataFrame(
    {"Feature": NUMERIC_FEATURES, "Value": row[NUMERIC_FEATURES].astype(float).values, "Training z-score": z_values.values}
)
left, right = st.columns([1, 1.3])
with left:
    st.subheader("Sensor values")
    st.dataframe(sensor_table, hide_index=True, use_container_width=True)
with right:
    fig = px.bar(sensor_table, x="Feature", y="Training z-score", color="Training z-score",
                 color_continuous_scale="RdBu_r", color_continuous_midpoint=0, title="Deviation from training distribution")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Failure-mode metadata")
failure_metadata = pd.DataFrame({"Failure mode": FAILURE_MODE_COLUMNS, "Label": [int(row[c]) for c in FAILURE_MODE_COLUMNS]})
st.dataframe(failure_metadata, hide_index=True, use_container_width=True)
st.caption("Failure-mode labels are display metadata only and were not used as model inputs.")

