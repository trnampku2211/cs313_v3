from pathlib import Path

import pandas as pd
import streamlit as st

from src.ui import ROOT, benchmark_required, cached_benchmark_results, configure_page, protocol_badge


configure_page("Experiment Results", "🧪")
st.title("Experiment Results")
st.caption("Configuration audit, run history, and reproducibility metadata")
protocol_badge()

if not benchmark_required():
    st.stop()

payload = cached_benchmark_results()
st.subheader("Protocol")
st.json(payload["protocol"])

st.subheader("Environment")
st.dataframe(pd.DataFrame([payload["environment"]]), hide_index=True, use_container_width=True)

history_path = ROOT / "results" / "experiments.csv"
if history_path.exists():
    history = pd.read_csv(history_path)
    st.subheader("Saved experiment history")
    st.dataframe(history, hide_index=True, use_container_width=True)
    st.download_button(
        "Export experiment history as CSV",
        data=history.to_csv(index=False).encode("utf-8"),
        file_name="experiments.csv",
        mime="text/csv",
    )
else:
    st.info("No experiment history CSV is available.")

st.subheader("Selected configurations")
for result in payload["results"]:
    with st.expander(result["model"]):
        st.json(
            {
                "learning_type": result["learning_type"],
                "parameters": result["parameters"],
                "validation_metrics": result["validation_metrics"],
                "test_metrics": result["test_metrics"],
                "score_source": result["score_source"],
            }
        )

if payload.get("sensitivity_results"):
    st.subheader("Sensitivity experiments")
    st.caption("Sensitivity runs are reported separately and are not treated as additional comparison models.")
    for result in payload["sensitivity_results"]:
        with st.expander(result["model"]):
            st.json(result)

st.caption("Official benchmark artifacts are loaded from local joblib/JSON files. No database is used.")
