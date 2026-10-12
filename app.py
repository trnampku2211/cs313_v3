import streamlit as st

from src.ui import configure_page, protocol_badge


configure_page("Industrial Anomaly Detection", "🏭")

st.title("Industrial Machine Anomaly Detection")
st.caption("DBSCAN-centered anomaly detection with comparative machine-learning methods")
protocol_badge()

left, right = st.columns([1.35, 1])
with left:
    st.subheader("Project objective")
    st.write(
        "Detect unusual machine operating states in the AI4I 2020 dataset and compare "
        "DBSCAN with Isolation Forest, Local Outlier Factor, and a supervised Random Forest baseline."
    )
    st.subheader("Navigate the study")
    st.markdown(
        """
        1. **Dataset Explorer** — inspect quality, imbalance, and sensor distributions.
        2. **DBSCAN Explorer** — interact with the primary algorithm and its density structure.
        3. **Model Comparison** — review the locked official benchmark.
        4. **Anomaly Inspector** — compare model decisions record by record.
        5. **Experiment Results** — audit configurations, artifacts, and run history.
        """
    )
with right:
    st.subheader("Methodological boundary")
    st.info(
        "Interactive DBSCAN runs are educational experiments. They do not overwrite or tune the "
        "official benchmark, which uses fixed validation-selected configurations."
    )
    st.markdown(
        """
        **Primary model:** DBSCAN  
        **Comparison models:** Isolation Forest, LOF, Random Forest  
        **Primary selection metric:** Validation F1  
        **Ranking context:** Supervised and unsupervised models are identified separately
        """
    )

st.divider()
st.caption("Academic localhost dashboard • English interface • Reproducible evaluation protocol")

