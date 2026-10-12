import pandas as pd
import plotly.express as px
import streamlit as st

from src.data import FAILURE_MODE_COLUMNS, NUMERIC_FEATURES, TARGET_COLUMN, quality_report
from src.ui import cached_dataset, configure_page, protocol_badge


configure_page("Dataset Explorer", "📊")
st.title("Dataset Explorer")
st.caption("Source data quality, class balance, and sensor distributions")
protocol_badge()

frame = cached_dataset()
report = quality_report(frame)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{report.rows:,}")
c2.metric("Columns", report.columns)
c3.metric("Machine failures", f"{report.positive_count:,}")
c4.metric("Failure rate", f"{report.positive_rate:.2%}")

left, right = st.columns(2)
with left:
    counts = frame[TARGET_COLUMN].map({0: "Normal", 1: "Failure"}).value_counts().rename_axis("Class").reset_index(name="Count")
    fig = px.bar(counts, x="Class", y="Count", color="Class", title="Class distribution",
                 color_discrete_map={"Normal": "#2563eb", "Failure": "#ef4444"})
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
with right:
    by_type = frame.groupby("Type", observed=True)[TARGET_COLUMN].agg(["count", "sum", "mean"]).reset_index()
    by_type.columns = ["Type", "Records", "Failures", "Failure rate"]
    fig = px.bar(by_type, x="Type", y="Records", color="Failure rate", text="Failures",
                 title="Product type distribution and failures", color_continuous_scale="Reds")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Feature distribution")
feature = st.selectbox("Numerical feature", NUMERIC_FEATURES)
plot_frame = frame.assign(Status=frame[TARGET_COLUMN].map({0: "Normal", 1: "Failure"}))
fig = px.histogram(plot_frame, x=feature, color="Status", marginal="box", barmode="overlay",
                   opacity=0.65, color_discrete_map={"Normal": "#2563eb", "Failure": "#ef4444"})
st.plotly_chart(fig, use_container_width=True)

left, right = st.columns([1.2, 1])
with left:
    st.subheader("Numerical correlation")
    corr_columns = NUMERIC_FEATURES + [TARGET_COLUMN]
    corr = frame[corr_columns].corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    st.plotly_chart(fig, use_container_width=True)
with right:
    st.subheader("Data quality")
    quality = pd.DataFrame(
        {
            "Check": ["Missing cells", "Exact duplicate rows", "UDI unique", "Product ID unique"],
            "Result": [report.missing_values, report.exact_duplicates, report.udi_unique, report.product_id_unique],
        }
    )
    st.dataframe(quality, hide_index=True, use_container_width=True)
    st.markdown("**Failure-mode positives (metadata only)**")
    st.dataframe(
        pd.DataFrame({"Failure mode": FAILURE_MODE_COLUMNS, "Positive rows": [int(frame[c].sum()) for c in FAILURE_MODE_COLUMNS]}),
        hide_index=True,
        use_container_width=True,
    )

with st.expander("Feature policy and leakage controls", expanded=False):
    st.markdown(
        """
        - Model inputs: five numerical sensor features plus one-hot encoded `Type`.
        - Excluded identifiers: `UDI`, `Product ID`.
        - Excluded labels: `Machine failure`, `TWF`, `HDF`, `PWF`, `OSF`, `RNF`.
        - Failure-mode labels are retained only as Anomaly Inspector metadata.
        - `Product ID` is unique and its prefix duplicates `Type`; `UDI` is an identifier, not a verified timestamp.
        """
    )

st.subheader("Data preview")
st.dataframe(frame, use_container_width=True, height=340)

