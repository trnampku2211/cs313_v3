from __future__ import annotations

from pathlib import Path

import streamlit as st

from .artifacts import load_joblib, load_json
from .data import load_dataset


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "ai4i2020.csv"
BENCHMARK_PATH = ROOT / "artifacts" / "benchmark" / "official_benchmark.joblib"
RESULTS_PATH = ROOT / "results" / "benchmark_results.json"


def configure_page(title: str, icon: str = "⚙️") -> None:
    st.set_page_config(page_title=title, page_icon=icon, layout="wide")
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1500px;}
        [data-testid="stMetric"] {background: #111827; border: 1px solid #263244;
            border-radius: 12px; padding: 14px 16px;}
        [data-testid="stMetricLabel"] {color: #9ca3af;}
        div[data-testid="stAlert"] {border-radius: 10px;}
        h1, h2, h3 {letter-spacing: -0.02em;}
        .method-note {border-left: 4px solid #22c55e; background: rgba(34,197,94,.08);
            padding: .8rem 1rem; border-radius: 0 8px 8px 0; margin-bottom: 1rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def cached_dataset():
    return load_dataset(DATA_PATH)


@st.cache_resource(show_spinner=False)
def cached_benchmark_bundle():
    if not BENCHMARK_PATH.exists():
        return None
    return load_joblib(BENCHMARK_PATH)


@st.cache_data(show_spinner=False)
def cached_benchmark_results():
    if not RESULTS_PATH.exists():
        return None
    return load_json(RESULTS_PATH)


def benchmark_required() -> bool:
    if BENCHMARK_PATH.exists() and RESULTS_PATH.exists():
        return True
    st.warning(
        "Official benchmark artifacts are not available yet. Run "
        "`python scripts/run_benchmark.py` after installing the requirements."
    )
    return False


def protocol_badge() -> None:
    st.markdown(
        '<div class="method-note"><b>Official protocol:</b> 60/20/20 stratified split, '
        'random_state=42, train-only fitting, label-guided validation, and one untouched test set.</div>',
        unsafe_allow_html=True,
    )

