# Industrial Machine Anomaly Detection

An English-language Streamlit dashboard for detecting unusual operating states in the AI4I 2020 predictive-maintenance dataset. DBSCAN is the primary algorithm; Isolation Forest, Local Outlier Factor, and Random Forest provide comparison baselines.

## Methodology

- Fixed 60/20/20 stratified split with `random_state=42`.
- The preprocessor and all estimators are fitted on training data only.
- Validation labels select hyperparameters and F1-optimal thresholds.
- The test set is evaluated only after configurations are locked.
- DBSCAN test prediction uses distance to training core samples; this is an explicit project extension, not native DBSCAN prediction.
- Unsupervised estimators are fitted without target labels, but their model selection is label-guided.
- Interactive DBSCAN exploration is separate from the official benchmark.

## Input features

The primary feature set contains `Type` plus five numerical sensor features. `Type` is one-hot encoded and numerical features are standardized. Identifiers, the target, and the five failure-mode labels are excluded from model inputs.

The benchmark also records a separate numeric-only DBSCAN sensitivity experiment. It is not treated as a fifth model in the official comparison table.

## Setup on Windows

Install Python 3.11 or 3.12, then run:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Verify the project

```powershell
python -m pytest -q
```

## Create official benchmark artifacts

For a short smoke run:

```powershell
python scripts/run_benchmark.py --quick
```

For the complete validation search and official test evaluation:

```powershell
python scripts/run_benchmark.py
```

The benchmark writes local joblib, JSON, CSV, and split-index artifacts. Re-running it creates a new experiment-history entry. The dashboard does not train models automatically.

## Start the dashboard

```powershell
python -m streamlit run app.py
```

Open the localhost URL printed by Streamlit. The DBSCAN Explorer can run independent interactive demonstrations, while Model Comparison and Anomaly Inspector load the saved official benchmark.

## Project layout

```text
app.py                       Streamlit landing page
pages/                       Five dashboard pages
src/data.py                  Schema and feature policy
src/splitting.py             Reproducible official split
src/preprocessing.py         Train-fitted transformations
src/models/                  Four model implementations
src/benchmark.py             Validation selection and final evaluation
src/evaluation.py            Shared metrics and threshold optimization
src/artifacts.py             Atomic local artifact persistence
scripts/run_benchmark.py     Benchmark command-line entry point
tests/                       Data, split, model, metric, and artifact tests
```

## Interpretation warning

DBSCAN noise means the point is not density-reachable under the selected feature representation and parameters. It does not by itself prove a physical machine failure. Random Forest is supervised and therefore has access to training labels that the three unsupervised estimators do not use during fitting.
