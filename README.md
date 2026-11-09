# Autonomous Data Scientist

A Streamlit workspace for supervised data science: CSV exploration, preprocessing, cross-validation model comparison, held-out evaluation, and Markdown reporting.

## Setup

From the project root in PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Use the existing environment; dependencies only need installing if they are missing. To launch directly without activation, run from the project root:

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`). Stop the server with Ctrl+C.

## Using the dashboard

1. Upload a UTF-8, comma-separated CSV with unique, nonblank headers and at least one feature plus a target column.
2. Preview the dataset and choose the target. The Planner suggests a task when possible; ambiguous numerical labels require an explicit classification/regression selection.
   Use **Exclude feature columns** to remove identifiers or other unsuitable inputs. `Id` (case-insensitive) is excluded by default when it is a feature; you can change the selection. Keep at least one feature selected. Changing exclusions hides stale results without retraining.
3. Set the test fraction, maximum CV folds, random seed, and numerical scaling in the sidebar.
4. Click **Run Autonomous Analysis**. Training runs only on this button click.
5. Explore the Plan, EDA, Preprocessing, Models, Evaluation, and Report tabs. Download the Markdown report from the Report tab.

Completed results persist in the current Streamlit session. The complete CSV contents, target, task, and run settings identify the results; changing them hides stale results until a matching analysis is run. Ordinary reruns, tab interactions, and downloads do not retrain models. Closing the session or restarting the server clears in-memory results; saved charts and reports remain on disk.

## Analysis behavior

The six agents run in order: Planner → EDA → Preprocessing → Modeling → Evaluation → Reporting. Existing agent interfaces remain available for direct Python use. `utils/workflow.py` provides `parse_csv`, `analysis_key`, and `run_analysis` for integrations without Streamlit.

Preprocessing and modeling use the same deterministic train/test split. The Modeling Agent independently prepares that split through its existing interface, clones the preprocessing configuration, and fits it afresh inside every training-only CV fold. Classification compares Logistic Regression, Decision Tree, and Random Forest by stratified-CV accuracy; regression compares Linear Regression, Decision Tree, and Random Forest by mean CV RMSE. Test performance is never used for selection. The selected pipeline's held-out predictions are passed to Evaluation; classification probabilities include explicit class ordering for ROC-AUC.

Feature missing values are imputed using training data. Missing target labels are rejected; the application never silently removes or fills them. Entirely missing training feature columns are dropped with a warning. Unknown categories are ignored during encoding. Unsupported feature types, insufficient class counts for CV, and failed model fits produce actionable messages. Completed stages remain visible if a later stage fails. Undefined evaluation metrics display **Unavailable** with an explanation.

EDA charts are saved in `outputs/eda/`, evaluation charts in `outputs/evaluation/`, and uniquely named reports in `reports/`. Report downloads contain Markdown with relative chart references: retain the corresponding outputs folders when sharing the report with its images. The dashboard renders these chart references locally. Saved files are never overwritten by a new run.

## Tests

```powershell
.\venv\Scripts\python.exe -m pytest tests -q
```

Tests cover each agent, real classification/regression workflows, leakage safeguards, CSV validation, and Streamlit interactions including rerun persistence and stale-result isolation. Test-generated charts and reports use pytest temporary folders.

## Project structure

- `agents/`: planner, EDA, preprocessing, modeling, evaluation, and reporting implementations.
- `datasets/`: input datasets.
- `outputs/`: generated analysis outputs.
- `reports/`: generated reports.
- `utils/`: CSV validation and end-to-end workflow orchestration.
- `app.py`: Streamlit dashboard.
- `tests/`: agent, workflow, and dashboard tests.
- `requirements.txt`: Python dependencies.

Generated outputs and reports are ignored by Git.
