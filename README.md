# VerityAI — Explainable Fake-Profile Intelligence

A responsive React + Vite frontend and FastAPI backend for the SR University fake-profile detection capstone.

## What's included
- Research dashboard with the actual notebook's recorded dataset and evaluation metrics.
- Profile Analyzer (clearly marked demo until trained artifacts are installed).
- Dataset Lab: CSV schema, missing-value, duplicate and class-count inspection.
- Model Performance: classical baseline metrics and hybrid ablation results.
- Hybrid Architecture: Feature Attention + genuine-only Autoencoder + feature-similarity GCN + 41-D fusion head.
- Explainability Studio: attention patterns, anomaly threshold, SHAP methods and an explicitly labelled false-positive case study.
- Responsive dark research UI.

## Important scientific honesty
The interactive analyzer currently uses a transparent illustrative heuristic until the model-artifact ZIP exported from the executed notebook is installed. It is not a trained prediction. The displayed research metrics are recorded notebook results. The notebook's GCN experiment used a transductive graph containing fit, validation and held-out test feature vectors (without test labels); query-time serving against stored fit references is an inductive adaptation and must be evaluated separately before claiming the original test metrics validate deployment.

## Correct folder structure
After extracting this ZIP, `backend` and `frontend` should be directly inside the extracted project folder:

```text
VerityAI/
  backend/
  frontend/
  README.md
  NOTEBOOK_MODEL_CONNECTION.md
  EXPORT_NOTEBOOK_MODELS.py
```

## Run on Windows PowerShell
Open TWO PowerShell terminals. In the first, change directory into the extracted project folder, then run:

```powershell
cd backend
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

If PowerShell blocks activation, run this once in that terminal, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

In the SECOND terminal, go to the same extracted project folder:

```powershell
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite (normally http://localhost:5173). Keep both terminals running. API docs: http://127.0.0.1:8000/docs.

## Connect the trained notebook
1. Open `capstone_verityai_export_ready.ipynb` in Google Colab (or add the provided `EXPORT_NOTEBOOK_MODELS.py` as the final cell in your executed notebook).
2. Run all notebook cells in order so the models are trained in the current runtime.
3. Run the final export cell. It creates `VerityAI_Model_Artifacts.zip`.
4. Download that ZIP from Colab and extract its contents into `backend/model_artifacts/`.
5. The export contains the four PyTorch state dictionaries, scaler, ordered feature list, metadata, fit-reference feature vectors and best classical baseline.

The current API intentionally remains in DEMO HEURISTIC mode until a model-serving adapter is installed and validated. Do not silently relabel the heuristic as a trained model. The GCN's query-time graph construction is a serving adaptation of the notebook's transductive experiment and requires a separate validation run.
