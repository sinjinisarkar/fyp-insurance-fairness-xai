# Transparent and Fair ML for Young Driver Insurance Risk Prediction

[![Streamlit App](https://img.shields.io/badge/Live%20Demo-Hugging%20Face-blue)](https://huggingface.co/spaces/sc23ss2/fyp-insurance-fairness)
[![Python](https://img.shields.io/badge/Python-3.12.7-green)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

A final year dissertation project at the University of Leeds, School of Computing. This project builds a transparent and fair machine learning framework for predicting motor insurance risk for young UK drivers (aged 17–25), using publicly available UK road safety data.

## Live Demo

Try the deployed Streamlit application here: **[https://sc23ss2-fyp-insurance-fairness.hf.space/](https://sc23ss2-fyp-insurance-fairness.hf.space/)**

---
## Project Overview

Young drivers in the UK face significantly higher insurance premiums than older drivers, with age cited as the primary justification. This project investigates whether this is actually supported by the data — and whether ML models used for risk prediction treat different groups of drivers fairly.

**Key findings:**
- `age_of_driver` ranks only **13th out of 49 features** in terms of predictive importance
- Road environment features (police force area, number of vehicles, speed limit) matter far more than age
- No single model achieves fairness across both sex and age band dimensions simultaneously
- Intersectional fairness gaps are substantially larger than individual attribute gaps

---
## Project Structure
```
fyp-insurance-fairness-xai/
├── data/
│   └── processed/          # Cleaned datasets 
│   └── raw/                # Uncleaned datasets
│   └── figures/            # Saved figures 
├── models/                 # Trained model files (.pkl)
│   ├── feature_names.pkl
│   ├── lr_balanced.pkl
│   ├── rf_balanced.pkl
│   ├── xgb.pkl
├── notebooks/
│   ├── 01_data_loading_merge_audit.ipynb
│   ├── 02_feature_engineering_and_preprocessing.ipynb
│   ├── 03_baseline_model.ipynb
│   ├── 04_tree_baseline_models.ipynb
│   ├── 05_fairness_metrics.ipynb
│   └── 06_shap_explanations.ipynb
│   └── 07_save_models.ipynb
├── src/
│   └── app.py              # Streamlit application code
├── tests/
│   ├── __init__.py
│   ├── test_data_pipeline.py
│   ├── test_fairness.py
│   ├── test_leakage.py
│   ├── test_models.py
│   └── test_shap.py
├── README.md
└── requirements.txt
```
---

## Running Locally

### Prerequisites

- Python 3.11 or higher
- Git

### Step 1 — Clone the repository

```bash
git clone https://github.com/sinjinisarkar/fyp-insurance-fairness-xai.git
cd fyp-insurance-fairness-xai
```

### Step 2 — Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
# or
.venv\Scripts\activate     # Windows
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 5 — Run the notebooks

Run the notebooks in order: 01 → 02 → 03 → 04 → 05 → 06

This will generate the cleaned datasets and trained models.

### Step 6 — Run the Streamlit app

```bash
streamlit run src/app.py
```

The app will open at `http://localhost:8501`

---
## Running Tests

```bash
pytest tests/ -v
```

All 5 test files should pass:
- `test_data_pipeline.py` — data cleaning and filtering
- `test_leakage.py` — leakage prevention
- `test_models.py` — model loading and predictions
- `test_fairness.py` — fairness metrics
- `test_shap.py` — SHAP explainability

---
## Models

Three models were trained and evaluated:

| Model | Threshold | Accuracy | Recall | ROC-AUC |
|-------|-----------|----------|--------|---------|
| Logistic Regression | 0.5 | 57.8% | 59.0% | 0.629 |
| Random Forest | 0.3 | 65.0% | 61.4% | 0.689 |
| XGBoost | 0.3 | 69.1% | 51.8% | 0.689 |

---
## Fairness Results

| Model | DP (sex) | EO (sex) | DP (age) | EO (age) |
|-------|----------|----------|----------|----------|
| Logistic Regression | 0.054 | 0.040 | 0.147 | 0.175 |
| Random Forest | 0.125 | 0.118 | 0.133 | 0.166 |
| XGBoost | 0.136 | 0.124 | 0.089 | 0.149 |

---
## Tech Stack

| Tool | Version |
|------|---------|
| Python | 3.12.7 |
| scikit-learn | 1.8.0 |
| XGBoost | 3.1.3 |
| SHAP | 0.50.0 |
| Fairlearn | 0.13.0 |
| Streamlit | 1.53.1 |
| pandas | 2.3.3 |
| numpy | 2.3.5 |

---
## Data

The raw DfT STATS19 2024 road safety data is included in `data/raw/` for reproducibility:
- `dft-road-casualty-statistics-collision-2024.csv` (19MB)
- `dft-road-casualty-statistics-vehicle-2024.csv` (19MB)

Data is published under the Open Government Licence v3.0 by the UK Department for Transport.
Original source: https://www.data.gov.uk/dataset/road-accidents-safety-data

Processed datasets are included in `data/processed/` so you can skip directly to model training if preferred.

---
## Author

**Sinjini Sarkar**
MEng, BSc Computer Science with AI
University of Leeds, 2026

---
## Licence

This project is licensed under the MIT Licence.
See [LICENSE](LICENSE) for details.