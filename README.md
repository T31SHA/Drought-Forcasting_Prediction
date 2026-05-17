# Drought-Forcasting_Prediction
# 🌵 SPEI Drought Forecasting & Intelligence Platform

> Multi-horizon drought prediction for Northwestern Algeria using 76 years of climate data.

[![Live App](https://img.shields.io/badge/Live%20App-drought--intelligence.streamlit.app-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://drought-intelligence.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## Overview

This project builds a complete end-to-end machine learning pipeline to **forecast drought conditions** using the **Standardised Precipitation-Evapotranspiration Index (SPEI)** at a grid point in Northwestern Algeria (35.75°N, 0.75°E).

The dataset spans **January 1950 to February 2026** — 914 monthly observations across 48 SPEI accumulation windows. The primary target is **SPEI-12**, the 12-month accumulation window and the international standard for hydrological drought monitoring.

---

## Live Demo

**[→ drought-intelligence.streamlit.app](https://drought-intelligence.streamlit.app/)**

Upload the CSV file and the full pipeline — data exploration, model performance, uncertainty quantification, and feature intelligence — runs automatically in the browser.

---

## What is SPEI?

SPEI is a dimensionless z-score measuring the balance between precipitation and atmospheric water demand. Values follow N(0,1) by construction.

| SPEI-12 Value | Classification |
|:---:|:---|
| < −2.0 | 🔴 Extreme drought |
| −2.0 to −1.5 | 🟠 Severe drought |
| −1.5 to −1.0 | 🟡 Moderate drought |
| −1.0 to +1.0 | ⚪ Near normal |
| > +1.0 | 🔵 Wet conditions |

---

## Repository Structure

```
Drought-Forcasting_Prediction/
│
├── drought_forecasting_.ipynb    # Full ML pipeline (10 stages)
├── app.py                        # Streamlit interactive dashboard
├── SPEI_0.75_35.75.csv           # Raw SPEI data (Jan 1950 → Feb 2026)
├── model_results.pkl             # Trained model objects + predictions
├── metrics_comparison.csv        # RMSE, MAE, R², MAPE, Bias, Hit Rate
└── classification_report.csv     # WMO drought class precision/recall/F1
```

---

## Pipeline — 10 Stages

| Stage | Description |
|:---:|:---|
| 1 | Data ingestion & datetime parsing |
| 2 | Exploratory data analysis — 6 publication-quality plots |
| 3 | Feature engineering — lags, rolling statistics, interactions |
| 4 | Chronological train / validation / test split |
| 5 | Model training — Ridge, XGBoost across 3 forecast horizons |
| 6 | Evaluation — RMSE, MAE, R², MAPE, Bias, Hit Rate |
| 7 | Feature importance & SHAP explainability |
| 8 | WMO drought classification — confusion matrix, Cohen's κ |
| 9 | Bootstrap 90% prediction intervals |
| 10 | Report generation & model serialisation |

---

## Models & Forecast Horizons

Three forecast horizons are evaluated: **t+1**, **t+3**, and **t+6** months ahead.

**Models trained:**
- **Ridge Regression** — regularised linear baseline
- **XGBoost** — gradient boosted trees with Optuna hyperparameter tuning

**Split (strictly chronological — no shuffling):**

| Split | Period |
|:---|:---|
| Train | Jan 1950 → Dec 2005 |
| Validation | Jan 2006 → Dec 2015 |
| Test | Jan 2016 → Feb 2026 |

---

## Features Engineered (28 total)

- **Temporal:** month sin/cos encoding, year, decade, month counter
- **Autoregressive lags:** SPEI-12 at t−1, t−2, t−3, t−6, t−9, t−12, t−24
- **Cross-scale lags:** SPEI-1 and SPEI-6 at t−1, t−3
- **Rolling statistics:** 3/6/12-month mean, 3/6-month std, 12-month min/max
- **Interactions:** SPEI-1 − SPEI-12, SPEI-6 − SPEI-24, SPEI-12 × SPEI-24

---

## Dashboard Pages

| Page | What it shows |
|:---|:---|
| 🌍 **Overview** | KPI cards, 76-year time series, decade averages, worst drought events |
| 📊 **Data Explorer** | Multi-scale SPEI, calendar heatmap, distribution tests, seasonality |
| 🤖 **Model Performance** | Metrics table, actual vs predicted, residuals, WMO confusion matrix |
| 🎯 **Forecast & Uncertainty** | Bootstrap 90% & 50% PIs, PI width, monthly coverage reliability |
| 🔬 **Feature Intelligence** | Ridge coefficients, XGBoost importance, feature correlation heatmap |

---

## Quick Start

```bash
# Clone
git clone https://github.com/T31SHA/Drought-Forcasting_Prediction.git
cd Drought-Forcasting_Prediction

# Install dependencies
pip install streamlit plotly pandas numpy scikit-learn scipy xgboost joblib

# Run the app
streamlit run app.py
```

Then upload `SPEI_0.75_35.75.csv` in the sidebar. Everything computes automatically.

To re-run the full notebook pipeline:

```bash
jupyter notebook drought_forecasting_.ipynb
```

---

## Data Source

SPEI data retrieved from the **SPEI Global Drought Monitor** (CSIC, Zaragoza, Spain), derived from the CRU TS4.x monthly climatology at 0.5° grid resolution.

**Reference:** Vicente-Serrano, S.M., Beguería, S., López-Moreno, J.I. (2010). A Multiscalar Drought Index Sensitive to Global Warming: The Standardized Precipitation Evapotranspiration Index. *Journal of Climate*, 23(7), 1696–1718.

---

## Study Region

**Northwestern Algeria** is a semi-arid Mediterranean region under accelerating drying pressure. The 1984–1985 multi-year drought (SPEI-12 minimum: −2.29, October 1984) remains the most severe on record. Recent years (2020–2026) show an anomalous positive SPEI trend — the wettest period in the 76-year record.

---

*Built with Python · scikit-learn · XGBoost · Streamlit · Plotly*
