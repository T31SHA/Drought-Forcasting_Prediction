# Drought Outlook: SPEI-12 forecasting for Northwestern Algeria

**Author: Sharahbill Mohamed Abdi**

Monthly drought forecasts 1, 3 and 6 months ahead for one 0.5° grid cell in Northwestern Algeria (35.75°N, 0.75°E). The pipeline is reproducible and leakage-checked, the evaluation is honest, and the results are served by a static, accessible web dashboard on Netlify.

- **Web dashboard (Netlify):** https://drought-forcast.netlify.app/, built from [`web/`](web/)
- **Original Streamlit app:** [`app.py`](app.py), live at https://drought-intelligence.streamlit.app/
- **Every number in this README** is produced by `python -m pipeline.train` and saved in [`reports/RESULTS.md`](reports/RESULTS.md).

---

## Contents

1. [What is forecast](#what-is-forecast)
2. [Repository layout](#repository-layout)
3. [Reproduce everything](#reproduce-everything)
4. [Method](#method)
5. [Results](#results), with before vs after
6. [Error by season and by regime](#error-by-season-and-by-regime)
7. [Explainability](#explainability)
8. [Uncertainty](#uncertainty)
9. [Limitations](#limitations)
10. [Web dashboard](#web-dashboard)

---

## What is forecast

**SPEI-12**, the 12-month Standardised Precipitation-Evapotranspiration Index (Vicente-Serrano et al., 2010), is a standard score of rainfall minus atmospheric water demand summed over the last 12 months. It tracks hydrological drought.

| SPEI-12 | Class |
|---|---|
| ≤ −2.0 | Extreme drought |
| −2.0 to −1.5 | Severe drought |
| −1.5 to −1.0 | Moderate drought |
| −1.0 to +1.0 | Near normal |
| +1.0 to +1.5 | Moderately wet |
| +1.5 to +2.0 | Very wet |
| ≥ +2.0 | Extremely wet |

**Data:** `SPEI_0.75_35.75.csv`, 914 months (Jan 1950 to Feb 2026) × 48 accumulation windows, from the SPEI Global Drought Monitor (CSIC, CRU TS climatology).

---

## Repository layout

```
config.yaml                 # every setting: seed, split dates, features, search space, CV
requirements-pipeline.txt   # pinned environment for the pipeline (Python 3.14)
pipeline/
  data.py                   # load CSV, causal fill of the one missing SPEI-1 month
  features.py               # leakage-safe features, purged time split, season/regime labels
  models.py                 # persistence, climatology, damped persistence, Ridge, XGBoost, blend
  evaluate.py               # metrics, conformal intervals, block bootstrap
  legacy.py                 # exact re-implementation of the original notebook's Ridge
  train.py                  # tune → CV → select → test once → refit → export
tests/test_pipeline.py      # leakage / protocol tests (pytest)
models/                     # xgboost_h{1,3,6}.json, ridge_h*.joblib, selected_h*.joblib
reports/                    # RESULTS.md, metrics.json, CSVs, bootstrap, importances, best params
web/                        # Vite + TypeScript dashboard (Netlify)
netlify.toml                # Netlify build config (base = web/)
drought_forecasting_.ipynb  # original notebook (unchanged)
app.py                      # original Streamlit app (unchanged)
requirements.txt            # original environment for app.py / notebook (unchanged)
```

---

## Reproduce everything

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-pipeline.txt

python -m pytest -q                         # 5 leakage/protocol tests
python -m pipeline.train --config config.yaml   # ~5 min on 8 CPU cores
```

The run is deterministic: a fixed seed (`seed: 42`), a seeded Optuna TPE sampler, `n_jobs=1` for the search, and fixed XGBoost threads. Two consecutive runs produced identical tuned hyperparameters and metrics. Outputs:

- `models/`: production models refit on all observed data, used for the live forecast
- `reports/metrics.json`, `metrics_test.csv`, `metrics_cv.csv`: all metrics
- `reports/RESULTS.md`: every table in this README
- `web/src/data/forecast.json`: the payload the dashboard renders

Web app:

```bash
cd web && npm ci
npm run dev        # http://localhost:5173
npm run build      # type-check + production build to web/dist
npm run preview    # serve the build on http://localhost:4173
```

---

## Method

### Features (37, all known at the forecast month *t*)

| Group | Features |
|---|---|
| Current multi-scale SPEI | SPEI-1, 2, 3, 4, 5, 6, 9, 11, 12, 18, 24, 36 at *t*. SPEI-(12−h) is the part of the future 12-month window that is already observed |
| Autoregressive lags | SPEI-12 at *t*−1, −2, −3, −6, −12; SPEI-1 at *t*−1, −2, −3 |
| SPI-style accumulations | rolling mean and std of SPEI-1 over 3/6/12 months; rolling mean of SPEI-12 over 3/6/12 |
| Momentum and contrasts | SPEI-12 change over 1 and 3 months; SPEI-3 − SPEI-12; SPEI-12 − SPEI-24 |
| Drought regime | months in the current drought (SPEI-12 < −1); months since the last drought |
| Season | month sine and cosine |

Choices worth knowing:

- **No year or time-trend features.** Trees cannot extrapolate them, and a trend fitted on 1950–2015 is a fragile assumption for the future. See [Results](#results) for what this cost on the test decade.
- **SPEI-48 is not used.** The source file leaves it empty for Nov 2025 to Feb 2026, so a model that needed it could not forecast from the latest month.
- **SPEI-1 for Nov 2025 is missing in the source file.** It is filled by a regression `SPEI_1[t] ~ SPEI_2[t] + SPEI_1[t−1]` fitted only on earlier months (5-fold CV R² 0.915, RMSE 0.29). Without this, every SPEI-1 lag and rolling feature would be undefined for the following year.
- **NDVI was requested but is not available.** The repo contains no vegetation data, and satellite NDVI only starts in 1981. Adding it would mean a new external data source, so it is listed under [Limitations](#limitations) rather than approximated.

### Leakage controls

- **Purged split.** Development rows are forecast origins whose *target* month is ≤ Dec 2015. The test set is origins Jan 2016 to Aug 2025, the same 116 forecasts per horizon as the original notebook.
- **Time-series CV.** `TimeSeriesSplit` with 5 expanding folds, a 240-month minimum training window and `gap = h`, so no training label overlaps a validation month. Validation covers 1973–2015.
- **Test touched once.** Hyperparameters and model selection use CV only. The test window is scored after selection.
- **Tests.** `tests/test_pipeline.py` checks that features at month *t* are identical whether or not any later data exists, that splits and folds are purged, and that the SPEI-1 imputation is causal.

### Models compared (per horizon)

| Model | Description |
|---|---|
| Persistence | ŷ(t+h) = SPEI-12(t) |
| Climatology | mean SPEI-12 of the target calendar month (training rows only) |
| Damped persistence | ŷ = a + b·SPEI-12(t), OLS |
| Ridge | standardised features. α ∈ {0.01 … 1000} tuned by CV |
| XGBoost | 60-trial Optuna TPE search over 8 hyperparameters, scored by CV RMSE |
| Ridge + XGBoost | equal-weight average |
| Previous Ridge | the original notebook's model, re-implemented exactly (`pipeline/legacy.py`) |

Ridge and XGBoost learn the **change from persistence** (y(t+h) − SPEI-12(t)). Regularisation then shrinks toward "no change" rather than toward the long-term mean.

---

## Results

### Before vs after: the headline

The original notebook's numbers are reproduced exactly by `pipeline/legacy.py` (RMSE 0.228 / 0.481 / 0.701, matching `metrics_comparison.csv`). That gives an exact baseline to compare against.

**Test window, 116 forecasts per horizon, origins 2016-01 to 2025-08**

| Horizon | Persistence | Previous Ridge (before) | New selected: Ridge (after) | XGBoost (tuned) | Ridge + XGBoost |
|---|---|---|---|---|---|
| t+1 RMSE | 0.341 | **0.228** | 0.238 | 0.283 | 0.248 |
| t+3 RMSE | 0.728 | **0.481** | 0.547 | 0.607 | 0.557 |
| t+6 RMSE | 1.073 | **0.701** | 0.904 | 0.861 | 0.857 |

**Cross-validation, 1973–2015, about 510 out-of-fold forecasts per horizon, same folds for every model**

| Horizon | Persistence | Previous Ridge (before) | New Ridge (after) | XGBoost | Ridge + XGBoost |
|---|---|---|---|---|---|
| t+1 RMSE | 0.345 | 0.256 | **0.238** | 0.301 | 0.256 |
| t+3 RMSE | 0.655 | 0.484 | **0.467** | 0.535 | 0.483 |
| t+6 RMSE | 0.967 | 0.708 | **0.687** | 0.729 | 0.690 |

**Plain reading:**

1. **On the 2016–2025 test decade, the new model is worse than the previous one at every horizon.** RMSE rose from 0.228 to 0.238 at 1 month (+4%), 0.481 to 0.547 at 3 months (+14%), and 0.701 to 0.904 at 6 months (+29%). R² fell from 0.951 to 0.946, 0.782 to 0.718, and 0.539 to 0.233. A moving-block bootstrap (12-month blocks, 2000 resamples) gives the previous Ridge a lower RMSE in 93%, 98% and 98% of resamples, so this is not noise.
2. **On the 43-year cross-validation backtest, the new Ridge is better at every horizon** (−7%, −4%, −3% RMSE), and that is why it was selected. Selection was made before the test window was scored, and I did not change the selection after seeing the test result.
3. **Why the two disagree.** A diagnostic (not used for selection) removed the previous model's `year`, `decade` and `month_since_start` features. Its 6-month test RMSE rose from 0.701 to 0.744, with bias going from −0.17 to −0.30. Its trend features suit the unusually wet 2018–2025 period. The new model, which has no trend term, under-predicts that wet run: at 6 months in wet months its bias is −1.00, against −0.56 for the previous model. The remaining gap is likely due to the previous model's 37–48-month SPEI windows and its alpha being tuned on 2006–2015 alone. Whether trend extrapolation helps in the next decade is unknown. It helped in this one.
4. **Every learned model beats persistence** on both test and CV at every horizon. Test skill (1 − MSE/MSE_persistence): 51% / 44% / 29% for the new Ridge, 55% / 56% / 57% for the previous Ridge.
5. **Tuned XGBoost does not beat Ridge** in CV at any horizon, despite a 60-trial search. On the test window it beats the new Ridge only at 6 months (0.861 vs 0.904). The signal here is mostly linear persistence, and 500–700 training months is little data for trees.
6. **Climatology is the worst model** at all horizons on the test window (RMSE about 1.27, bias −0.75), because 2016–2025 was far wetter than the long-term average.

### Full test-window metrics (selected model)

| Horizon | RMSE | MAE | R² | Explained var. | Bias | Skill vs persistence | 80% PI coverage | 95% PI coverage |
|---|---|---|---|---|---|---|---|---|
| t+1 | 0.238 | 0.183 | 0.946 | 0.950 | −0.059 | +51.4% | 78.4% | 92.2% |
| t+3 | 0.547 | 0.430 | 0.718 | 0.774 | −0.245 | +43.5% | 72.4% | 93.1% |
| t+6 | 0.904 | 0.753 | 0.233 | 0.495 | −0.528 | +29.0% | 62.9% | 84.5% |

All models and horizons, including MAE, explained variance and drought detection (POD/FAR), are in [`reports/RESULTS.md`](reports/RESULTS.md).

### Drought detection

The test decade contains only **4 drought months** (SPEI-12 < −1), too few for meaningful detection scores. On the CV backtest (71–78 drought months), the new Ridge detects drought with:

| Horizon | POD (hit rate) | FAR (false-alarm ratio) | Persistence POD / FAR |
|---|---|---|---|
| t+1 | 0.83 | 0.10 | 0.73 / 0.27 |
| t+3 | 0.41 | 0.33 | 0.45 / 0.56 |
| t+6 | 0.21 | 0.46 | 0.27 / 0.75 |

Beyond 1 month, onset of drought is mostly missed. The model is a persistence-plus-correction tool, not a drought-onset predictor.

The original README's "98% hit rate" counted agreement on drought *and* non-drought months. In a decade with 4 drought months, a model that never predicts drought would also score about 97%, so that number is not used here.

---

## Error by season and by regime

**By meteorological season of the target month, test window, RMSE:**

| Horizon | Model | DJF | MAM | JJA | SON |
|---|---|---|---|---|---|
| t+1 | Ridge (new) | 0.223 | 0.242 | 0.215 | 0.270 |
| t+1 | Previous Ridge | 0.223 | 0.224 | 0.195 | 0.267 |
| t+1 | Persistence | 0.323 | 0.388 | 0.276 | 0.368 |
| t+3 | Ridge (new) | 0.584 | 0.559 | 0.433 | 0.601 |
| t+3 | Previous Ridge | 0.529 | 0.509 | 0.369 | 0.507 |
| t+6 | Ridge (new) | 1.005 | 0.872 | 0.865 | 0.863 |
| t+6 | Previous Ridge | 0.760 | 0.745 | 0.731 | 0.555 |
| t+6 | Persistence | 1.129 | 1.260 | 1.131 | 0.720 |

Summer (JJA) targets are easiest at 1 and 3 months, because little rain falls in summer and the 12-month sum barely moves. Autumn and winter targets (SON/DJF), the rainy season, are hardest for the new model. At 6 months it is biased low in every season (bias −0.40 to −0.65), where the previous model's bias was −0.11 to −0.21.

**By region:** not possible. The dataset is a single 0.5° grid cell, so no regional breakdown exists. The closest honest substitute is a breakdown **by observed drought regime**. On the CV backtest, in drought months, the new Ridge's RMSE is 0.213 / 0.542 / 0.932 (persistence 0.363 / 0.735 / 1.195), with a positive bias: it under-forecasts how dry it gets. Full regime tables are in `reports/RESULTS.md`.

---

## Explainability

SHAP values are computed exactly: XGBoost's native TreeSHAP (`pred_contribs=True`) and linear SHAP (coefficient × standardised value) for Ridge. Mean |SHAP| on the test window for XGBoost:

| Horizon | Top features |
|---|---|
| t+1 | Current SPEI-11 (0.310), current SPEI-12 (0.191), SPEI-12 12-month mean (0.068) |
| t+3 | Current SPEI-9 (0.189), SPEI-3 − SPEI-12 (0.080), SPEI-12 12-month mean (0.074) |
| t+6 | SPEI-3 − SPEI-12 (0.168), SPEI-12 6-month mean (0.121), SPEI-12 6 months ago (0.100) |

This is what the arithmetic of SPEI predicts. The best single predictor of SPEI-12 at *t*+h is SPEI-(12−h) at *t*, the part of the future window already observed (SPEI-11 for h=1, SPEI-9 for h=3). The dashboard shows per-forecast contributions for the live forecast.

---

## Uncertainty

Prediction intervals are **split-conformal**: the 80% and 95% radii are quantiles of absolute out-of-fold CV residuals. The drought probability is the share of those residuals that, added to the current forecast, would put SPEI-12 at or below −1.

| Horizon | 80% PI coverage (test) | 95% PI coverage (test) | 80% width |
|---|---|---|---|
| t+1 | 78.4% | 92.2% | 0.57 |
| t+3 | 72.4% | 93.1% | 1.18 |
| t+6 | 62.9% | 84.5% | 1.77 |

- **Before:** the original bootstrap intervals covered 44.8% of outcomes against a nominal 90%, as reported in the notebook.
- **After:** coverage is close to nominal at 1 month. At 6 months it under-covers (63% vs 80%), because the model is biased in the wet test decade and conformal intervals assume errors behave like the past.

---

## Limitations

- **One grid cell.** No spatial or regional information. The forecast does not describe all of Algeria.
- **No exogenous predictors.** There are no seasonal rainfall forecasts, NAO/ENSO indices, soil moisture or NDVI. Most skill at 1 month is arithmetic: 11 of 12 months are already known.
- **The test decade is unusual.** 2016–2025 was the wettest period in the record, with only 4 drought months. Test-window detection scores are therefore uninformative, and model rankings differ between the test window and the long backtest.
- **SPEI calibration.** The data provider standardises SPEI over the full record, so a small amount of future distributional information is baked into historic values. This affects all models equally.
- **Data gaps.** SPEI-1 for Nov 2025 is imputed (R² ≈ 0.92), and SPEI-48 is unused.
- **Not an operational warning system.** For decisions, use the national meteorological service.

---

## Web dashboard

`web/` is a dependency-free static site: Vite + TypeScript with hand-built SVG charts. It renders:

- current conditions and the 1/3/6-month forecast, with 80% and 95% intervals, a drought probability and a risk level (Low / Watch / Elevated / High)
- a forecast fan chart, and "why the model expects this" (top SHAP contributions)
- 76-year history at five accumulation windows with 10 y / 30 y / all ranges, and a month × year drought calendar
- backtests, an all-model metrics table (including the previous model), error by season, and feature importance
- a plain-language method and limitations section

**Accessibility:** semantic landmarks and a skip link; every chart has a title, a description and a data-table alternative; chart tooltips work with the keyboard (arrow keys, Home/End); horizon tabs use ARIA tabs; risk is shown with icon + label, never colour alone; dark mode and reduced motion are supported.

**Lighthouse** (local production build via `vite preview`, Lighthouse 13.6.0, Chrome headless):

| | Performance | Accessibility | Best Practices | SEO |
|---|---|---|---|---|
| Mobile | 100 | 100 | 100 | 100 |
| Desktop | 100 | 100 | 100 | 100 |

The hero forecast cards are pre-rendered into `index.html` at build time by a small Vite plugin, so there is no layout shift (CLS 0) and the content is visible before JavaScript runs.

**Deploy:** `netlify.toml` builds `web/` with Node 22 (`npm ci && npm run build`, publish `web/dist`). The forecast data is committed, so Netlify needs no Python. To refresh the forecast with new data, replace the CSV, re-run `python -m pipeline.train`, and commit.

---

## References

1. Vicente-Serrano, S.M., Beguería, S., López-Moreno, J.I. (2010). A multiscalar drought index sensitive to global warming: the SPEI. *J. Climate* 23(7), 1696–1718.
2. Chen, T., Guestrin, C. (2016). XGBoost. *KDD*, 785–794.
3. Lundberg, S., Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS* 30.
4. Vovk, V., Gammerman, A., Shafer, G. (2005). *Algorithmic Learning in a Random World* (conformal prediction).
5. Künsch, H.R. (1989). The jackknife and the bootstrap for general stationary observations. *Ann. Statist.* 17(3), 1217–1241.
