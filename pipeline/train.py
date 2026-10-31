"""End-to-end training, evaluation and export.

    python -m pipeline.train --config config.yaml

Protocol (per horizon h):
  1. Development rows = origins whose target month <= split.dev_end.
  2. Hyper-parameters (Ridge alpha, XGBoost via Optuna TPE) are tuned on an
     expanding-window TimeSeriesSplit over the development rows with gap = h,
     so no training label overlaps a validation month.
  3. The model used for the forecast is chosen by CV RMSE only. The test
     window (origins 2016-01 .. 2025-08) is scored once, after selection.
  4. Prediction intervals are split-conformal radii computed from out-of-fold
     CV residuals, and their coverage is checked on the test window.
  5. For the live forecast the selected model is refit on every row whose
     target is already observed and applied at the latest month.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.model_selection import TimeSeriesSplit

from .data import fill_spei1_gaps, load_spei
from .evaluate import (adaptive_conformal, block_bootstrap_rmse, select_gamma, breakdown, conformal_quantile, coverage,
                       regression_metrics)
from .features import add_trend, build_features, build_targets, regime_of, season_of, split_rows
from .legacy import LegacyRidge, legacy_frame, legacy_ridge
from .models import ANCHOR, BASELINES, Blend, RidgeModel, XGBModel

ROOT = Path(__file__).resolve().parent.parent
AUTHOR = "Sharahbill Mohamed Abdi"


def set_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def make_folds(n: int, h: int, cfg: dict):
    c = cfg["cv"]
    test_size = (n - c["min_train_months"] - h) // c["n_splits"]
    return list(TimeSeriesSplit(n_splits=c["n_splits"], test_size=test_size, gap=h).split(np.arange(n)))


def cv_predict(factory, X: pd.DataFrame, y: pd.Series, folds) -> pd.Series:
    parts = []
    for tr, va in folds:
        m = factory().fit(X.iloc[tr], y.iloc[tr])
        parts.append(pd.Series(m.predict(X.iloc[va]), index=X.index[va]))
    return pd.concat(parts)


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def cv_rmse(factory, X, y, folds) -> float:
    oof = cv_predict(factory, X, y, folds)
    return rmse(y.loc[oof.index], oof)


def tune_xgb(X, y, folds, h, cfg) -> dict:
    space, seed = cfg["search"]["xgboost"], cfg["seed"]

    def objective(trial: optuna.Trial) -> float:
        p = {
            "n_estimators": trial.suggest_int("n_estimators", *space["n_estimators"], step=50),
            "max_depth": trial.suggest_int("max_depth", *space["max_depth"]),
            "learning_rate": trial.suggest_float("learning_rate", *space["learning_rate"], log=True),
            "subsample": trial.suggest_float("subsample", *space["subsample"]),
            "colsample_bytree": trial.suggest_float("colsample_bytree", *space["colsample_bytree"]),
            "min_child_weight": trial.suggest_int("min_child_weight", *space["min_child_weight"]),
            "reg_lambda": trial.suggest_float("reg_lambda", *space["reg_lambda"], log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", *space["reg_alpha"], log=True),
        }
        return cv_rmse(lambda: XGBModel(h, seed=seed, **p), X, y, folds)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=cfg["search"]["n_trials"], n_jobs=1)
    return study.best_params


FEATURE_LABELS = {
    "spei_12": "Current SPEI-12",
    "month_sin": "Season (sine)",
    "month_cos": "Season (cosine)",
    "drought_run_months": "Months in current drought",
    "months_since_drought": "Months since last drought",
    "spei12_change_1": "SPEI-12 change, last month",
    "spei12_change_3": "SPEI-12 change, last 3 months",
    "spei3_minus_12": "SPEI-3 minus SPEI-12",
    "spei12_minus_24": "SPEI-12 minus SPEI-24",
    "year": "Year (trend)",
    "decade": "Decade (trend)",
    "month_since_start": "Months since 1950 (trend)",
}
TREND = " +trend"  # name suffix for models trained with the calendar-trend features


def feature_label(name: str) -> str:
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    parts = name.split("_")
    if parts[0] == "spei" and len(parts) == 2:
        return f"Current SPEI-{parts[1]}"
    if "lag" in parts:
        return f"SPEI-{parts[0][4:]} {parts[-1]} month(s) ago"
    if parts[1] in ("mean", "std"):
        stat = "mean" if parts[1] == "mean" else "variability"
        return f"SPEI-{parts[0][4:]} {parts[2]}-month {stat}"
    return name


def spei_class(v: float) -> str:
    if v <= -2: return "Extreme drought"
    if v <= -1.5: return "Severe drought"
    if v <= -1: return "Moderate drought"
    if v < 1: return "Near normal"
    if v < 1.5: return "Moderately wet"
    if v < 2: return "Very wet"
    return "Extremely wet"


def r(v, nd=4):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return None
    return round(float(v), nd)


def main(config_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())
    seed = cfg["seed"]
    set_seeds(seed)
    out_models = ROOT / cfg["output"]["models_dir"]
    out_reports = ROOT / cfg["output"]["reports_dir"]
    out_models.mkdir(parents=True, exist_ok=True)
    out_reports.mkdir(parents=True, exist_ok=True)

    raw = load_spei(ROOT / cfg["data"]["path"], cfg["data"]["date_column"], cfg["data"]["date_format"])
    df, imputed = fill_spei1_gaps(raw)
    print(f"imputed SPEI_1 for: {imputed or 'none'}")
    X_all, Y = build_features(df, cfg), build_targets(df, cfg)
    # Feature-set variants: "" = base features, TREND = base + calendar trend.
    variants = ["" if not v else TREND for v in cfg["features"]["trend_variants"]]
    Xsets = {"": X_all, TREND: add_trend(X_all, df.index[0])}
    levels = cfg["intervals"]["levels"]
    thr = cfg["features"]["drought_threshold"]
    ada = cfg["intervals"].get("adaptive", {})
    test_start = pd.Timestamp(cfg["split"]["test_start"])

    test_rows, cv_rows, test_frames, cv_frames, interval_rows = [], [], [], [], []
    params_out, importance, horizon_out, backtest, bootstrap = {}, {}, [], {}, {}
    gamma_choice = {}

    # The original notebook's Ridge: exact original protocol for the test
    # window, and the same model re-evaluated under the new CV.
    LEGACY = "Previous Ridge"
    legacy = legacy_ridge(raw)
    lframe, lfeats = legacy_frame(raw)

    for h in cfg["horizons"]:
        y = Y[h]
        dev, test = split_rows(X_all, y, h, cfg)
        Xd, yd, Xt, yt = X_all.loc[dev], y.loc[dev], X_all.loc[test], y.loc[test]
        folds = make_folds(len(dev), h, cfg)
        print(f"[h={h}] dev {dev[0]:%Y-%m}..{dev[-1]:%Y-%m} ({len(dev)}), "
              f"test {test[0]:%Y-%m}..{test[-1]:%Y-%m} ({len(test)}), folds={len(folds)}")

        # Baselines only use SPEI-12 and the calendar, so they need no variant.
        factories = {B.name: (lambda B=B: B(h)) for B in BASELINES}
        xkey = {name: "" for name in factories}
        params_out[h] = {}
        for v in variants:
            Xv = Xsets[v].loc[dev]
            alpha = min(cfg["search"]["ridge_alphas"],
                        key=lambda a: cv_rmse(lambda: RidgeModel(h, alpha=a), Xv, yd, folds))
            xgb_params = tune_xgb(Xv, yd, folds, h, cfg)
            params_out[h]["trend" if v else "base"] = {"ridge_alpha": alpha, "xgboost": xgb_params}
            print(f"[h={h}{v}] ridge alpha={alpha}; xgb {xgb_params}")
            factories[f"Ridge{v}"] = lambda a=alpha: RidgeModel(h, alpha=a)
            factories[f"XGBoost{v}"] = lambda p=xgb_params: XGBModel(h, seed=seed, **p)
            factories[f"Ridge + XGBoost{v}"] = lambda a=alpha, p=xgb_params: Blend(h, {"alpha": a}, p, seed=seed)
            xkey.update({f"Ridge{v}": v, f"XGBoost{v}": v, f"Ridge + XGBoost{v}": v})

        per_model = {}
        for name, factory in factories.items():
            Xdv, Xtv = Xsets[xkey[name]].loc[dev], Xsets[xkey[name]].loc[test]
            oof = cv_predict(factory, Xdv, yd, folds)
            y_oof = yd.loc[oof.index]
            res_oof = (y_oof - oof).to_numpy()
            model = factory().fit(Xdv, yd)
            pred = model.predict(Xtv)
            radii = {lv: conformal_quantile(np.abs(res_oof), lv) for lv in levels}
            per_model[name] = dict(oof=oof, res_oof=res_oof, pred=pred, radii=radii, model=model,
                                   cv_rmse=rmse(y_oof, oof), xkey=xkey[name])

            mt = regression_metrics(yt, pred, Xt[ANCHOR], thr)
            mt.update({f"coverage_{int(lv*100)}": coverage(yt, pred - q, pred + q) for lv, q in radii.items()})
            mt.update({f"interval_width_{int(lv*100)}": 2 * q for lv, q in radii.items()})
            test_rows.append({"model": name, "horizon": h, **mt})
            cv_rows.append({"model": name, "horizon": h,
                            **regression_metrics(y_oof, oof, Xd.loc[oof.index, ANCHOR], thr)})
            test_frames.append(pd.DataFrame({"date": test, "model": name, "horizon": h, "y": yt.to_numpy(),
                                             "pred": pred, "persistence": Xt[ANCHOR].to_numpy()}))
            cv_frames.append(pd.DataFrame({"date": oof.index, "model": name, "horizon": h, "y": y_oof.to_numpy(),
                                           "pred": oof.to_numpy(), "persistence": Xd.loc[oof.index, ANCHOR].to_numpy()}))

        selected = min(per_model, key=lambda k: per_model[k]["cv_rmse"])
        print(f"[h={h}] CV RMSE " + ", ".join(f"{k}={v['cv_rmse']:.4f}" for k, v in per_model.items())
              + f" -> selected {selected}")

        # Previous Ridge: CV on identical folds (alpha tuned the same way),
        # test predictions from the exact original protocol.
        Xl = lframe[lfeats].reindex(dev)
        la = min(cfg["search"]["ridge_alphas"], key=lambda a: cv_rmse(lambda: LegacyRidge(h, alpha=a), Xl, yd, folds))
        loof = cv_predict(lambda: LegacyRidge(h, alpha=la), Xl, yd, folds)
        legacy_cv = rmse(yd.loc[loof.index], loof)
        print(f"[h={h}] {LEGACY} CV RMSE={legacy_cv:.4f} (alpha={la})")
        if legacy_cv < per_model[selected]["cv_rmse"]:
            print(f"[h={h}] WARNING: {LEGACY} beats the selected model in CV")
        cv_rows.append({"model": LEGACY, "horizon": h,
                        **regression_metrics(yd.loc[loof.index], loof, Xd.loc[loof.index, ANCHOR], thr)})
        lpred = legacy[h].reindex(test)
        assert lpred.notna().all()
        test_rows.append({"model": LEGACY, "horizon": h, **regression_metrics(yt, lpred, Xt[ANCHOR], thr)})
        test_frames.append(pd.DataFrame({"date": test, "model": LEGACY, "horizon": h, "y": yt.to_numpy(),
                                         "pred": lpred.to_numpy(), "persistence": Xt[ANCHOR].to_numpy()}))
        boot_preds = {k: v["pred"] for k, v in per_model.items()
                      if k not in ("Climatology", "Damped persistence")}
        boot_preds[LEGACY] = lpred.to_numpy()
        bootstrap[h] = block_bootstrap_rmse(yt, boot_preds, seed=seed)

        # Explainability on the test window, for the selected feature variant.
        sv = per_model[selected]["xkey"]
        Xs = Xsets[sv]
        feats = list(Xs.columns)
        importance[h] = {"variant": sv}
        for base in ("XGBoost", "Ridge"):
            name = f"{base}{sv}"
            contrib = per_model[name]["model"].contributions(Xs.loc[test])
            imp = pd.Series(np.abs(contrib).mean(axis=0), index=feats).sort_values(ascending=False)
            importance[h][base] = imp
            imp.rename("mean_abs_shap").to_csv(out_reports / f"importance_{base.lower()}_h{h}.csv",
                                               index_label="feature")

        # Production refit on every row whose target is observed.
        full = Xs.index[Xs.notna().all(axis=1) & y.notna()]
        prod = factories[selected]().fit(Xs.loc[full], y.loc[full])
        for base in ("XGBoost", "Ridge"):
            name = f"{base}{sv}"
            m = prod if name == selected else factories[name]().fit(Xs.loc[full], y.loc[full])
            if base == "XGBoost":
                m.model_.save_model(out_models / f"xgboost_h{h}.json")
            else:
                joblib.dump(m, out_models / f"ridge_h{h}.joblib")
        joblib.dump(prod, out_models / f"selected_h{h}.joblib")

        origin = Xs.index[Xs.notna().all(axis=1)][-1]
        if origin != df.index[-1]:
            raise ValueError(f"latest complete feature row is {origin:%Y-%m}, data ends {df.index[-1]:%Y-%m}")
        x0 = Xs.loc[[origin]]
        p0 = float(prod.predict(x0)[0])
        sel = per_model[selected]

        # Intervals. Static split-conformal radii come from out-of-fold CV
        # residuals. For adaptive horizons the radius is updated month by month
        # from outcomes already observed (dev-period model, origins from the
        # test start up to the last origin whose outcome is known).
        cal = np.abs(sel["res_oof"])
        seq_idx = Xs.index[(Xs.index >= test_start) & Xs.notna().all(axis=1) & y.notna()]
        assert (seq_idx[: len(test)] == test).all()
        seq_err = np.abs(y.loc[seq_idx].to_numpy() - sel["model"].predict(Xs.loc[seq_idx]))
        adaptive = h in ada.get("horizons", [])
        gamma = ada.get("gamma")
        # gamma_grid: one list for every horizon, or a {horizon: list} mapping.
        grid = ada.get("gamma_grid", [])
        grid = grid.get(h, []) if isinstance(grid, dict) else grid
        if adaptive and gamma == "auto":
            # Per-fold OOF errors of the selected model (folds are contiguous, in order).
            sizes = [len(va) for _, va in folds]
            fold_err = np.split(np.abs(sel["res_oof"]), np.cumsum(sizes)[:-1])
            gamma, gtable = select_gamma(fold_err, levels[0], grid, delay=h)
            gamma_choice[h] = {"gamma": gamma, "candidates": gtable}
            print(f"[h={h}] adaptive gamma chosen on 1973-2015 backtest: {gamma}")
        band, live_r = {}, {}
        for lv in levels:
            q = sel["radii"][lv]
            interval_rows.append({"horizon": h, "model": selected, "method": "static split-conformal", "gamma": None,
                                  "level": lv, "coverage": coverage(yt, sel["pred"] - q, sel["pred"] + q),
                                  "mean_width": 2 * q, "live_width": 2 * q})
            for g in grid if adaptive else []:
                rs, rl = adaptive_conformal(cal, seq_err, lv, g, delay=h)
                rt = rs[: len(test)]
                interval_rows.append({"horizon": h, "model": selected, "method": "adaptive conformal", "gamma": g,
                                      "level": lv, "coverage": coverage(yt, sel["pred"] - rt, sel["pred"] + rt),
                                      "mean_width": float(2 * rt.mean()), "live_width": 2 * rl})
            if adaptive:
                rs, rl = adaptive_conformal(cal, seq_err, lv, gamma, delay=h)
                band[lv], live_r[lv] = rs[: len(test)], rl
            else:
                band[lv], live_r[lv] = np.full(len(test), q), q
        # Drought probability: OOF residual distribution, rescaled to the live
        # 80% width when the interval is adaptive.
        scale = live_r[levels[0]] / sel["radii"][levels[0]] if sel["radii"][levels[0]] > 0 else 1.0
        p_drought = float(np.mean(p0 + scale * sel["res_oof"] < thr))
        drivers = []
        if hasattr(prod, "contributions"):
            c0 = pd.Series(prod.contributions(x0)[0], index=feats)
            for f in c0.abs().sort_values(ascending=False).index[:5]:
                drivers.append({"feature": f, "label": feature_label(f), "value": r(x0[f].iloc[0], 3),
                                "contribution": r(c0[f], 3)})
        horizon_out.append({
            "horizon": h,
            "target_date": f"{origin + pd.DateOffset(months=h):%Y-%m}",
            "model": selected,
            "prediction": r(p0, 3),
            "class": spei_class(p0),
            "intervals": {str(int(lv * 100)): [r(p0 - q, 3), r(p0 + q, 3)] for lv, q in live_r.items()},
            "interval_method": "adaptive conformal" if adaptive else "split conformal",
            "p_drought": r(p_drought, 3),
            "drivers": drivers,
        })
        backtest[h] = {
            "dates": [f"{d:%Y-%m}" for d in test + pd.DateOffset(months=h)],
            "observed": [r(v, 3) for v in yt],
            "selected": selected,
            "predictions": {**{k: [r(v, 3) for v in v_["pred"]] for k, v_ in per_model.items()},
                            LEGACY: [r(v, 3) for v in lpred]},
            "radius": {str(int(lv * 100)): [r(v, 4) for v in band[lv]] for lv in levels},
            "interval_method": "adaptive conformal" if adaptive else "split conformal",
        }

    # ---- Reports -------------------------------------------------------
    test_df, cv_df = pd.DataFrame(test_rows), pd.DataFrame(cv_rows)
    test_df.to_csv(out_reports / "metrics_test.csv", index=False)
    cv_df.to_csv(out_reports / "metrics_cv.csv", index=False)

    tf = pd.concat(test_frames)
    tf["season"] = season_of([(d + pd.DateOffset(months=int(h))).month for d, h in zip(tf["date"], tf["horizon"])])
    tf["regime"] = regime_of(tf["y"], thr)
    tf.to_csv(out_reports / "test_predictions.csv", index=False)
    cvf = pd.concat(cv_frames)
    cvf["season"] = season_of([(d + pd.DateOffset(months=int(h))).month for d, h in zip(cvf["date"], cvf["horizon"])])
    cvf["regime"] = regime_of(cvf["y"], thr)
    by_season, by_regime = breakdown(tf, "season"), breakdown(tf, "regime")
    cv_by_regime = breakdown(cvf, "regime")
    by_season.to_csv(out_reports / "breakdown_season_test.csv", index=False)
    by_regime.to_csv(out_reports / "breakdown_regime_test.csv", index=False)
    cv_by_regime.to_csv(out_reports / "breakdown_regime_cv.csv", index=False)

    # Verify the original notebook's Ridge numbers on the identical window.
    before_csv = pd.read_csv(ROOT / "metrics_comparison.csv")
    legacy_rows = []
    for h, pred in legacy.items():
        yt = Y.loc[pred.index, h]
        m = regression_metrics(yt, pred, X_all.loc[pred.index, ANCHOR], thr)
        reported = before_csv[(before_csv.Model == "Ridge") & (before_csv.Horizon == f"t+{h}")]["RMSE"]
        legacy_rows.append({"horizon": h, "n": m["n"], "rmse": m["rmse"], "mae": m["mae"], "r2": m["r2"],
                            "explained_variance": m["explained_variance"],
                            "skill_vs_persistence": m["skill_vs_persistence"],
                            "reported_rmse": float(reported.iloc[0]) if len(reported) else None})
    legacy_df = pd.DataFrame(legacy_rows)
    legacy_df.to_csv(out_reports / "legacy_ridge_reproduction.csv", index=False)

    interval_df = pd.DataFrame(interval_rows)
    interval_df.to_csv(out_reports / "intervals_test.csv", index=False)
    (out_reports / "best_params.json").write_text(json.dumps(params_out, indent=2))
    (out_reports / "bootstrap_test.json").write_text(json.dumps(bootstrap, indent=2))
    metrics = {
        "config": cfg,
        "test": test_df.to_dict(orient="records"),
        "cv": cv_df.to_dict(orient="records"),
        "legacy_ridge": legacy_df.to_dict(orient="records"),
        "selected": {h["horizon"]: h["model"] for h in horizon_out},
        "best_params": params_out,
        "bootstrap_test": bootstrap,
        "intervals_test": interval_df.to_dict(orient="records"),
        "adaptive_gamma": gamma_choice,
    }
    (out_reports / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    write_results_md(out_reports, test_df, cv_df, legacy_df, by_season, by_regime, cv_by_regime, importance,
                     {h["horizon"]: h["model"] for h in horizon_out}, bootstrap, interval_df, cfg, gamma_choice)

    # ---- Web payload ---------------------------------------------------
    keep = ["SPEI_1", "SPEI_3", "SPEI_6", "SPEI_12", "SPEI_24"]
    last = df.index[-1]
    web = {
        "meta": {
            "author": AUTHOR,
            "location": cfg["data"]["location"],
            "data_start": f"{df.index[0]:%Y-%m}",
            "data_end": f"{last:%Y-%m}",
            "target": "SPEI-12",
            "test_window": [cfg["split"]["test_start"][:7], cfg["split"]["test_end"][:7]],
            "dev_end": cfg["split"]["dev_end"][:7],
            "n_trials": cfg["search"]["n_trials"],
            "cv_folds": cfg["cv"]["n_splits"],
            "n_features": X_all.shape[1],
            "trend_features": list(Xsets[TREND].columns[X_all.shape[1]:]),
            "imputed_spei1": imputed,
        },
        "observed": {"dates": [f"{d:%Y-%m}" for d in raw.index],
                     **{c.replace("_", "").lower(): [r(v, 3) for v in raw[c]] for c in keep}},
        "current": {"date": f"{last:%Y-%m}", "spei12": r(df["SPEI_12"].iloc[-1], 3),
                    "class": spei_class(df["SPEI_12"].iloc[-1])},
        "forecast": horizon_out,
        "metrics": [{k: (r(v, 6) if isinstance(v, float) else v) for k, v in row.items()} for row in test_rows],
        "cv_metrics": [{k: (r(v, 6) if isinstance(v, float) else v) for k, v in row.items()} for row in cv_rows],
        "legacy": [{k: r(v, 6) if isinstance(v, float) else v for k, v in row.items()} for row in legacy_rows],
        "season": [{k: (r(v, 6) if isinstance(v, float) else v) for k, v in row.items()}
                   for row in by_season.to_dict(orient="records")],
        "regime_cv": [{k: (r(v, 6) if isinstance(v, float) else v) for k, v in row.items()}
                      for row in cv_by_regime.to_dict(orient="records")],
        "importance": {str(h): [{"feature": f, "label": feature_label(f), "value": r(v)}
                                for f, v in importance[h]["XGBoost"].head(10).items()]
                       for h in cfg["horizons"]},
        "backtest": {str(h): v for h, v in backtest.items()},
        "bootstrap": {str(h): {"rmse_90ci": {k: [r(a, 6), r(b, 6)] for k, (a, b) in v["rmse_90ci"].items()},
                               "p_lower": {k: r(p, 6) for k, p in v["p_lower"].items()}}
                      for h, v in bootstrap.items()},
    }
    web_path = ROOT / cfg["output"]["web_data"]
    web_path.parent.mkdir(parents=True, exist_ok=True)
    web_path.write_text(json.dumps(web, separators=(",", ":")))
    print(f"wrote {web_path.relative_to(ROOT)} and reports/ + models/")


def _fmt(v, nd=3):
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def write_results_md(out, test_df, cv_df, legacy_df, by_season, by_regime, cv_by_regime, importance, selected,
                     bootstrap, interval_df, cfg, gamma_choice):
    L = ["# Results (auto-generated by `python -m pipeline.train`)", ""]
    L += ["## Selected model per horizon (lowest 1973–2015 backtest RMSE)", ""]
    for h, name in selected.items():
        L.append(f"- t+{h}: **{name}**")
    L += ["", "## Calendar-trend features: with vs without (`features.trend_variants`)", "",
          "Same model family, same folds. The backtest decides; the test years are reported only.", "",
          "| Horizon | Model | Backtest RMSE without | Backtest RMSE with trend | Test RMSE without | Test RMSE with trend |",
          "|---|---|---|---|---|---|"]
    for h in sorted(test_df.horizon.unique()):
        for base in ("Ridge", "XGBoost", "Ridge + XGBoost"):
            def get(df, name):
                v = df[(df.horizon == h) & (df.model == name)]["rmse"]
                return float(v.iloc[0]) if len(v) else None
            cb, ct = get(cv_df, base), get(cv_df, base + TREND)
            tb, tt = get(test_df, base), get(test_df, base + TREND)
            mark = lambda a, b: (f"**{_fmt(a)}**" if a is not None and b is not None and a < b else _fmt(a))
            L.append(f"| t+{h} | {base} | {mark(cb, ct)} | {mark(ct, cb)} | {mark(tb, tt)} | {mark(tt, tb)} |")
    a = cfg["intervals"].get("adaptive", {})
    used = {h: g["gamma"] for h, g in gamma_choice.items()}
    L += ["", "## Adaptive-interval step size (gamma), chosen on the 1973–2015 backtest only", "",
          "Each CV fold is replayed as a live period calibrated on earlier folds. Criterion: mean |rolling",
          "60-month coverage − 80%| (lower is better); ties within 0.001 go to the narrower interval.", "",
          "| Horizon | gamma | Backtest coverage | Rolling-coverage deviation | Backtest mean 80% width |", "|---|---|---|---|---|"]
    for h, g in gamma_choice.items():
        for c in g["candidates"]:
            star = " ★" if c["gamma"] == g["gamma"] else ""
            L.append(f"| t+{h} | {c['gamma']:g}{star} | {c['dev_coverage'] * 100:.1f}% | {c['dev_rolling_mad']:.3f} | "
                     f"{c['dev_mean_width']:.2f} |")
    L += ["", "## Prediction-interval coverage on the test window", "",
          f"Adaptive conformal (Gibbs & Candès 2021) is used for horizons {a.get('horizons')} with the backtest-chosen",
          f"gamma ({used}). The other gammas are shown for transparency only; they were not used to choose.", "",
          "| Horizon | Method | gamma | Level | Coverage | Mean width | Width for the live forecast |",
          "|---|---|---|---|---|---|---|"]
    for _, x in interval_df.iterrows():
        g = "—" if x.gamma is None or not np.isfinite(x.gamma) else f"{x.gamma:g}"
        mark = " (used)" if x.method == "adaptive conformal" and g != "—" and float(g) == used.get(x.horizon) else ""
        L.append(f"| t+{x.horizon} | {x.method}{mark} | {g} | {int(x.level * 100)}% | {x.coverage * 100:.1f}% | "
                 f"{x.mean_width:.2f} | {x.live_width:.2f} |")
    L.append("")
    L += ["## Test window (origins 2016-01 → 2025-08, 116 forecasts per horizon)", "",
          "| Horizon | Model | RMSE | MAE | R² | Expl. var. | Bias | Skill vs persistence | 80% PI cover | 95% PI cover |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for _, x in test_df.iterrows():
        star = " ★" if selected[x.horizon] == x.model else ""
        cov = lambda v: "—" if v is None or not np.isfinite(v) else f"{v * 100:.1f}%"
        L.append(f"| t+{x.horizon} | {x.model}{star} | {_fmt(x.rmse)} | {_fmt(x.mae)} | {_fmt(x.r2)} | "
                 f"{_fmt(x.explained_variance)} | {_fmt(x.bias)} | {_fmt(x.skill_vs_persistence)} | "
                 f"{cov(x.coverage_80)} | {cov(x.coverage_95)} |")
    L += ["", "★ = model selected for the live forecast by cross-validated RMSE (test set not used for selection).",
          "PI coverage in this table uses static split-conformal intervals; see the interval table above for the",
          "adaptive intervals actually used.", ""]
    L += ["## Time-series cross-validation (out-of-fold, development period ≤ 2015)", "",
          "| Horizon | Model | n | RMSE | MAE | R² | Skill vs persistence | Drought months | POD | FAR |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for _, x in cv_df.iterrows():
        L.append(f"| t+{x.horizon} | {x.model} | {x.n} | {_fmt(x.rmse)} | {_fmt(x.mae)} | {_fmt(x.r2)} | "
                 f"{_fmt(x.skill_vs_persistence)} | {x.drought_months} | {_fmt(x.pod, 2)} | {_fmt(x.far, 2)} |")
    L += ["", "## Is the test-window difference real? (moving-block bootstrap, 12-month blocks, 2000 resamples)", "",
          "| Horizon | Model | RMSE 90% interval | P(lower RMSE than Previous Ridge) | P(lower RMSE than Persistence) |",
          "|---|---|---|---|---|"]
    for h, b in bootstrap.items():
        for k, (lo, hi) in b["rmse_90ci"].items():
            pl = b["p_lower"].get(f"{k} < Previous Ridge")
            pp = b["p_lower"].get(f"{k} < Persistence")
            L.append(f"| t+{h} | {k} | {lo:.3f} – {hi:.3f} | {_fmt(pl, 2)} | {_fmt(pp, 2)} |")
    L += ["", "## Original notebook Ridge, re-run on the same window", "",
          "| Horizon | RMSE (re-run) | RMSE (metrics_comparison.csv) | R² | Skill vs persistence |", "|---|---|---|---|---|"]
    for _, x in legacy_df.iterrows():
        L.append(f"| t+{int(x.horizon)} | {_fmt(x.rmse)} | {_fmt(x.reported_rmse)} | {_fmt(x.r2)} | {_fmt(x.skill_vs_persistence)} |")
    for title, frame, key in [("Test error by season of target month", by_season, "season"),
                              ("Test error by observed regime", by_regime, "regime"),
                              ("CV error by observed regime (more drought months)", cv_by_regime, "regime")]:
        L += ["", f"## {title}", "", f"| Horizon | Model | {key} | n | RMSE | MAE | Bias |", "|---|---|---|---|---|---|---|"]
        keep = ["Persistence", "Previous Ridge"] + [b + v for b in ("Ridge", "XGBoost", "Ridge + XGBoost") for v in ("", TREND)]
        for _, x in frame[frame.model.isin(keep)].iterrows():
            L.append(f"| t+{x.horizon} | {x.model} | {x[key]} | {x.n} | {_fmt(x.rmse)} | {_fmt(x.mae)} | {_fmt(x.bias)} |")
    L += ["", "## Top features (mean |SHAP| on test window, XGBoost of the selected feature variant)", ""]
    for h, imps in importance.items():
        top = ", ".join(f"`{f}` {v:.3f}" for f, v in imps["XGBoost"].head(6).items())
        L.append(f"- t+{h} ({'with' if imps['variant'] else 'without'} trend features): {top}")
    (out / "RESULTS.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    main(ap.parse_args().config)
