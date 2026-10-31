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
from .evaluate import block_bootstrap_rmse, breakdown, conformal_quantile, coverage, regression_metrics
from .features import build_features, build_targets, regime_of, season_of, split_rows
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
}


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
    feats = list(X_all.columns)
    levels = cfg["intervals"]["levels"]
    thr = cfg["features"]["drought_threshold"]

    test_rows, cv_rows, test_frames, cv_frames = [], [], [], []
    params_out, importance, horizon_out, backtest, bootstrap = {}, {}, [], {}, {}

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

        alpha = min(cfg["search"]["ridge_alphas"],
                    key=lambda a: cv_rmse(lambda: RidgeModel(h, alpha=a), Xd, yd, folds))
        xgb_params = tune_xgb(Xd, yd, folds, h, cfg)
        params_out[h] = {"ridge_alpha": alpha, "xgboost": xgb_params}
        print(f"[h={h}] ridge alpha={alpha}; xgb {xgb_params}")

        factories = {B.name: (lambda B=B: B(h)) for B in BASELINES}
        factories["Ridge"] = lambda: RidgeModel(h, alpha=alpha)
        factories["XGBoost"] = lambda: XGBModel(h, seed=seed, **xgb_params)
        factories["Ridge + XGBoost"] = lambda: Blend(h, {"alpha": alpha}, xgb_params, seed=seed)

        per_model = {}
        for name, factory in factories.items():
            oof = cv_predict(factory, Xd, yd, folds)
            y_oof = yd.loc[oof.index]
            res_oof = (y_oof - oof).to_numpy()
            model = factory().fit(Xd, yd)
            pred = model.predict(Xt)
            radii = {lv: conformal_quantile(np.abs(res_oof), lv) for lv in levels}
            per_model[name] = dict(oof=oof, res_oof=res_oof, pred=pred, radii=radii, model=model,
                                   cv_rmse=rmse(y_oof, oof))

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
        boot_preds = {k: per_model[k]["pred"] for k in ("Persistence", "Ridge", "XGBoost", "Ridge + XGBoost")}
        boot_preds[LEGACY] = lpred.to_numpy()
        bootstrap[h] = block_bootstrap_rmse(yt, boot_preds, seed=seed)

        # Explainability on the test window.
        importance[h] = {}
        for name in ("XGBoost", "Ridge"):
            contrib = per_model[name]["model"].contributions(Xt)
            imp = pd.Series(np.abs(contrib).mean(axis=0), index=feats).sort_values(ascending=False)
            importance[h][name] = imp
            imp.rename("mean_abs_shap").to_csv(out_reports / f"importance_{name.lower()}_h{h}.csv",
                                               index_label="feature")

        # Production refit on every row whose target is observed.
        full = X_all.index[X_all.notna().all(axis=1) & y.notna()]
        prod = factories[selected]().fit(X_all.loc[full], y.loc[full])
        for name in ("XGBoost", "Ridge"):
            m = prod if name == selected else factories[name]().fit(X_all.loc[full], y.loc[full])
            if name == "XGBoost":
                m.model_.save_model(out_models / f"xgboost_h{h}.json")
            else:
                joblib.dump(m, out_models / f"ridge_h{h}.joblib")
        joblib.dump(prod, out_models / f"selected_h{h}.joblib")

        origin = X_all.index[X_all.notna().all(axis=1)][-1]
        if origin != df.index[-1]:
            raise ValueError(f"latest complete feature row is {origin:%Y-%m}, data ends {df.index[-1]:%Y-%m}")
        x0 = X_all.loc[[origin]]
        p0 = float(prod.predict(x0)[0])
        sel = per_model[selected]
        p_drought = float(np.mean(p0 + sel["res_oof"] < thr))
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
            "intervals": {str(int(lv * 100)): [r(p0 - q, 3), r(p0 + q, 3)] for lv, q in sel["radii"].items()},
            "p_drought": r(p_drought, 3),
            "drivers": drivers,
        })
        backtest[h] = {
            "dates": [f"{d:%Y-%m}" for d in test + pd.DateOffset(months=h)],
            "observed": [r(v, 3) for v in yt],
            "selected": selected,
            "predictions": {**{k: [r(v, 3) for v in v_["pred"]] for k, v_ in per_model.items()},
                            LEGACY: [r(v, 3) for v in lpred]},
            "radius": {str(int(lv * 100)): r(q, 4) for lv, q in sel["radii"].items()},
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
    }
    (out_reports / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    write_results_md(out_reports, test_df, cv_df, legacy_df, by_season, by_regime, cv_by_regime, importance,
                     {h["horizon"]: h["model"] for h in horizon_out}, bootstrap)

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
            "n_features": len(feats),
            "imputed_spei1": imputed,
        },
        "observed": {"dates": [f"{d:%Y-%m}" for d in raw.index],
                     **{c.replace("_", "").lower(): [r(v, 3) for v in raw[c]] for c in keep}},
        "current": {"date": f"{last:%Y-%m}", "spei12": r(df["SPEI_12"].iloc[-1], 3),
                    "class": spei_class(df["SPEI_12"].iloc[-1])},
        "forecast": horizon_out,
        "metrics": [{k: (r(v) if isinstance(v, float) else v) for k, v in row.items()} for row in test_rows],
        "cv_metrics": [{k: (r(v) if isinstance(v, float) else v) for k, v in row.items()} for row in cv_rows],
        "legacy": [{k: r(v) if isinstance(v, float) else v for k, v in row.items()} for row in legacy_rows],
        "season": [{k: (r(v) if isinstance(v, float) else v) for k, v in row.items()}
                   for row in by_season.to_dict(orient="records")],
        "regime_cv": [{k: (r(v) if isinstance(v, float) else v) for k, v in row.items()}
                      for row in cv_by_regime.to_dict(orient="records")],
        "importance": {str(h): [{"feature": f, "label": feature_label(f), "value": r(v)}
                                for f, v in importance[h]["XGBoost"].head(10).items()]
                       for h in cfg["horizons"]},
        "backtest": {str(h): v for h, v in backtest.items()},
        "bootstrap": {str(h): {"rmse_90ci": {k: [r(a, 3), r(b, 3)] for k, (a, b) in v["rmse_90ci"].items()},
                               "p_lower": {k: r(p, 3) for k, p in v["p_lower"].items()}}
                      for h, v in bootstrap.items()},
    }
    web_path = ROOT / cfg["output"]["web_data"]
    web_path.parent.mkdir(parents=True, exist_ok=True)
    web_path.write_text(json.dumps(web, separators=(",", ":")))
    print(f"wrote {web_path.relative_to(ROOT)} and reports/ + models/")


def _fmt(v, nd=3):
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def write_results_md(out, test_df, cv_df, legacy_df, by_season, by_regime, cv_by_regime, importance, selected,
                     bootstrap):
    L = ["# Results (auto-generated by `python -m pipeline.train`)", ""]
    L += ["## Test window (origins 2016-01 → 2025-08, 116 forecasts per horizon)", "",
          "| Horizon | Model | RMSE | MAE | R² | Expl. var. | Bias | Skill vs persistence | 80% PI cover | 95% PI cover |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for _, x in test_df.iterrows():
        star = " ★" if selected[x.horizon] == x.model else ""
        L.append(f"| t+{x.horizon} | {x.model}{star} | {_fmt(x.rmse)} | {_fmt(x.mae)} | {_fmt(x.r2)} | "
                 f"{_fmt(x.explained_variance)} | {_fmt(x.bias)} | {_fmt(x.skill_vs_persistence)} | "
                 f"{_fmt(x.coverage_80 * 100, 1)}% | {_fmt(x.coverage_95 * 100, 1)}% |")
    L += ["", "★ = model selected for the live forecast by cross-validated RMSE (test set not used for selection).", ""]
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
        for _, x in frame[frame.model.isin(["Persistence", "Ridge", "XGBoost", "Ridge + XGBoost", "Previous Ridge"])].iterrows():
            L.append(f"| t+{x.horizon} | {x.model} | {x[key]} | {x.n} | {_fmt(x.rmse)} | {_fmt(x.mae)} | {_fmt(x.bias)} |")
    L += ["", "## Top features (mean |SHAP| on test window, XGBoost)", ""]
    for h, imps in importance.items():
        top = ", ".join(f"`{f}` {v:.3f}" for f, v in imps["XGBoost"].head(6).items())
        L.append(f"- t+{h}: {top}")
    (out / "RESULTS.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    main(ap.parse_args().config)
