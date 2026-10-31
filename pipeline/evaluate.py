"""Metrics, breakdowns and conformal prediction intervals."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y, p, persistence=None, thr: float = -1.0) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    out = {
        "n": int(len(y)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "mae": float(mean_absolute_error(y, p)),
        "r2": float(r2_score(y, p)) if len(y) > 1 else float("nan"),
        "explained_variance": float(explained_variance_score(y, p)) if len(y) > 1 else float("nan"),
        "bias": float(np.mean(p - y)),
    }
    if persistence is not None:
        mse_p = mean_squared_error(y, np.asarray(persistence, float))
        out["skill_vs_persistence"] = float(1 - mean_squared_error(y, p) / mse_p) if mse_p > 0 else float("nan")
    # Drought detection (SPEI-12 < thr). Reported with counts because the
    # 2016-2025 test period contains very few drought months.
    obs, pred = y < thr, p < thr
    hits, misses, fa = int((obs & pred).sum()), int((obs & ~pred).sum()), int((~obs & pred).sum())
    out.update(
        drought_months=int(obs.sum()),
        drought_hits=hits,
        drought_misses=misses,
        drought_false_alarms=fa,
        pod=hits / (hits + misses) if hits + misses else None,
        far=fa / (hits + fa) if hits + fa else None,
    )
    return out


def breakdown(frame: pd.DataFrame, by: str) -> pd.DataFrame:
    """frame needs columns: model, horizon, y, pred, persistence and `by`."""
    rows = []
    for (model, h, key), g in frame.groupby(["model", "horizon", by], sort=True):
        m = regression_metrics(g["y"], g["pred"], g["persistence"])
        rows.append({"model": model, "horizon": h, by: key, "n": m["n"], "rmse": m["rmse"],
                     "mae": m["mae"], "bias": m["bias"], "skill_vs_persistence": m.get("skill_vs_persistence")})
    return pd.DataFrame(rows)


def conformal_quantile(abs_residuals: np.ndarray, level: float) -> float:
    """Split-conformal radius from out-of-fold absolute residuals."""
    r = np.sort(np.asarray(abs_residuals, float))
    n = len(r)
    k = min(int(np.ceil((n + 1) * level)), n)
    return float(r[k - 1])


def coverage(y, lo, hi) -> float:
    y = np.asarray(y)
    return float(np.mean((y >= lo) & (y <= hi)))


def block_bootstrap_rmse(y, preds: dict, block: int = 12, n_boot: int = 2000, seed: int = 42) -> dict:
    """Moving-block bootstrap of test RMSE (keeps autocorrelation within blocks).

    Returns, per model, the RMSE 90% interval, plus for every pair (a, b) the
    share of resamples in which a has lower RMSE than b.
    """
    y = np.asarray(y, float)
    n = len(y)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - block + 1, size=(n_boot, int(np.ceil(n / block))))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
    sq = {k: (np.asarray(p, float) - y) ** 2 for k, p in preds.items()}
    boot = {k: np.sqrt(v[idx].mean(axis=1)) for k, v in sq.items()}
    out = {"rmse_90ci": {k: [float(np.percentile(b, 5)), float(np.percentile(b, 95))] for k, b in boot.items()},
           "p_lower": {}}
    for a in boot:
        for b in boot:
            if a != b:
                out["p_lower"][f"{a} < {b}"] = float(np.mean(boot[a] < boot[b]))
    return out
