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


def adaptive_conformal(cal_scores, abs_errors, level: float, gamma: float, delay: int):
    """Adaptive conformal inference (Gibbs & Candes 2021) with delayed feedback.

    cal_scores: out-of-fold |residuals| from the development period (fixed pool).
    abs_errors: |y - pred| for consecutive monthly forecasts, in time order.
    The outcome of forecast j is only observed `delay` steps later, so the
    radius for step i uses feedback from steps <= i - delay only.

    Update: alpha <- alpha + gamma * (target_miss_rate - missed_j). The radius
    is the (1 - alpha) conformal quantile of cal_scores; alpha <= 0 maps to
    the largest calibration score, alpha >= 1 to zero.

    Returns (radii per step, radius for the next forecast after all feedback
    that is observable once the series ends).
    """
    cal = np.sort(np.asarray(cal_scores, float))
    target = 1.0 - level

    def radius(alpha: float) -> float:
        if alpha <= 1.0 / (len(cal) + 1):
            return float(cal[-1])
        if alpha >= 1.0:
            return 0.0
        return conformal_quantile(cal, 1.0 - alpha)

    alpha, radii = target, []
    e = np.asarray(abs_errors, float)
    for i in range(len(e)):
        j = i - delay
        if j >= 0:
            alpha += gamma * (target - float(e[j] > radii[j]))
        radii.append(radius(alpha))
    # Feedback still pending at the end of the series becomes observable
    # by the time of the next live forecast (the last `delay` outcomes).
    for j in range(max(0, len(e) - delay), len(e)):
        alpha += gamma * (target - float(e[j] > radii[j]))
    return np.array(radii), radius(alpha)


def select_gamma(fold_abs_errors: list, level: float, grid, delay: int, window: int = 60) -> tuple[float, list]:
    """Choose the adaptive-conformal step size on the development period only.

    Each CV fold's out-of-fold errors are treated as a live period, calibrated
    on the errors of all *earlier* folds, so the procedure faces the regime
    shifts inside 1973-2015 the way the live system faces new ones. Score:
    mean |rolling `window`-month coverage - level|; ties (within 0.001) go to
    the narrower interval. Returns (best gamma, table of all candidates).
    """
    rows = []
    for g in grid:
        hits, widths = [], []
        for k in range(1, len(fold_abs_errors)):
            cal = np.concatenate(fold_abs_errors[:k])
            e = np.asarray(fold_abs_errors[k], float)
            r, _ = adaptive_conformal(cal, e, level, g, delay)
            hits.append((e <= r).astype(float))
            widths.append(r * 2)
        hit = np.concatenate(hits)
        roll = pd.Series(hit).rolling(window).mean().dropna()
        rows.append({"gamma": float(g), "dev_coverage": float(hit.mean()),
                     "dev_rolling_mad": float(np.mean(np.abs(roll - level))),
                     "dev_mean_width": float(np.concatenate(widths).mean())})
    best_mad = min(x["dev_rolling_mad"] for x in rows)
    tied = [x for x in rows if x["dev_rolling_mad"] <= best_mad + 1e-3]
    return min(tied, key=lambda x: x["dev_mean_width"])["gamma"], rows
