"""Feature engineering and leakage-safe splits.

Every feature for forecast-origin month t uses only values observed at or
before t. Targets are SPEI-12 at t + h.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _run_length(mask: pd.Series) -> pd.Series:
    """Consecutive months (ending at t) for which mask is True."""
    groups = (~mask).cumsum()
    return mask.astype(int).groupby(groups).cumsum()


def _months_since(mask: pd.Series, cap: int = 120) -> pd.Series:
    """Months since mask was last True (0 when True now), capped."""
    out, last = [], None
    for i, flag in enumerate(mask.to_numpy()):
        if flag:
            last = i
        out.append(cap if last is None else min(i - last, cap))
    return pd.Series(out, index=mask.index, dtype=float)


def build_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    fc = cfg["features"]
    target = f"SPEI_{cfg['data']['target_scale']}"
    s12, s1 = df[target], df["SPEI_1"]
    thr = fc["drought_threshold"]

    X = pd.DataFrame(index=df.index)
    # Current multi-scale SPEI: SPEI_{12-h}(t) is the part of the future
    # 12-month window that is already observed at the origin.
    for k in fc["spei_scales"]:
        X[f"spei_{k}"] = df[f"SPEI_{k}"]
    for lag in fc["target_lags"]:
        X[f"spei12_lag_{lag}"] = s12.shift(lag)
    for lag in fc["short_lags"]:
        X[f"spei1_lag_{lag}"] = s1.shift(lag)
    # SPI-style accumulations of the monthly water-balance index.
    for w in fc["rolling_windows"]:
        X[f"spei1_mean_{w}"] = s1.rolling(w).mean()
        X[f"spei1_std_{w}"] = s1.rolling(w).std()
        X[f"spei12_mean_{w}"] = s12.rolling(w).mean()
    X["spei12_change_1"] = s12 - s12.shift(1)
    X["spei12_change_3"] = s12 - s12.shift(3)
    X["spei3_minus_12"] = df["SPEI_3"] - s12
    X["spei12_minus_24"] = s12 - df["SPEI_24"]
    # Drought-regime descriptors.
    in_drought = s12 < thr
    X["drought_run_months"] = _run_length(in_drought).astype(float)
    X["months_since_drought"] = _months_since(in_drought)
    X["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    X["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)
    return X


def build_targets(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    s = df[f"SPEI_{cfg['data']['target_scale']}"]
    return pd.DataFrame({h: s.shift(-h) for h in cfg["horizons"]}, index=df.index)


def split_rows(X: pd.DataFrame, y: pd.Series, h: int, cfg: dict):
    """Return (dev_index, test_index) for horizon h.

    dev rows: complete features and target month <= dev_end (purged so no
    target leaks into the test period). test rows: origins inside the fixed
    evaluation window.
    """
    sp = cfg["split"]
    dev_end = pd.Timestamp(sp["dev_end"])
    ok = X.notna().all(axis=1) & y.notna()
    target_month = X.index + pd.DateOffset(months=h)
    dev = X.index[ok & (target_month <= dev_end)]
    test = X.index[ok & (X.index >= pd.Timestamp(sp["test_start"])) & (X.index <= pd.Timestamp(sp["test_end"]))]
    assert dev.max() < test.min()
    assert (dev + pd.DateOffset(months=h)).max() < test.min()
    return dev, test


def season_of(months) -> np.ndarray:
    m = np.asarray(months)
    return np.select(
        [np.isin(m, [12, 1, 2]), np.isin(m, [3, 4, 5]), np.isin(m, [6, 7, 8])],
        ["DJF (winter)", "MAM (spring)", "JJA (summer)"],
        "SON (autumn)",
    )


def regime_of(spei: np.ndarray, thr: float = -1.0) -> np.ndarray:
    v = np.asarray(spei)
    return np.select([v < thr, v > -thr], ["Drought (< -1)", "Wet (> +1)"], "Near normal")
