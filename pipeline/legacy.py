"""Faithful re-implementation of the original notebook's Ridge model.

Used only to verify the "before" numbers in metrics_comparison.csv against
the same code path and test window as the new pipeline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TARGETS = {1: "y_t1", 3: "y_t3", 6: "y_t6"}


def legacy_frame(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Original feature table (including targets y_t1/y_t3/y_t6) and feature names."""
    df = df_raw.copy()
    df["month"] = df.index.month
    df["year"] = df.index.year
    df["decade"] = (df["year"] // 10) * 10
    df["month_since_start"] = np.arange(len(df))
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    for lag in [1, 2, 3, 6, 9, 12, 24]:
        df[f"SPEI_12_lag_{lag}"] = df["SPEI_12"].shift(lag)
    for lag in [1, 3]:
        df[f"SPEI_1_lag_{lag}"] = df["SPEI_1"].shift(lag)
        df[f"SPEI_6_lag_{lag}"] = df["SPEI_6"].shift(lag)
    for w in [3, 6, 12]:
        df[f"SPEI_12_roll_mean_{w}"] = df["SPEI_12"].shift(1).rolling(w, min_periods=1).mean()
    for w in [3, 6]:
        df[f"SPEI_12_roll_std_{w}"] = df["SPEI_12"].shift(1).rolling(w, min_periods=1).std()
    df["SPEI_12_roll_min_12"] = df["SPEI_12"].shift(1).rolling(12, min_periods=1).min()
    df["SPEI_12_roll_max_12"] = df["SPEI_12"].shift(1).rolling(12, min_periods=1).max()
    df["SPEI_1_minus_12"] = df["SPEI_1"] - df["SPEI_12"]
    df["SPEI_6_minus_24"] = df["SPEI_6"] - df["SPEI_24"]
    df["SPEI_12_times_24"] = df["SPEI_12"] * df["SPEI_24"]
    for h, c in TARGETS.items():
        df[c] = df["SPEI_12"].shift(-h)
    feats = [c for c in df.columns if c not in TARGETS.values()]
    return df.dropna(subset=feats + list(TARGETS.values())), feats


def legacy_ridge(df_raw: pd.DataFrame) -> dict[int, pd.Series]:
    """Exact original protocol: alpha picked on 2006-2015, refit on <=2015."""
    df, feats = legacy_frame(df_raw)
    targets = TARGETS

    tr = df.index <= pd.Timestamp("2005-12-01")
    va = (df.index > pd.Timestamp("2005-12-01")) & (df.index <= pd.Timestamp("2015-12-01"))
    te = df.index > pd.Timestamp("2015-12-01")
    sc = StandardScaler().fit(df.loc[tr, feats])
    Xtr, Xva, Xte = (pd.DataFrame(sc.transform(df.loc[m, feats]), index=df.index[m]) for m in (tr, va, te))

    out = {}
    for h, c in targets.items():
        ytr, yva = df.loc[tr, c], df.loc[va, c]
        best = min([0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
                   key=lambda a: mean_squared_error(yva, Ridge(alpha=a).fit(Xtr, ytr).predict(Xva)))
        model = Ridge(alpha=best).fit(pd.concat([Xtr, Xva]), pd.concat([ytr, yva]))
        out[h] = pd.Series(model.predict(Xte), index=Xte.index)
    return out


class LegacyRidge:
    """Original features + level-target Ridge, for evaluation under the new CV."""

    name = "Previous Ridge"

    def __init__(self, h: int, alpha: float = 1.0, **_):
        self.h, self.alpha = h, alpha

    def fit(self, X, y):
        ok = X.notna().all(axis=1).to_numpy()  # first ~2 years lack 48-month features
        self.model_ = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha)).fit(X[ok], np.asarray(y)[ok])
        return self

    def predict(self, X):
        return self.model_.predict(X)
