"""Data loading."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_spei(path: str | Path, date_column: str = "DATA", date_format: str = "%b%Y") -> pd.DataFrame:
    """Load the SPEI Global Drought Monitor CSV into a monthly DataFrame.

    Index: month-start timestamps. Columns: SPEI_1 ... SPEI_48.
    """
    df = pd.read_csv(path)
    df.index = pd.to_datetime(df.pop(date_column), format=date_format)
    df.index.name = "date"
    df = df.sort_index()
    if not df.index.is_unique:
        raise ValueError("duplicate months in input data")
    expected = pd.date_range(df.index[0], df.index[-1], freq="MS")
    if len(expected) != len(df):
        raise ValueError("input data has missing months")
    df.index.freq = "MS"
    return df.astype(float)


def fill_spei1_gaps(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Causally impute isolated missing SPEI-1 values.

    The source file is missing SPEI-1 for Nov 2025, which would otherwise
    break every SPEI-1 lag / rolling feature for the following year. Each gap
    is filled from a linear fit SPEI_1[t] ~ SPEI_2[t] + SPEI_1[t-1] estimated
    on months *before* the gap only (5-fold CV R^2 ~0.92, RMSE ~0.29).
    Returns the filled frame and the list of imputed months.
    """
    from sklearn.linear_model import LinearRegression

    df = df.copy()
    imputed = []
    for t in df.index[df["SPEI_1"].isna()]:
        i = df.index.get_loc(t)
        if i == 0 or pd.isna(df["SPEI_2"].iloc[i]) or pd.isna(df["SPEI_1"].iloc[i - 1]):
            continue
        hist = pd.DataFrame({"s2": df["SPEI_2"], "l1": df["SPEI_1"].shift(1), "y": df["SPEI_1"]}).iloc[:i].dropna()
        model = LinearRegression().fit(hist[["s2", "l1"]], hist["y"])
        x = pd.DataFrame({"s2": [df["SPEI_2"].iloc[i]], "l1": [df["SPEI_1"].iloc[i - 1]]})
        df.iloc[i, df.columns.get_loc("SPEI_1")] = float(model.predict(x)[0])
        imputed.append(f"{t:%Y-%m}")
    return df, imputed
