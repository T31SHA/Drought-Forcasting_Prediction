"""Leakage and protocol checks. Run: python -m pytest -q"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from pipeline.data import load_spei
from pipeline.evaluate import conformal_quantile
from pipeline.features import build_features, build_targets, split_rows
from pipeline.train import make_folds

ROOT = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
DF = load_spei(ROOT / CFG["data"]["path"])


def test_features_do_not_use_future_data():
    """Features at month t are identical whether or not data after t exists."""
    full = build_features(DF, CFG)
    for cut in ["1990-06-01", "2005-12-01", "2015-12-01"]:
        part = build_features(DF.loc[:cut], CFG)
        pd.testing.assert_frame_equal(full.loc[:cut], part)


def test_split_has_no_label_overlap():
    X, Y = build_features(DF, CFG), build_targets(DF, CFG)
    for h in CFG["horizons"]:
        dev, test = split_rows(X, Y[h], h, CFG)
        assert (dev + pd.DateOffset(months=h)).max() <= pd.Timestamp(CFG["split"]["dev_end"])
        assert test.min() > dev.max()
        assert len(test) == 116  # same window as the original notebook


def test_cv_folds_are_chronological_with_gap():
    for h in CFG["horizons"]:
        for tr, va in make_folds(700, h, CFG):
            assert tr.max() + h < va.min()


def test_conformal_quantile():
    r = np.arange(1, 101, dtype=float)
    assert conformal_quantile(r, 0.8) == 81.0


def test_spei1_imputation_is_causal():
    from pipeline.data import fill_spei1_gaps

    full, imputed = fill_spei1_gaps(DF)
    assert imputed == ["2025-11"]
    part, _ = fill_spei1_gaps(DF.loc[:"2025-11-01"])
    assert full.loc["2025-11-01", "SPEI_1"] == part.loc["2025-11-01", "SPEI_1"]
    assert full["SPEI_1"].notna().all()


def test_trend_features_match_original_notebook():
    from pipeline.features import add_trend
    from pipeline.legacy import legacy_frame

    X = add_trend(build_features(DF, CFG), DF.index[0])
    lf, _ = legacy_frame(DF)
    for c in ["year", "decade", "month_since_start"]:
        assert (X.loc[lf.index, c].to_numpy() == lf[c].to_numpy()).all()


def test_adaptive_conformal_uses_only_observed_outcomes():
    from pipeline.evaluate import adaptive_conformal

    cal = np.linspace(0.01, 1.0, 100)
    err = np.r_[np.full(20, 5.0), np.zeros(20)]
    r1, _ = adaptive_conformal(cal, err, 0.8, 0.05, delay=6)
    # The first `delay` radii cannot react to anything yet.
    assert np.allclose(r1[:6], r1[0])
    # Changing errors after step i must not change radii up to i + delay.
    err2 = err.copy(); err2[30:] = 9.0
    r2, _ = adaptive_conformal(cal, err2, 0.8, 0.05, delay=6)
    assert np.allclose(r1[:36], r2[:36])
    # Misses widen the interval.
    assert r1[19] > r1[0]
