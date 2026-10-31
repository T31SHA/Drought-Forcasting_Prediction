"""Baselines and learned models behind one small interface.

All models predict the SPEI-12 *level* at t + h. Ridge and XGBoost learn the
change from persistence (y[t+h] - SPEI12[t]), so regularisation shrinks them
towards "no change" rather than towards the long-term mean.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ANCHOR = "spei_12"


class Persistence:
    name = "Persistence"

    def __init__(self, h: int, **_):
        self.h = h

    def fit(self, X, y):
        return self

    def predict(self, X):
        return X[ANCHOR].to_numpy()


class Climatology:
    """Mean SPEI-12 of the target calendar month, from training rows only."""

    name = "Climatology"

    def __init__(self, h: int, **_):
        self.h = h

    def _target_month(self, X):
        return (X.index.month - 1 + self.h) % 12 + 1

    def fit(self, X, y):
        y = pd.Series(np.asarray(y), index=X.index)
        self.by_month_ = y.groupby(self._target_month(X)).mean()
        self.overall_ = float(y.mean())
        return self

    def predict(self, X):
        return self.by_month_.reindex(self._target_month(X)).fillna(self.overall_).to_numpy()


class DampedPersistence:
    """AR(1)-style regression y = a + b * SPEI12[t]."""

    name = "Damped persistence"

    def __init__(self, h: int, **_):
        self.h = h

    def fit(self, X, y):
        self.model_ = LinearRegression().fit(X[[ANCHOR]], y)
        return self

    def predict(self, X):
        return self.model_.predict(X[[ANCHOR]])


class RidgeModel:
    name = "Ridge"

    def __init__(self, h: int, alpha: float = 1.0, **_):
        self.h, self.alpha = h, alpha

    def fit(self, X, y):
        self.model_ = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha))
        self.model_.fit(X, np.asarray(y) - X[ANCHOR].to_numpy())
        return self

    def predict(self, X):
        return X[ANCHOR].to_numpy() + self.model_.predict(X)

    def contributions(self, X) -> np.ndarray:
        """Exact linear SHAP values for the residual model (vs. training mean)."""
        scaler, ridge = self.model_[0], self.model_[1]
        return scaler.transform(X) * ridge.coef_


class XGBModel:
    name = "XGBoost"

    def __init__(self, h: int, seed: int = 42, **params):
        self.h, self.seed, self.params = h, seed, params

    def fit(self, X, y):
        self.model_ = xgb.XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            random_state=self.seed,
            n_jobs=4,
            **self.params,
        )
        self.model_.fit(X, np.asarray(y) - X[ANCHOR].to_numpy())
        return self

    def predict(self, X):
        return X[ANCHOR].to_numpy() + self.model_.predict(X)

    def contributions(self, X) -> np.ndarray:
        """Exact TreeSHAP values from XGBoost (bias column dropped)."""
        dm = xgb.DMatrix(X, feature_names=list(X.columns))
        return self.model_.get_booster().predict(dm, pred_contribs=True)[:, :-1]


BASELINES = [Persistence, Climatology, DampedPersistence]


class Blend:
    """Equal-weight average of Ridge and XGBoost."""

    name = "Ridge + XGBoost"

    def __init__(self, h: int, ridge: dict, xgb_params: dict, seed: int = 42):
        self.parts = [RidgeModel(h, **ridge), XGBModel(h, seed=seed, **xgb_params)]

    def fit(self, X, y):
        for p in self.parts:
            p.fit(X, y)
        return self

    def predict(self, X):
        return np.mean([p.predict(X) for p in self.parts], axis=0)

    def contributions(self, X) -> np.ndarray:
        return np.mean([p.contributions(X) for p in self.parts], axis=0)
