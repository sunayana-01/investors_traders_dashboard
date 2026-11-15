"""All forecasting models in one file.

Every model has the same two methods:
    fit(X, y, extra)           learn from the training rows
    predict_proba(X, extra)    return P(y = 1) for each row, a number between 0 and 1
X = features, y = the 0/1 labels, extra = the table with fwd_rv and threshold.

Simple baselines (each turns ONE score into a probability with a one-variable logistic regression):
    B0  base rate        always forecasts the share of 1s in the training data
    B1  trailing vol     score = log of the last 21 days volatility
    B2  EWMA vol         score = log of the exponentially weighted volatility
    B3  GARCH(1,1)       score = log of the GARCH volatility forecast
    B4  HAR regression   score = predicted log future volatility from 5/21/63-day volatility
    B5  VIX              score = log of the VIX

Main models:
    M1  HAR + VIX logistic   logistic regression on log rv_5, rv_21, rv_63 and log VIX (the reference to beat)
    M2  full logistic        logistic regression on all features
    M3  LightGBM             small gradient boosted trees on all features
    M4  GARCH probability    assumes log future volatility is normal around the GARCH forecast

Scaling, clipping and fitting all happen inside fit(), on training rows only.
"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

from riskml.config import TRADING_DAYS
from riskml.features import to_model_matrix

PROB_MIN = 1e-6          # probabilities are kept inside [1e-6, 1 - 1e-6]
LOG_FLOOR = 1e-6         # smallest value allowed before taking a log

LIGHTGBM_SETTINGS = dict(n_estimators=250, learning_rate=0.03, num_leaves=7, max_depth=3,
                         min_child_samples=50, subsample=0.8, subsample_freq=1,
                         colsample_bytree=0.8, reg_lambda=5.0, random_state=0,
                         n_jobs=1, deterministic=True, force_row_wise=True, verbose=-1)

HAR_VIX_FEATURES = ["rv_5", "rv_21", "rv_63", "vix"]


def clip_prob(p):
    return np.clip(np.asarray(p, dtype=float), PROB_MIN, 1 - PROB_MIN)


def safe_log(x):
    return np.log(np.clip(np.asarray(x, dtype=float), LOG_FLOOR, None))


def base_rate_of(y):
    return float(np.clip(np.mean(y), PROB_MIN, 1 - PROB_MIN))


# helper: logistic regression on a single score

class OneVarLogit:
    """Standardise one score, then fit a logistic regression on it.

    If there is too little data or only one class in the labels, it just
    returns the base rate.
    """

    def fit(self, score, y):
        score = np.asarray(score, dtype=float)
        y = np.asarray(y, dtype=int)
        ok = np.isfinite(score)
        self.base_rate = base_rate_of(y)
        self.logit = None
        if ok.sum() < 20 or len(np.unique(y[ok])) < 2:
            return self
        self.mean = score[ok].mean()
        spread = score[ok].std()
        self.spread = spread if spread > 1e-12 else 1.0
        z = ((score[ok] - self.mean) / self.spread).reshape(-1, 1)
        self.logit = LogisticRegression(C=1e4, max_iter=1000).fit(z, y[ok])   # C=1e4 means almost no regularisation
        return self

    def predict(self, score):
        score = np.asarray(score, dtype=float)
        p = np.full(len(score), self.base_rate)
        if self.logit is None:
            return clip_prob(p)
        ok = np.isfinite(score)
        if ok.any():
            z = ((score[ok] - self.mean) / self.spread).reshape(-1, 1)
            p[ok] = self.logit.predict_proba(z)[:, 1]
        return clip_prob(p)


# helper: GARCH(1,1)

def fit_garch(train_returns):
    """Fit a zero-mean GARCH(1,1) with the arch package.

    Returns (omega, alpha, beta, start_variance). If the fit fails we fall back to
    a simple RiskMetrics style model (alpha 0.06, beta 0.94).
    """
    from arch import arch_model   # imported here because it is slow to import

    x = np.asarray(train_returns, dtype=float)
    x = x[np.isfinite(x)]
    start_variance = float(np.var(x)) if len(x) > 1 else 1e-4
    fallback = (0.0, 0.06, 0.94, start_variance)
    if len(x) < 250 or start_variance <= 0:
        return fallback
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = arch_model(100 * x, mean="Zero", vol="GARCH", p=1, q=1, dist="normal", rescale=False)
            result = model.fit(disp="off", show_warning=False)
        omega = float(result.params["omega"]) / 1e4   # we fitted on returns * 100
        alpha = float(result.params["alpha[1]"])
        beta = float(result.params["beta[1]"])
    except Exception:
        return fallback
    if not (np.isfinite([omega, alpha, beta]).all() and omega > 0 and alpha >= 0 and beta >= 0):
        return fallback
    if alpha + beta >= 0.9999:   # keep the forecast formula finite
        shrink = 0.9999 / (alpha + beta)
        alpha, beta = alpha * shrink, beta * shrink
    return omega, alpha, beta, start_variance


def garch_forecast(returns, params, h):
    """Annualised GARCH forecast of the average volatility over the next h days, for every day.

    The variance is updated day by day with fixed parameters, so the value of day t
    only uses returns up to day t.
    """
    omega, alpha, beta, start_variance = params
    x = np.nan_to_num(returns.to_numpy(dtype=float), nan=0.0)
    next_variance = np.empty(len(x))
    variance = start_variance
    for t in range(len(x)):
        variance = omega + alpha * x[t] ** 2 + beta * variance
        next_variance[t] = variance

    # average expected variance over the next h days (closed form for GARCH(1,1))
    persistence = min(alpha + beta, 0.9999)
    long_run = omega / (1 - persistence)
    average = long_run + (next_variance - long_run) * (1 - persistence ** h) / (h * (1 - persistence))
    return pd.Series(np.sqrt(np.clip(average, 0, None) * TRADING_DAYS), index=returns.index)


def garch_scores(returns, params, h, dates):
    """GARCH forecast on the given dates, using returns only up to the last of these dates."""
    forecast = garch_forecast(returns.loc[:dates.max()], params, h)
    return forecast.reindex(dates).to_numpy()


# baselines

class BaseRate:
    """B0"""

    def fit(self, X, y, extra=None):
        self.p = base_rate_of(y)
        return self

    def predict_proba(self, X, extra=None):
        return np.full(len(X), self.p)


class OneFeatureLogit:
    """B1, B2 and B5: log of one feature column, then a one-variable logistic regression."""

    def __init__(self, column):
        self.column = column

    def fit(self, X, y, extra=None):
        self.mapper = OneVarLogit().fit(safe_log(X[self.column]), y)
        return self

    def predict_proba(self, X, extra=None):
        return self.mapper.predict(safe_log(X[self.column]))


class GarchLogit:
    """B3"""

    def __init__(self, returns, h):
        self.returns = returns
        self.h = h

    def fit(self, X, y, extra=None):
        # parameters are estimated with returns up to the last training day only
        self.params = fit_garch(self.returns.loc[:X.index[-1]])
        score = safe_log(garch_scores(self.returns, self.params, self.h, X.index))
        self.mapper = OneVarLogit().fit(score, y)
        return self

    def predict_proba(self, X, extra=None):
        score = safe_log(garch_scores(self.returns, self.params, self.h, X.index))
        return self.mapper.predict(score)


class HarRegression:
    """B4: regress log future volatility on log rv_5, rv_21, rv_63, then map the prediction to a probability."""

    columns = ["rv_5", "rv_21", "rv_63"]

    def design_matrix(self, X):
        logs = safe_log(X[self.columns].to_numpy())
        return np.column_stack([np.ones(len(logs)), logs])   # first column of ones = intercept

    def fit(self, X, y, extra):
        D = self.design_matrix(X)
        target = safe_log(extra["fwd_rv"].to_numpy())
        ok = np.isfinite(D).all(axis=1) & np.isfinite(target)
        self.coef, *_ = np.linalg.lstsq(D[ok], target[ok], rcond=None)
        self.mapper = OneVarLogit().fit(D @ self.coef, y)
        return self

    def predict_proba(self, X, extra=None):
        return self.mapper.predict(self.design_matrix(X) @ self.coef)


# main models

class TrainingScaler:
    """Clip extreme values, fill gaps with the median, then standardise.

    Everything is learned from the training rows only.
    """

    def fit(self, M):
        A = np.asarray(M, dtype=float)
        with np.errstate(all="ignore"):
            self.low = np.nanquantile(A, 0.01, axis=0)
            self.high = np.nanquantile(A, 0.99, axis=0)
            clipped = np.clip(A, self.low, self.high)
            self.median = np.nanmedian(clipped, axis=0)
        self.median = np.where(np.isfinite(self.median), self.median, 0.0)
        clipped = np.where(np.isfinite(clipped), clipped, self.median)
        self.mean = clipped.mean(axis=0)
        spread = clipped.std(axis=0)
        self.spread = np.where(spread > 1e-12, spread, 1.0)
        return self

    def transform(self, M):
        A = np.asarray(M, dtype=float)
        clipped = np.clip(A, self.low, self.high)
        clipped = np.where(np.isfinite(clipped), clipped, self.median)
        return (clipped - self.mean) / self.spread


class FeatureLogit:
    """M1 and M2: logistic regression with an L2 penalty (C is its inverse strength).

    features=None means use every column.
    """

    def __init__(self, C, features=None):
        self.C = C
        self.features = features

    def matrix(self, X):
        M = to_model_matrix(X)
        if self.features is None:
            return M
        # the log columns are called log_rv_5 and so on, so we compare the name without "log_"
        wanted = [c for c in M.columns if c.replace("log_", "", 1) in self.features]
        return M[wanted]

    def fit(self, X, y, extra=None):
        y = np.asarray(y, dtype=int)
        M = self.matrix(X)
        self.columns = list(M.columns)
        self.scaler = TrainingScaler().fit(M)
        self.base_rate = base_rate_of(y)
        self.logit = None
        if len(np.unique(y)) == 2:
            self.logit = LogisticRegression(C=self.C, max_iter=2000).fit(self.scaler.transform(M), y)
        return self

    def predict_proba(self, X, extra=None):
        if self.logit is None:
            return np.full(len(X), self.base_rate)
        M = self.matrix(X)[self.columns]
        return clip_prob(self.logit.predict_proba(self.scaler.transform(M))[:, 1])


class LightGBMModel:
    """M3: boosted decision trees. Settings are small and fixed so it does not overfit and gives the same result every run."""

    def fit(self, X, y, extra=None):
        from lightgbm import LGBMClassifier

        y = np.asarray(y, dtype=int)
        self.columns = list(X.columns)
        self.base_rate = base_rate_of(y)
        self.trees = None
        if len(np.unique(y)) == 2:
            self.trees = LGBMClassifier(**LIGHTGBM_SETTINGS).fit(X[self.columns], y)
        return self

    def predict_proba(self, X, extra=None):
        if self.trees is None:
            return np.full(len(X), self.base_rate)
        return clip_prob(self.trees.predict_proba(X[self.columns])[:, 1])


class HistGradientBoostingModel:
    """M3H: histogram-based gradient boosting as a stronger tabular alternative to LightGBM."""

    def __init__(self, learning_rate=0.05, max_depth=3, max_leaf_nodes=31,
                 min_samples_leaf=20, l2_regularization=0.1, features=None):
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.features = features

    def _cols(self, X):
        if self.features is None:
            return list(X.columns)
        missing = [c for c in self.features if c not in X.columns]
        if missing:
            raise KeyError(f"Missing columns for M3H: {missing}")
        return list(self.features)

    def fit(self, X, y, extra=None):
        from sklearn.ensemble import HistGradientBoostingClassifier

        y = np.asarray(y, dtype=int)
        self.columns = self._cols(X)
        self.base_rate = base_rate_of(y)
        self.model = None
        if len(np.unique(y)) == 2:
            self.model = HistGradientBoostingClassifier(
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                max_leaf_nodes=self.max_leaf_nodes,
                min_samples_leaf=self.min_samples_leaf,
                l2_regularization=self.l2_regularization,
                random_state=0,
            )
            self.model.fit(X[self.columns], y)
        return self

    def predict_proba(self, X, extra=None):
        if self.model is None:
            return np.full(len(X), self.base_rate)
        return clip_prob(self.model.predict_proba(X[self.columns])[:, 1])


class GarchProbability:
    """M4: assume log(future vol) = a + b * log(GARCH vol) + normal noise.

    The probability is then P(future vol > threshold) under that normal distribution.
    This is the only model that uses the threshold at prediction time. That is fine
    because the threshold only uses the past.
    """

    def __init__(self, returns, h):
        self.returns = returns
        self.h = h

    def fit(self, X, y, extra):
        self.params = fit_garch(self.returns.loc[:X.index[-1]])
        score = safe_log(garch_scores(self.returns, self.params, self.h, X.index))
        target = safe_log(extra["fwd_rv"].to_numpy())
        ok = np.isfinite(score) & np.isfinite(target)
        D = np.column_stack([np.ones(ok.sum()), score[ok]])
        self.coef, *_ = np.linalg.lstsq(D, target[ok], rcond=None)
        residuals = target[ok] - D @ self.coef
        self.noise = float(max(residuals.std(ddof=2), 1e-3))
        self.base_rate = base_rate_of(y)
        return self

    def predict_proba(self, X, extra):
        score = safe_log(garch_scores(self.returns, self.params, self.h, X.index))
        expected_log_vol = self.coef[0] + self.coef[1] * score
        z = (safe_log(extra["threshold"].to_numpy()) - expected_log_vol) / self.noise
        p = norm.sf(z)   # probability of being above the threshold
        p = np.where(np.isfinite(p), p, self.base_rate)
        return clip_prob(p)


MODEL_NAMES = ["B0", "B1", "B2", "B3", "B4", "B5", "M1", "M2", "M3", "M3H", "M4"]


def build_model(code, returns, h):
    """A new, untrained model. returns = the portfolio return series (needed by GARCH models)."""
    code = code.strip().upper()
    if code == "B0":
        return BaseRate()
    if code == "B1":
        return OneFeatureLogit("rv_21")
    if code == "B2":
        return OneFeatureLogit("ewma_vol")
    if code == "B3":
        return GarchLogit(returns, h)
    if code == "B4":
        return HarRegression()
    if code == "B5":
        return OneFeatureLogit("vix")
    if code == "M1":
        return FeatureLogit(C=1.0, features=HAR_VIX_FEATURES)
    if code == "M2":
        return FeatureLogit(C=0.1)
    if code == "M3":
        return LightGBMModel()
    if code == "M3H":
        return HistGradientBoostingModel()
    if code == "M4":
        return GarchProbability(returns, h)
    raise KeyError(f"Unknown model {code!r}. Choose from {MODEL_NAMES}.")
