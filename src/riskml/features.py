"""Features: the inputs of the models.

Every feature on day t only uses information from day t or earlier. All rolling
windows look backwards. Nothing here looks into the future.

Scaling and filling gaps are NOT done here. That happens inside the models,
and only on the training data (see models.py).
"""
import warnings

import numpy as np
import pandas as pd

from riskml.config import TRADING_DAYS

SQRT_YEAR = np.sqrt(TRADING_DAYS)
EWMA_LAMBDA = 0.94      # weight of yesterday in the exponentially weighted volatility
DRAWDOWN_WINDOW = 252   # drawdown is measured against the highest value of the last year
FILL_LIMIT = 5          # VIX / macro values can be carried forward for at most 5 days

# Volatility style features are very skewed, so the logistic models use their log.
# The number is the smallest value we allow before taking the log.
LOG_FLOORS = {
    "rv_5": 1e-6, "rv_21": 1e-6, "rv_63": 1e-6,
    "vol_ratio_5_21": 1e-6, "vol_ratio_21_63": 1e-6,
    "vol_of_vol": 1e-6, "ewma_vol": 1e-6, "downside_dev_63": 1e-6,
    "abs_ret_1d": 1e-4, "spy_rv_21": 1e-6, "vix": 1e-6,
}


def realized_vol(returns, window):
    """Annualised standard deviation of the last `window` returns."""
    return returns.rolling(window).std(ddof=1) * SQRT_YEAR


def drawdown(value, window):
    """How far the value is below its highest value of the last `window` days."""
    return value / value.rolling(window).max() - 1


def total_return(returns, days):
    """Compounded return over the last `days` days."""
    return np.expm1(np.log1p(returns).rolling(days).sum())


def average_pairwise_corr(asset_returns, window):
    """Average of the rolling correlations of every pair of assets."""
    columns = list(asset_returns.columns)
    total = None
    pairs = 0
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            corr = asset_returns[columns[i]].rolling(window).corr(asset_returns[columns[j]])
            total = corr if total is None else total + corr
            pairs += 1
    return total / pairs


def build_features(port_ret, asset_ret, vix, macro, tickers):
    """Build the feature table. One row per trading day, one column per feature.

    port_ret   daily return of the portfolio
    asset_ret  daily returns of the assets (must contain the portfolio tickers, SPY and IEF)
    vix        VIX close of every day
    macro      columns curve_slope, credit_spread, fin_stress (already delayed by data.py)
    tickers    the tickers in the portfolio
    """
    if not port_ret.index.is_monotonic_increasing or port_ret.index.has_duplicates:
        raise ValueError("The return dates must be sorted and unique.")

    days = port_ret.index
    port_ret = port_ret.astype(float)
    assets = asset_ret.reindex(days)
    vix = vix.reindex(days).ffill(limit=FILL_LIMIT).astype(float)
    macro = macro.reindex(days).ffill(limit=FILL_LIMIT).astype(float)
    value = (1 + port_ret).cumprod()   # growth of 1 invested in the portfolio

    f = {}

    # volatility
    rv5, rv21, rv63 = realized_vol(port_ret, 5), realized_vol(port_ret, 21), realized_vol(port_ret, 63)
    f["rv_5"], f["rv_21"], f["rv_63"] = rv5, rv21, rv63
    f["vol_ratio_5_21"] = rv5 / rv21
    f["vol_ratio_21_63"] = rv21 / rv63
    f["vol_of_vol"] = rv21.rolling(63).std(ddof=1)
    ewma_variance = (port_ret ** 2).ewm(alpha=1 - EWMA_LAMBDA, adjust=False, min_periods=21).mean()
    f["ewma_vol"] = np.sqrt(ewma_variance * TRADING_DAYS)
    f["abs_ret_1d"] = port_ret.abs()
    # where today's 21-day volatility ranks among all values seen so far (0 to 1)
    f["rv_21_pct"] = rv21.expanding(min_periods=252).rank(method="max", pct=True)

    # trend and tail risk
    f["ret_21"] = total_return(port_ret, 21)
    f["ret_63"] = total_return(port_ret, 63)
    f["ret_126"] = total_return(port_ret, 126)
    f["drawdown"] = drawdown(value, DRAWDOWN_WINDOW)
    f["dist_ma200"] = value / value.rolling(200).mean() - 1
    losses_only = port_ret.clip(upper=0)
    f["downside_dev_63"] = np.sqrt((losses_only ** 2).rolling(63).mean() * TRADING_DAYS)
    f["skew_63"] = port_ret.rolling(63).skew()

    # correlation and the wider market
    in_portfolio = [t for t in tickers if t in assets.columns]
    if len(in_portfolio) >= 2:   # a one-asset portfolio has no correlation between assets
        f["avg_corr_63"] = average_pairwise_corr(assets[in_portfolio], 63)
    f["stock_bond_corr_63"] = assets["SPY"].rolling(63).corr(assets["IEF"])
    f["spy_rv_21"] = realized_vol(assets["SPY"], 21)
    f["spy_drawdown"] = drawdown((1 + assets["SPY"]).cumprod(), DRAWDOWN_WINDOW)

    # VIX
    f["vix"] = vix
    f["vix_chg_5"] = np.log(vix.where(vix > 0)).diff(5)
    f["vix_rel_63"] = vix / vix.rolling(63).mean() - 1

    # macro and credit
    f["curve_slope"] = macro["curve_slope"]
    f["curve_chg_21"] = macro["curve_slope"].diff(21)
    f["credit_spread"] = macro["credit_spread"]
    f["credit_chg_21"] = macro["credit_spread"].diff(21)
    f["fin_stress"] = macro["fin_stress"]

    X = pd.DataFrame(f, index=days).astype(float).replace([np.inf, -np.inf], np.nan)

    # drop the first rows where the longest window is not full yet (and any row with a gap)
    complete = X.notna().all(axis=1).to_numpy()
    if not complete.any():
        raise ValueError("No row has all features. The price history may be too short.")
    first = int(np.argmax(complete))
    gaps_after_start = int((~complete[first:]).sum())
    if gaps_after_start:
        warnings.warn(f"{gaps_after_start} row(s) after the start were dropped because an input was "
                      "missing, so the feature table has gaps.", stacklevel=2)
    return X.loc[complete]


def to_model_matrix(X):
    """Take the log of the volatility features (they are called log_<name> afterwards).

    This works row by row and has no parameters, so it cannot leak anything.
    """
    out = {}
    for column in X.columns:
        if column in LOG_FLOORS:
            out["log_" + column] = np.log(X[column].clip(lower=LOG_FLOORS[column]))
        else:
            out[column] = X[column]
    return pd.DataFrame(out, index=X.index).replace([np.inf, -np.inf], np.nan)
