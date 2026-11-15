"""Fake market data for the tests, so nothing has to be downloaded."""
import numpy as np
import pandas as pd

from riskml.data import AlignedData

WEIGHTS = {"SPY": 0.6, "IEF": 0.3, "GLD": 0.1}
# fast settings: few models, few bootstrap samples, no saved results
FAST = dict(models=("M1", "M2"), ensemble=None, n_boot=50, use_cache=False)


def make_aligned(end="2021-12-31", seed=0, shock_from=None, shock=3.0):
    """Random prices for SPY, IEF and GLD plus fake VIX and macro series.

    If shock_from is given, everything from that date on is made 3 times as wild.
    """
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2004-11-18", end)
    n = len(days)

    vol = np.exp(np.cumsum(rng.normal(0, 0.02, n)) * 0.3)
    vol = 0.008 * vol / vol.mean()
    market = rng.normal(0, 1, n)
    returns = pd.DataFrame({"SPY": market * vol + rng.normal(0, 0.004, n),
                            "IEF": -0.3 * market * vol + rng.normal(0, 0.003, n),
                            "GLD": 0.1 * market * vol + rng.normal(0, 0.008, n)}, index=days)
    vix = pd.Series(100 * returns["SPY"].abs().rolling(21, min_periods=1).mean() * 16 + 8, index=days, name="vix")
    macro = pd.DataFrame({"curve_slope": np.cumsum(rng.normal(0, 0.01, n)),
                          "credit_spread": 2 + np.cumsum(rng.normal(0, 0.01, n)),
                          "fin_stress": np.cumsum(rng.normal(0, 0.02, n))}, index=days)

    if shock_from is not None:
        after = days >= pd.Timestamp(shock_from)
        returns.loc[after] *= shock
        vix[after] *= shock
        macro[after] *= shock

    prices = 100 * (1 + returns).cumprod()
    return AlignedData(prices=prices, vix=vix, macro=macro,
                       cash_daily=pd.Series(0.02 / 252, index=days, name="cash_daily"),
                       start=days[0], end=days[-1])
