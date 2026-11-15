"""The target we want to predict: will the next h days be unusually volatile?

For a day t (position = close of day t):
  future volatility = annualised standard deviation of the returns on days t+1 ... t+h
  threshold         = 75th percentile of all future-volatility values we already KNOW at day t
  label (y)         = 1 if future volatility is above the threshold, otherwise 0

A future-volatility value for day s is only known once day s+h has closed. So the
threshold at day t only uses days s <= t - h. This way the label never uses
information that was not available at the time.
"""
import numpy as np
import pandas as pd

from riskml.config import TRADING_DAYS

MIN_THRESHOLD_OBS = 252   # need at least a year of known values before a threshold exists


def make_labels(port_ret, h, quantile=0.75, burn_in=504):
    """Returns a table with the columns fwd_rv, threshold, y and label_end.

    fwd_rv     volatility over the next h days (empty for the last h rows)
    threshold  the past-only threshold described above
    y          1.0 or 0.0, empty where it cannot be known (burn-in period, last h rows)
    label_end  date of the last return used by the label, needed later for purging
    """
    if h < 2:
        raise ValueError("The horizon must be at least 2 days.")
    if not port_ret.index.is_monotonic_increasing or port_ret.index.has_duplicates:
        raise ValueError("The return dates must be sorted and unique.")

    # std of the h returns that END on day t, then shift back so it sits on day t-h:
    # at day t this gives the volatility of days t+1 ... t+h
    fwd_rv = port_ret.rolling(h).std(ddof=1).shift(-h) * np.sqrt(TRADING_DAYS)

    # at day t the newest value we know is the one from day t-h
    known_at_t = fwd_rv.shift(h)
    threshold = known_at_t.expanding(min_periods=MIN_THRESHOLD_OBS).quantile(quantile)

    defined = fwd_rv.notna() & threshold.notna()
    y = (fwd_rv > threshold).astype(float).where(defined)
    y.iloc[:burn_in] = np.nan

    label_end = port_ret.index.to_series().shift(-h)
    return pd.DataFrame({"fwd_rv": fwd_rv, "threshold": threshold, "y": y,
                         "label_end": label_end.to_numpy()}, index=port_ret.index)
