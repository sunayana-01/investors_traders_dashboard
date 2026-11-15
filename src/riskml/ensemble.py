"""Ensembles: one forecast made by averaging several models.

We average in logit space (log-odds) instead of averaging the probabilities.
That stops one very unsure model from dragging the average towards 50%.
"""
import numpy as np
import pandas as pd

from riskml.evaluate import logit

ENSEMBLES = {
    "E-core": ["M1", "M2", "M3", "M4"],   # main models, excluding the newer boosted-tree variant
    "E-ML": ["M2", "M3"],                 # only the core machine-learning models
    "E-M1M4": ["M1", "M3H", "M4"],       # simple ensemble of the reference, histogram boosting, and GARCH probability models
}


def expit(z):
    """The opposite of logit: turns log-odds back into a probability."""
    return 1 / (1 + np.exp(-z))


def logit_average(preds, members):
    """Average of the members' log-odds, turned back into a probability.

    A day only gets a forecast if every member has one.
    """
    log_odds = pd.DataFrame({m: logit(preds[m]) for m in members}, index=preds.index)
    return expit(log_odds.mean(axis=1, skipna=False))


def build_ensembles(preds):
    """Build every ensemble that has at least two members among the available models.

    Returns the ensemble forecasts and a dict with the members that were used.
    """
    forecasts = {}
    used = {}
    for name, members in ENSEMBLES.items():
        available = [m for m in members if m in preds.columns]
        if len(available) >= 2:
            forecasts[name] = logit_average(preds, available)
            used[name] = available
    return pd.DataFrame(forecasts, index=preds.index), used
