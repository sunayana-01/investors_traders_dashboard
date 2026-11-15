"""Scores for probability forecasts.

y is the true outcome (0 or 1) and p is the forecast probability that y is 1.
Both can be pandas Series or arrays. They must be in the same order.
Rows where either value is missing are ignored.
"""
import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

EPS = 1e-6   # keeps probabilities away from exactly 0 and 1


def clean_pair(y, p):
    """Both inputs as float arrays, with the rows that have a missing value removed."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    keep = np.isfinite(y) & np.isfinite(p)
    return y[keep].astype(int), p[keep]


def has_one_class(y):
    return len(y) == 0 or y.min() == y.max()


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def expanding_base_rate(y, h, prior=0.25):
    """Reference forecast B0: the share of 1s among the labels already known at day t.

    A label of day s is known at day s+h, so at day t we use the labels up to day t-h.
    Days without any history yet get the prior (25%, because of how the label is built).
    """
    return y.shift(h).expanding().mean().fillna(prior).rename("B0")


# the scores

def brier(y, p):
    """Mean squared error of the probabilities. Lower is better."""
    y, p = clean_pair(y, p)
    return float(np.mean((p - y) ** 2)) if len(y) else float("nan")


def log_loss(y, p):
    """Punishes confident wrong forecasts even harder than Brier. Lower is better."""
    y, p = clean_pair(y, p)
    if not len(y):
        return float("nan")
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier_skill(y, p, p_ref):
    """1 minus (Brier of p / Brier of the reference). Above 0 means better than the reference."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    p_ref = np.asarray(p_ref, dtype=float)
    keep = np.isfinite(y) & np.isfinite(p) & np.isfinite(p_ref)
    y, p, p_ref = y[keep], p[keep], p_ref[keep]
    if not len(y):
        return float("nan")
    brier_ref = np.mean((p_ref - y) ** 2)
    return float(1 - np.mean((p - y) ** 2) / brier_ref) if brier_ref > 0 else float("nan")


def roc_auc(y, p):
    """Chance that a random 1-day gets a higher forecast than a random 0-day. 0.5 = coin flip."""
    y, p = clean_pair(y, p)
    return float("nan") if has_one_class(y) else float(roc_auc_score(y, p))


def pr_auc(y, p):
    """Average precision. A forecaster with no skill scores the base rate."""
    y, p = clean_pair(y, p)
    return float("nan") if has_one_class(y) else float(average_precision_score(y, p))


def lift(y, p, top_frac=0.2):
    """Share of 1s among the 20% highest forecasts, divided by the share of 1s overall."""
    y, p = clean_pair(y, p)
    if has_one_class(y) or y.mean() == 0:
        return float("nan")
    k = max(1, math.ceil(top_frac * len(y)))
    top = np.argsort(-p, kind="stable")[:k]
    return float(y[top].mean() / y.mean())


def calibration_slope_intercept(y, p):
    """Fit logit(P(y=1)) = a + b * logit(p). Perfect calibration is a = 0, b = 1."""
    y, p = clean_pair(y, p)
    if has_one_class(y) or len(y) < 10:
        return float("nan"), float("nan")
    x = logit(p).reshape(-1, 1)
    if np.ptp(x) < 1e-9:   # all forecasts are the same, so no slope can be estimated
        return float("nan"), float("nan")
    try:
        model = LogisticRegression(C=1e8, solver="lbfgs", max_iter=1000).fit(x, y)
        return float(model.intercept_[0]), float(model.coef_[0, 0])
    except Exception:
        return float("nan"), float("nan")


def reliability_table(y, p, n_bins=10):
    """For each probability bin: how many days, average forecast, how often y was really 1."""
    y, p = clean_pair(y, p)
    columns = ["bin", "n", "mean_p", "frac_pos"]
    if not len(y):
        return pd.DataFrame(columns=columns)
    edges = np.linspace(0, 1, n_bins + 1)
    bin_of = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        in_bin = bin_of == b
        if in_bin.any():
            rows.append({"bin": b, "n": int(in_bin.sum()),
                         "mean_p": float(p[in_bin].mean()), "frac_pos": float(y[in_bin].mean())})
    return pd.DataFrame(rows, columns=columns)


def ece(y, p, n_bins=10):
    """Expected calibration error: the average gap between forecast and reality, weighted by bin size."""
    table = reliability_table(y, p, n_bins)
    if table.empty:
        return float("nan")
    gap = (table["frac_pos"] - table["mean_p"]).abs()
    return float((table["n"] * gap).sum() / table["n"].sum())


def evaluate_all(y, p, p_ref):
    """All headline scores for one forecaster."""
    y_clean, p_clean = clean_pair(y, p)
    intercept, slope = calibration_slope_intercept(y, p)
    return {"n": int(len(y_clean)),
            "base_rate": float(y_clean.mean()) if len(y_clean) else float("nan"),
            "mean_p": float(p_clean.mean()) if len(p_clean) else float("nan"),
            "brier": brier(y, p),
            "brier_skill": brier_skill(y, p, p_ref),
            "log_loss": log_loss(y, p),
            "pr_auc": pr_auc(y, p),
            "lift": lift(y, p),
            "roc_auc": roc_auc(y, p),
            "cal_intercept": intercept,
            "cal_slope": slope,
            "ece": ece(y, p)}


def metrics_table(preds, y, p_ref):
    """One row per forecaster.

    Only days where EVERY forecaster has a value are used, so no forecaster
    looks better or worse just because it was scored on different days.
    """
    common = preds.dropna(how="any")
    rows = {name: evaluate_all(y.reindex(common.index), common[name], p_ref.reindex(common.index))
            for name in common.columns}
    return pd.DataFrame(rows).T
