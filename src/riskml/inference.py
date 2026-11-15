"""Is one model really better than another, or is it just luck?

Here "difference" always means: positive = the candidate model is better than the reference.

Daily forecasts for a horizon of h days overlap (neighbouring days share h-1 days of the future),
so the daily errors are not independent. Both tools below take that into account:
  * block bootstrap: resample blocks of consecutive days instead of single days
  * Diebold-Mariano test: uses a Newey-West variance that allows for correlated errors
"""
import numpy as np
import pandas as pd
from scipy import stats

from riskml.evaluate import brier, pr_auc

HIGHER_IS_BETTER = {"brier": False, "pr_auc": True}
METRICS = {"brier": brier, "pr_auc": pr_auc}


def block_length(h):
    """Bootstrap block length: at least 21 days, and at least twice the horizon."""
    return max(21, 2 * h)


def block_indices(n, block_len, n_boot, rng):
    """Row numbers for n_boot bootstrap samples, built from blocks of consecutive days.

    The blocks wrap around the end of the data (circular), so every day has the same chance.
    Returns an array of shape (n_boot, n).
    """
    n_blocks = int(np.ceil(n / block_len))
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    rows = (starts[:, :, None] + np.arange(block_len)[None, None, :]) % n
    return rows.reshape(n_boot, n_blocks * block_len)[:, :n]


def bootstrap_diff(y, p_model, p_ref, metric, block_len, n_boot, seed, alpha=0.05):
    """Bootstrap confidence interval for (model score - reference score), signed so that positive = model better."""
    score = METRICS[metric]
    sign = 1.0 if HIGHER_IS_BETTER[metric] else -1.0

    rng = np.random.default_rng(seed)
    rows = block_indices(len(y), block_len, n_boot, rng)
    diffs = np.array([sign * (score(y[r], p_model[r]) - score(y[r], p_ref[r])) for r in rows])
    diffs = diffs[~np.isnan(diffs)]

    low, high = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p_value = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    p_value = max(p_value, 1 / (len(diffs) + 1))
    point = sign * (score(y, p_model) - score(y, p_ref))
    return {"point": float(point), "ci_low": float(low), "ci_high": float(high),
            "p_value": float(p_value), "model_better_sig": bool(low > 0)}


def newey_west_variance(x, lags):
    """Variance of the mean of x when neighbouring values are correlated (Bartlett weights)."""
    n = len(x)
    lags = int(min(max(lags, 0), n - 1))
    centred = x - x.mean()
    total = centred @ centred / n
    for k in range(1, lags + 1):
        weight = 1 - k / (lags + 1)
        total += 2 * weight * (centred[k:] @ centred[:-k] / n)
    return max(float(total), 1e-18)


def diebold_mariano(loss_model, loss_ref, h):
    """Test whether the average loss of two forecasts differs (positive = model better).

    Includes the Harvey-Leybourne-Newbold small-sample correction.
    """
    d = loss_ref - loss_model
    n = len(d)
    if n < 10:
        raise ValueError("Too few observations for the Diebold-Mariano test.")
    lags = max(h - 1, int(np.floor(4 * (n / 100) ** (2 / 9))))
    std_error = np.sqrt(newey_west_variance(d, lags) / n)
    stat = d.mean() / std_error
    if h > 1:
        stat *= np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-12))
    p_two_sided = float(2 * stats.t(df=n - 1).sf(abs(stat)))
    return {"mean_diff": float(d.mean()), "p_two_sided": p_two_sided}


def compare_to_reference(y, p_model, p_ref, h, n_boot, seed):
    """Compare a model with the reference on Brier score and PR-AUC.

    Returns a table with a row for each of brier, pr_auc and brier_DM.
    The column 'criterion' is True only if BOTH bootstrap intervals are completely above zero.
    That is the success rule we chose before running the project.
    """
    y, p_model, p_ref = (np.asarray(v, dtype=float) for v in (y, p_model, p_ref))
    rows = {m: bootstrap_diff(y, p_model, p_ref, m, block_length(h), n_boot, seed) for m in METRICS}

    dm = diebold_mariano((p_model - y) ** 2, (p_ref - y) ** 2, h)
    table = pd.DataFrame(rows).T[["point", "ci_low", "ci_high", "p_value", "model_better_sig"]]
    table.loc["brier_DM", ["point", "p_value"]] = [dm["mean_diff"], dm["p_two_sided"]]
    table["criterion"] = bool(rows["brier"]["model_better_sig"] and rows["pr_auc"]["model_better_sig"])
    return table
