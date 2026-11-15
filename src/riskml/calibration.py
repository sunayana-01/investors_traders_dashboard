"""Calibration: making forecast probabilities match how often things really happen.

If a model says 30% on many days, about 30% of those days should end up with y = 1.
A calibrator learns a small correction from the calibration slice of each training window.

Three options:
  none      leave the probabilities as they are
  platt     fit a logistic regression on logit(p)
  isotonic  fit a step-shaped increasing curve (more flexible, needs more data)

The option for each model is chosen on the development period only. It is then
frozen in the config before the holdout is touched.
"""
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from riskml.evaluate import brier, logit

EPS = 1e-6
METHODS = ["none", "platt", "isotonic"]   # from simplest to most flexible
MIN_SAMPLES = 60                          # with fewer calibration rows we do not calibrate


class Calibrator:
    """If the calibration data is too small or has one class only, it does nothing
    (the identity) and says why in fallback_reason, so a thin fold never crashes a run."""

    def __init__(self, method="none"):
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")
        self.method = method
        self.fallback_reason = ""
        self.model = None

    def fit(self, p, y):
        p = np.asarray(p, dtype=float)
        y = np.asarray(y, dtype=float)
        keep = np.isfinite(p) & np.isfinite(y)
        p, y = p[keep], y[keep].astype(int)

        self.model = None
        self.fallback_reason = ""
        if self.method == "none":
            return self
        if len(y) < MIN_SAMPLES:
            self.fallback_reason = f"only {len(y)} calibration samples"
        elif y.min() == y.max():
            self.fallback_reason = "only one class in the calibration slice"
        elif self.method == "platt":
            x = logit(p).reshape(-1, 1)
            if np.ptp(x) < 1e-9:
                self.fallback_reason = "all raw forecasts are equal"
            else:
                self.model = LogisticRegression(C=1e4, max_iter=1000).fit(x, y)
        else:
            self.model = IsotonicRegression(increasing=True, y_min=EPS, y_max=1 - EPS,
                                            out_of_bounds="clip").fit(p, y)
        return self

    def transform(self, p):
        p = np.asarray(p, dtype=float)
        out = p.copy()
        ok = np.isfinite(p)
        if self.model is not None and ok.any():
            if self.method == "platt":
                out[ok] = self.model.predict_proba(logit(p[ok]).reshape(-1, 1))[:, 1]
            else:
                out[ok] = self.model.predict(p[ok])
        out[ok] = np.clip(out[ok], EPS, 1 - EPS)
        return out


def calibrate_fold(p_calib, y_calib, p_test, method):
    """Learn the correction on the calibration slice, apply it to the test block."""
    calibrator = Calibrator(method).fit(p_calib, y_calib)
    return calibrator.transform(p_test), calibrator


def calibration_split(n, frac=0.2, gap=21, min_calib=MIN_SAMPLES):
    """Split n time-ordered rows into fit rows and calibration rows with a gap in between.

    Only used for the live forecast. The walk-forward folds do this split themselves.
    """
    n_calib = max(int(round(n * frac)), min_calib)
    calib_start = n - n_calib
    fit_end = calib_start - gap
    if fit_end <= 0:
        raise ValueError(f"{n} rows are too few for {n_calib} calibration rows plus a gap of {gap}")
    return np.arange(0, fit_end), np.arange(calib_start, n)


def select_method(fold_data, candidates=METHODS, min_improvement=0.005):
    """Pick the calibration method of ONE model, using the development folds.

    fold_data is a list of (p_calib, y_calib, p_test, y_test), one per fold.
    Each method is scored by the Brier score of all calibrated test forecasts together.

    We start with the simplest method and only move to a more flexible one if it
    improves the Brier score by at least 0.5%. Small gains are probably just noise.
    Returns the chosen method and a table with the scores.
    """
    methods = [m for m in METHODS if m in candidates]
    scores = {}
    for method in methods:
        calibrated, outcomes = [], []
        for p_calib, y_calib, p_test, y_test in fold_data:
            p_new, _ = calibrate_fold(p_calib, y_calib, p_test, method)
            calibrated.append(p_new)
            outcomes.append(np.asarray(y_test, dtype=float))
        scores[method] = {"brier": brier(np.concatenate(outcomes), np.concatenate(calibrated)),
                          "n_test": int(sum(len(o) for o in outcomes))}
    table = pd.DataFrame(scores).T

    chosen = methods[0]
    for method in methods[1:]:
        if table.loc[method, "brier"] < table.loc[chosen, "brier"] * (1 - min_improvement):
            chosen = method
    return chosen, table
