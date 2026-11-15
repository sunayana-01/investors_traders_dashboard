"""Trust layer: when should we NOT rely on a forecast?

A day gets a "low confidence" flag (abstain) if any of these is true:
  disagree        the models differ a lot (highest minus lowest probability)
  ood             an input is unusual compared with its own past ("out of distribution")
  near_threshold  the forecast is close to the Rule A threshold, so a small change flips the decision
  high_cal_error  recently the forecasts were too high or too low on average

Everything uses past information only. The outcome of day t is only known h days later,
so the calibration check shifts the outcomes by h days.
"""
import numpy as np
import pandas as pd

CAL_ERR_THRESHOLD = 0.15   # average forecast and average outcome differ by more than 15 points
OOD_MIN_OBS = 252          # need a year of history before judging what is unusual


def disagreement(member_probs):
    """Highest minus lowest member probability on each day."""
    return (member_probs.max(axis=1) - member_probs.min(axis=1)).rename("disagreement")


def ood_score(X, min_obs=OOD_MIN_OBS):
    """Largest z-score over all features. The mean and std only use rows BEFORE today (shift by 1)."""
    mean = X.expanding(min_periods=min_obs).mean().shift(1)
    std = X.expanding(min_periods=min_obs).std().shift(1).replace(0, np.nan)
    z = ((X - mean) / std).abs()
    return z.max(axis=1).rename("ood_z")


def recent_calibration_error(y, p, h, window):
    """|average forecast - average outcome| over the last `window` days, using known outcomes only.

    The outcome of day t is known at day t+h, so on day t the newest usable pair is the one from day t-h.
    """
    pairs = pd.DataFrame({"y": y.reindex(p.index), "p": p}).shift(h).dropna()
    min_periods = window // 4
    gap = (pairs["p"].rolling(window, min_periods=min_periods).mean()
           - pairs["y"].rolling(window, min_periods=min_periods).mean()).abs()
    return gap.reindex(p.index).rename("cal_err")


def abstain_flags(p_signal, member_probs, X, threshold, trust_cfg, y=None, h=21):
    """True/False table with one column per trigger and a final column 'abstain' (any trigger)."""
    flags = pd.DataFrame(index=p_signal.index)
    flags["disagree"] = disagreement(member_probs).reindex(p_signal.index) > trust_cfg["disagreement_threshold"]
    flags["ood"] = ood_score(X).reindex(p_signal.index) > trust_cfg["ood_z_threshold"]
    flags["near_threshold"] = (p_signal - threshold).abs() < trust_cfg["near_threshold_band"]
    if y is not None:
        error = recent_calibration_error(y, p_signal, h, trust_cfg["calibration_window_days"])
        flags["high_cal_error"] = error > CAL_ERR_THRESHOLD
    else:
        flags["high_cal_error"] = False   # no outcome history (live forecast), so this trigger is off
    flags = flags.fillna(False).astype(bool)   # a missing input means the trigger stays off
    flags["abstain"] = flags.any(axis=1)
    return flags


def coverage_risk_curve(y, p, uncertainty, coverages=np.linspace(0.3, 1.0, 15)):
    """Brier score when we only keep the days we are most sure about.

    Days are sorted by uncertainty (lowest first). Coverage 0.5 keeps the surest half.
    """
    data = pd.concat([y.rename("y"), p.rename("p"), uncertainty.rename("u")], axis=1).dropna()
    data = data.sort_values("u", kind="mergesort")
    squared_error = ((data["p"] - data["y"]) ** 2).to_numpy()
    rows = []
    for coverage in coverages:
        k = max(int(round(coverage * len(data))), 1)
        rows.append((float(coverage), k, float(squared_error[:k].mean())))
    return pd.DataFrame(rows, columns=["coverage", "n", "brier"])


def abstain_summary(y, p, flags):
    """Share of flagged days and the Brier score on kept days, flagged days and all days."""
    data = pd.concat([y.rename("y"), p.rename("p"), flags["abstain"].rename("a")], axis=1).dropna()
    squared_error = (data["p"] - data["y"]) ** 2
    flagged = data["a"].astype(bool)
    return {"abstain_rate": float(flagged.mean()),
            "brier_kept": float(squared_error[~flagged].mean()) if (~flagged).any() else np.nan,
            "brier_abstained": float(squared_error[flagged].mean()) if flagged.any() else np.nan,
            "brier_all": float(squared_error.mean())}
