"""Episodes: stretches of consecutive high-volatility days.

An episode is a run of days with y = 1. Runs that are less than `merge_gap` days apart are
joined, and runs shorter than `min_len` days are ignored.
An alarm is a day where the forecast is above a threshold.

To compare forecasters fairly, each gets its own threshold, chosen so that it raises a false alarm
on the same share (far) of calm days. This uses the test labels, so it is only a way to compare
forecasters. It is not a rule one could trade.
"""
import numpy as np
import pandas as pd


def find_episodes(y, min_len=5, merge_gap=10):
    """Start, end and length of each episode in a 0/1 series (missing counts as 0)."""
    y = y if isinstance(y, pd.Series) else pd.Series(np.asarray(y))
    values = y.fillna(0).to_numpy().astype(int)

    change = np.diff(np.r_[0, values, 0])
    starts = np.where(change == 1)[0]
    ends = np.where(change == -1)[0] - 1

    merged = []
    for s, e in zip(starts, ends):
        if merged and s - merged[-1][1] - 1 <= merge_gap:
            merged[-1][1] = int(e)   # close to the previous run, so join them
        else:
            merged.append([int(s), int(e)])
    merged = [m for m in merged if m[1] - m[0] + 1 >= min_len]

    return pd.DataFrame({"start_pos": [m[0] for m in merged], "end_pos": [m[1] for m in merged],
                         "start": [y.index[m[0]] for m in merged], "end": [y.index[m[1]] for m in merged],
                         "length": [m[1] - m[0] + 1 for m in merged]})


def threshold_for_far(y, p, far):
    """The probability level that is exceeded on about `far` of the calm (y = 0) days."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    keep = ~np.isnan(y) & ~np.isnan(p)
    calm_days = p[keep & (y == 0)]
    if len(calm_days) == 0:
        return float("nan")
    return float(np.quantile(calm_days, 1 - far))


def episode_report(y, p, far=0.10, pre_window=5, min_len=5, merge_gap=10):
    """How many episodes were detected, and how early.

    An episode counts as detected if there is an alarm from `pre_window` days before it
    until its end. lead_days = how many days before the first y = 1 day the first alarm came
    (negative = late).
    Returns {"summary": dict, "episodes": table}.
    """
    p = p.reindex(y.index)
    threshold = threshold_for_far(y.values, p.values, far)
    p_values = p.to_numpy(float)
    alarm = (p_values > threshold) & ~np.isnan(p_values)

    episodes = find_episodes(y, min_len, merge_gap)
    in_window = np.zeros(len(y), bool)       # days inside an episode window
    first_alarm_pos, lead_days = [], []
    for start, end in zip(episodes.start_pos, episodes.end_pos):
        window_start = max(0, start - pre_window)
        in_window[window_start:end + 1] = True
        alarm_days = np.where(alarm[window_start:end + 1])[0]
        if len(alarm_days):
            pos = window_start + alarm_days[0]
            first_alarm_pos.append(pos)
            lead_days.append(int(start - pos))
        else:
            first_alarm_pos.append(np.nan)
            lead_days.append(np.nan)

    episodes = episodes.copy()
    episodes["detected"] = ~pd.isna(first_alarm_pos)
    episodes["lead_days"] = lead_days
    episodes["first_alarm"] = [y.index[int(pos)] if not np.isnan(pos) else pd.NaT for pos in first_alarm_pos]

    y_values = y.fillna(0).to_numpy()
    calm = (y_values == 0) & ~np.isnan(p_values)
    alarms_outside = alarm & ~in_window
    false_alarm_clusters = int((np.diff(np.r_[0, alarms_outside.astype(int)]) == 1).sum())
    detected = episodes["detected"].to_numpy()

    summary = {"threshold": float(threshold),
               "n_episodes": int(len(episodes)),
               "recall": float(detected.mean()) if len(episodes) else float("nan"),
               "median_lead_days": float(np.nanmedian(lead_days)) if detected.any() else float("nan"),
               "mean_lead_days": float(np.nanmean(lead_days)) if detected.any() else float("nan"),
               "realized_far": float(alarm[calm].mean()) if calm.any() else float("nan"),
               "alarm_rate": float(alarm.mean()),
               "precision": float(y_values[alarm].mean()) if alarm.any() else float("nan"),
               "false_alarm_clusters": false_alarm_clusters}
    return {"summary": summary, "episodes": episodes}


def compare_episode_recall(y, forecasts, fars=(0.05, 0.10, 0.20)):
    """Table with one row per (forecaster, false alarm rate)."""
    rows = []
    for name, p in forecasts.items():
        for far in fars:
            summary = episode_report(y, p, far=far)["summary"]
            rows.append({"model": name, "target_far": far, **summary})
    return pd.DataFrame(rows).set_index(["model", "target_far"])
