"""Walk-forward splits (training on the past, testing on the next block).

Each row of the dataset is one trading day t. Its features are known at the close
of day t and its label uses returns up to label_end (which is t + h days).

Rules that stop information from the future leaking into a model:
  * Purging: a training row is only used if its label was completely known before the
    test block starts (label_end < first test date). This removes about h rows
    right before each test block.
  * One model per test block. The model is fitted once and then used for the whole block.
  * Calibration: the last 20% of the training rows are kept apart to calibrate the
    probabilities. The model itself is fitted on the earlier rows only, with a gap of
    about h rows in between, so the fit labels and the calibration labels do not overlap.
  * Development vs holdout: development test rows must have label_end before the holdout
    date. The holdout is only touched once, at the very end of the project.
  * Never shuffled. Training rows always come before the test block.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

# what the models are allowed to see besides the features
TRAIN_EXTRA = ["fwd_rv", "threshold"]   # for training rows, their labels are complete
TEST_EXTRA = ["threshold"]              # the threshold is past-only, so known on the test day

MIN_CALIB_ROWS = 63      # at least about 3 months in the calibration slice
MIN_LAST_BLOCK = 21      # a test block shorter than this is merged into the block before it


@dataclass
class Fold:
    fold_id: int
    train_idx: np.ndarray    # all training rows (purged)
    fit_idx: np.ndarray      # rows the model is fitted on
    calib_idx: np.ndarray    # rows used to calibrate the probabilities
    test_idx: np.ndarray     # the test block
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def align_xy(X, labels):
    """Keep only the days that have both features and a complete label.

    Returns the features, the 0/1 labels, and an 'extra' table with
    fwd_rv, threshold and label_end.
    """
    labelled = labels["y"].notna() & labels["threshold"].notna() & labels["fwd_rv"].notna()
    days = X.index.intersection(labels.index[labelled.to_numpy()])
    y = labels.loc[days, "y"].astype(int)
    extra = labels.loc[days, ["fwd_rv", "threshold", "label_end"]].copy()
    return X.loc[days], y, extra


def split_period(dates, label_end, holdout_start):
    """True/False arrays marking the development rows and the holdout rows."""
    holdout_start = pd.Timestamp(holdout_start)
    label_end = pd.DatetimeIndex(pd.to_datetime(np.asarray(label_end)))
    dates = pd.DatetimeIndex(dates)
    development = (dates < holdout_start) & (label_end < holdout_start)
    holdout = dates >= holdout_start
    return np.asarray(development), np.asarray(holdout)


def make_folds(dates, label_end, test_mask, min_train, test_block, calib_frac):
    """Make the list of walk-forward folds for the test rows selected by test_mask."""
    dates = pd.DatetimeIndex(dates)
    label_end = pd.DatetimeIndex(pd.to_datetime(np.asarray(label_end)))
    if not dates.is_monotonic_increasing or dates.has_duplicates:
        raise ValueError("The dates must be sorted and unique.")
    if label_end.isna().any() or (label_end < dates).any():
        raise ValueError("label_end must exist and can not be before the row date.")

    dates_ns, label_end_ns = dates.to_numpy(), label_end.to_numpy()

    # for each day: how many rows have a label that was finished before that day.
    # These are the rows we may train on if a test block starts on that day.
    n_trainable = np.searchsorted(label_end_ns, dates_ns, side="left")

    eligible = np.flatnonzero(np.asarray(test_mask, bool) & (n_trainable >= min_train))
    if len(eligible) == 0:
        return []

    # cut the eligible days into runs without holes, then into blocks of test_block days
    runs = np.split(eligible, np.flatnonzero(np.diff(eligible) > 1) + 1)
    blocks = []
    for run in runs:
        run_blocks = [run[i:i + test_block] for i in range(0, len(run), test_block)]
        if len(run_blocks) > 1 and len(run_blocks[-1]) < MIN_LAST_BLOCK:
            run_blocks[-2] = np.concatenate([run_blocks[-2], run_blocks[-1]])   # merge a tiny last block
            run_blocks.pop()
        blocks.extend(run_blocks)

    folds = []
    for k, test_idx in enumerate(blocks):
        first_test = int(test_idx[0])
        n_train = int(n_trainable[first_test])

        n_calib = max(int(round(calib_frac * n_train)), MIN_CALIB_ROWS)
        calib_start = n_train - n_calib
        # the model is fitted only on rows whose label was finished before the calibration slice starts
        n_fit = int(np.searchsorted(label_end_ns, dates_ns[calib_start], side="left"))
        if n_fit < MIN_CALIB_ROWS:
            raise ValueError(f"Fold {k} has only {n_fit} rows to fit on. "
                             "Increase min_train_days or lower calibration_fraction.")

        folds.append(Fold(fold_id=k,
                          train_idx=np.arange(n_train),
                          fit_idx=np.arange(n_fit),
                          calib_idx=np.arange(calib_start, n_train),
                          test_idx=test_idx,
                          train_end=dates[n_train - 1],
                          test_start=dates[first_test],
                          test_end=dates[int(test_idx[-1])]))
    return folds


def describe_folds(folds, dates):
    """A small table with one row per fold (shown in the dashboard)."""
    dates = pd.DatetimeIndex(dates)
    rows = []
    for f in folds:
        rows.append({"fold": f.fold_id,
                     "train_start": dates[f.train_idx[0]], "train_end": f.train_end,
                     "n_train": len(f.train_idx), "n_fit": len(f.fit_idx), "n_calib": len(f.calib_idx),
                     "test_start": f.test_start, "test_end": f.test_end, "n_test": len(f.test_idx),
                     "gap_rows": int(f.test_idx[0] - f.train_idx[-1] - 1)})
    return pd.DataFrame(rows).set_index("fold")
