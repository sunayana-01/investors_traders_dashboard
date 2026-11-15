"""Checks that nothing from the future reaches the features, labels or training rows."""
import numpy as np
import pandas as pd

from helpers import WEIGHTS, make_aligned
from riskml.features import build_features
from riskml.labels import make_labels
from riskml.portfolio import asset_returns, make_portfolio, portfolio_returns
from riskml.splits import align_xy, make_folds, split_period

SHOCK_DAY = "2014-06-02"


def inputs(aligned):
    portfolio = make_portfolio(WEIGHTS)
    assets = asset_returns(aligned.prices)
    port_ret = portfolio_returns(assets, portfolio)
    X = build_features(port_ret, assets, aligned.vix, aligned.macro, portfolio.tickers)
    return port_ret, X


def test_features_do_not_change_when_later_data_changes():
    _, X_calm = inputs(make_aligned())
    _, X_shock = inputs(make_aligned(shock_from=SHOCK_DAY))
    before = X_calm.index[X_calm.index < pd.Timestamp(SHOCK_DAY)]
    assert len(before) > 1000
    pd.testing.assert_frame_equal(X_calm.loc[before], X_shock.loc[before], atol=1e-10, check_freq=False)


def test_threshold_only_uses_known_values():
    port_ret, _ = inputs(make_aligned())
    h = 21
    labels = make_labels(port_ret, h)
    # change the returns of the last 100 days: thresholds more than h days earlier must not move
    changed = port_ret.copy()
    changed.iloc[-100:] *= 5
    labels2 = make_labels(changed, h)
    safe = labels.index[:-100 - h]
    pd.testing.assert_series_equal(labels.loc[safe, "threshold"], labels2.loc[safe, "threshold"])


def test_last_h_rows_have_no_label():
    port_ret, _ = inputs(make_aligned())
    labels = make_labels(port_ret, 21)
    assert labels["y"].iloc[-21:].isna().all()


def test_purge_gap_and_holdout_separation():
    port_ret, X = inputs(make_aligned())
    labels = make_labels(port_ret, 21)
    X, y, extra = align_xy(X, labels)
    development, holdout = split_period(X.index, extra["label_end"], "2019-01-01")
    label_end = pd.to_datetime(extra["label_end"])

    assert not (development & holdout).any()
    assert (label_end[development] < pd.Timestamp("2019-01-01")).all()   # no development label sees the holdout

    for mask in (development, holdout):
        for fold in make_folds(X.index, extra["label_end"], mask, 756, 126, 0.2):
            first_test_day = X.index[fold.test_idx[0]]
            assert (label_end.iloc[fold.train_idx] < first_test_day).all()          # purge
            assert fold.train_idx.max() < fold.test_idx.min()                      # train before test
            assert not set(fold.fit_idx) & set(fold.calib_idx)                      # fit and calibration are separate
            assert (label_end.iloc[fold.fit_idx] < X.index[fold.calib_idx[0]]).all()   # with a gap between them
