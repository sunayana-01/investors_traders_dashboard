import numpy as np
import pandas as pd
import pytest

from helpers import WEIGHTS, make_aligned
from riskml.calibration import Calibrator, select_method
from riskml.ensemble import build_ensembles
from riskml.episodes import find_episodes
from riskml.evaluate import brier, brier_skill, ece, expanding_base_rate, roc_auc
from riskml.inference import block_indices, compare_to_reference
from riskml.models import MODEL_NAMES, build_model
from riskml.overlay import run_overlay
from riskml.pipeline import prepare_inputs
from riskml.portfolio import make_portfolio


def test_brier_and_skill_known_values():
    y = np.array([1, 0, 1, 0])
    assert np.isclose(brier(y, [1, 0, 1, 0]), 0.0)
    assert np.isclose(brier(y, [0.5] * 4), 0.25)
    assert np.isclose(brier_skill(y, [0.5] * 4, [0.5] * 4), 0.0)
    assert roc_auc(y, [0.9, 0.1, 0.8, 0.2]) == 1.0


def test_ece_is_small_for_honest_forecasts():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 20000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    assert ece(y, p) < 0.02


def test_expanding_base_rate_only_uses_known_labels():
    y = pd.Series([1, 1, 0, 0, 1, 1], dtype=float)
    ref = expanding_base_rate(y, h=2, prior=0.25)
    assert ref.iloc[0] == 0.25 and ref.iloc[1] == 0.25   # nothing known yet
    assert ref.iloc[2] == 1.0                            # only y[0] is known on day 2


def test_every_model_returns_valid_probabilities():
    aligned = make_aligned()
    h = 21
    port_ret, _, _, X, y, extra = prepare_inputs(make_portfolio(WEIGHTS), aligned, h,
                                                 __import__("riskml.config", fromlist=["x"]).get_config())
    train, test = slice(0, 1500), slice(1500, 1700)
    for code in MODEL_NAMES:
        model = build_model(code, port_ret, h)
        model.fit(X.iloc[train], y.iloc[train], extra.iloc[train][["fwd_rv", "threshold"]])
        p = model.predict_proba(X.iloc[test], extra.iloc[test][["threshold"]])
        assert len(p) == 200 and np.isfinite(p).all() and ((p > 0) & (p < 1)).all(), code


def test_calibrator_falls_back_when_data_is_thin():
    calibrator = Calibrator("isotonic").fit(np.random.rand(20), np.random.randint(0, 2, 20))
    assert calibrator.fallback_reason
    p = np.array([0.2, 0.7])
    assert np.allclose(calibrator.transform(p), p)


def test_select_method_keeps_simplest_when_nothing_helps():
    rng = np.random.default_rng(1)
    folds = []
    for _ in range(4):
        p_calib, p_test = rng.uniform(0.1, 0.9, 300), rng.uniform(0.1, 0.9, 300)
        folds.append((p_calib, (rng.uniform(size=300) < p_calib).astype(int),
                      p_test, (rng.uniform(size=300) < p_test).astype(int)))
    method, _ = select_method(folds)
    assert method == "none"


def test_ensemble_is_between_members_and_needs_two():
    days = pd.bdate_range("2020-01-01", periods=5)
    preds = pd.DataFrame({"M1": [0.1] * 5, "M2": [0.5] * 5, "M3": [0.9] * 5, "M4": [0.5] * 5}, index=days)
    ensembles, used = build_ensembles(preds)
    assert (ensembles["E-core"] > 0.1).all() and (ensembles["E-core"] < 0.9).all()
    assert used["E-ML"] == ["M2", "M3"]
    assert build_ensembles(preds[["M1"]])[0].shape[1] == 0


def test_find_episodes_merges_close_runs_and_drops_short_ones():
    y = pd.Series([0] * 3 + [1] * 6 + [0] * 4 + [1] * 6 + [0] * 30 + [1] * 2 + [0] * 5)
    episodes = find_episodes(y, min_len=5, merge_gap=10)
    assert len(episodes) == 1 and episodes.loc[0, "length"] == 16   # the two runs are joined, the 2-day run is dropped


def test_bootstrap_blocks_stay_in_range():
    rows = block_indices(100, 10, 50, np.random.default_rng(0))
    assert rows.shape == (50, 100) and rows.min() >= 0 and rows.max() < 100


def test_better_model_gets_positive_difference():
    rng = np.random.default_rng(0)
    p_true = rng.uniform(0.05, 0.95, 1500)
    y = (rng.uniform(size=1500) < p_true).astype(int)
    good = p_true
    bad = np.full(1500, y.mean())
    table = compare_to_reference(y, good, bad, h=1 + 1, n_boot=100, seed=0)
    assert table.loc["brier", "point"] > 0 and table.loc["pr_auc", "point"] > 0


def test_overlay_waits_before_acting():
    days = pd.bdate_range("2020-01-01", periods=10)
    returns = pd.Series(0.001, index=days)
    p = pd.Series(0.0, index=days)
    p.iloc[3] = 1.0                                                  # alarm at the close of day 3
    path = run_overlay(returns, p, "A", cost_bps=0, extra_delay=1, threshold=0.5, cut=0.5)
    assert path["exposure"].loc[days[4]] == 1.0                      # next day: still fully invested
    assert path["exposure"].loc[days[5]] == 0.5                      # signal day + 2 (1 normal + 1 extra delay)
    fast = run_overlay(returns, p, "A", cost_bps=0, extra_delay=0, threshold=0.5, cut=0.5)
    assert fast["exposure"].loc[days[4]] == 0.5                      # without the extra delay: signal day + 1


def test_overlay_comparison_stops_where_the_forecasts_stop():
    from riskml.overlay import compare_overlays

    days = pd.bdate_range("2020-01-01", periods=300)
    returns = pd.Series(np.random.default_rng(0).normal(0.0004, 0.01, 300), index=days)
    p = pd.Series(0.3, index=days[100:200])                           # forecasts exist for days 100 to 199 only
    rules = {"Rule A [X]": {"signal": "X", "rule": "A", "threshold": 0.5, "cut": 0.5}}
    stats, paths = compare_overlays(returns, {"X": p}, rules)
    for path in paths.values():
        assert path.index[0] == days[100] and path.index[-1] == days[199]


def test_drawdown_limit_sends_the_strategy_to_cash_for_good():
    days = pd.bdate_range("2020-01-01", periods=60)
    returns = pd.Series(0.001, index=days)
    returns.iloc[10:15] = -0.03                                       # a sharp fall of about 14%
    p = pd.Series(0.0, index=days)                                    # the forecast never asks for a cut
    safe = run_overlay(returns, p, "A", cost_bps=0, extra_delay=0, threshold=0.5, cut=0.5, drawdown_limit=0.10)
    assert safe["exposure"].iloc[-1] == 0.0 and (safe["exposure"].iloc[15:] == 0.0).all()   # stays in cash
    assert (safe["exposure"].iloc[:10] == 1.0).all()                  # fully invested until the limit is hit
    unlimited = run_overlay(returns, p, "A", cost_bps=0, extra_delay=0, threshold=0.5, cut=0.5)
    assert (unlimited["exposure"] == 1.0).all()
