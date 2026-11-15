import pandas as pd
import pytest

from helpers import FAST, WEIGHTS, make_aligned
from riskml.attribution import build_context, describe_flag, forward_risk_contrib
from riskml.pipeline import PipelineError, run_pipeline
from riskml.trust import abstain_flags, abstain_summary, coverage_risk_curve, ood_score, recent_calibration_error

import numpy as np


def test_end_to_end_and_holdout_guards():
    aligned = make_aligned()
    result = run_pipeline(WEIGHTS, 21, aligned=aligned, **FAST)
    assert result.preds.index.max() < pd.Timestamp("2019-01-01")        # development period only
    assert {"M1", "M2", "B1"} <= set(result.preds.columns)
    assert result.preds.notna().all().all() and ((result.preds > 0) & (result.preds < 1)).all().all()
    assert "Buy & hold" in result.overlay_stats.index and len(result.flags) == len(result.preds)
    assert result.meta["exploratory"]

    with pytest.raises(PipelineError):                                  # holdout needs an explicit permission
        run_pipeline(WEIGHTS, 21, aligned=aligned, period="holdout", **FAST)

    final = run_pipeline(WEIGHTS, 21, aligned=aligned, period="holdout", allow_holdout=True, **FAST)
    assert final.preds.index.min() >= pd.Timestamp("2019-01-01")          # only the holdout period
    assert final.meta["calibration"]["M1"]["source"] == "config (frozen)"  # uses the choice frozen in the config


def test_forecasts_before_a_shock_do_not_change():
    shock_day = "2015-01-01"
    calm, shocked = make_aligned(), make_aligned(shock_from=shock_day)
    first = run_pipeline(WEIGHTS, 21, aligned=calm, **FAST)
    # the calibration method is chosen using all folds, so fix it to compare fairly
    methods = {code: info["method"] for code, info in first.meta["calibration"].items()}
    a = run_pipeline(WEIGHTS, 21, aligned=calm, calibration=methods, **FAST)
    b = run_pipeline(WEIGHTS, 21, aligned=shocked, calibration=methods, **FAST)
    before = a.preds.index[a.preds.index < pd.Timestamp(shock_day)]
    assert len(before) > 200
    pd.testing.assert_frame_equal(a.preds.loc[before], b.preds.loc[before], atol=1e-9)


def test_cache_returns_the_same_result(tmp_path, monkeypatch):
    import riskml.pipeline as pipeline
    monkeypatch.setattr(pipeline, "project_path", lambda relative: tmp_path)
    aligned = make_aligned()
    settings = {**FAST, "use_cache": True}
    first = run_pipeline(WEIGHTS, 21, aligned=aligned, **settings)
    second = run_pipeline(WEIGHTS, 21, aligned=aligned, **settings)
    assert not first.meta["cache_hit"] and second.meta["cache_hit"]
    pd.testing.assert_frame_equal(first.preds, second.preds, check_freq=False)


def test_risk_contributions():
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2015-01-01", periods=400)
    assets = pd.DataFrame({"A": rng.normal(0, 0.02, 400), "B": rng.normal(0, 0.005, 400),
                           "C": rng.normal(0, 0.004, 400)}, index=days)
    weights = {"A": 0.4, "B": 0.3, "C": 0.3}
    shares = forward_risk_contrib(assets, weights, 21).dropna()
    assert np.allclose(shares.sum(axis=1), 1.0)                  # shares add up to 100%
    assert (shares["A"] > 0.6).all()                             # A is by far the most volatile
    assert describe_flag(False, 0) == "missed" and describe_flag(True, 8) == "flagged 8d early"


def test_dashboard_context():
    aligned = make_aligned()
    result = run_pipeline(WEIGHTS, 21, aligned=aligned, **FAST)
    context = build_context(result, aligned)
    assert sorted(context["assets"]) == ["GLD", "IEF", "SPY"]
    episodes = context["episodes"]
    if len(episodes):
        assert episodes["top_driver"].isin(context["assets"]).all()
        assert (episodes["start"] >= result.y.index.min()).all() and (episodes["end"] <= result.y.index.max()).all()


# trust layer

def trust_data(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2010-01-01", periods=n)
    X = pd.DataFrame(rng.normal(size=(n, 3)), index=days, columns=list("abc"))
    p = pd.Series(rng.uniform(0.05, 0.95, n), index=days)
    y = pd.Series((rng.uniform(size=n) < p).astype(float), index=days)
    members = pd.DataFrame({"m1": p, "m2": (p + rng.normal(0, 0.05, n)).clip(0, 1)}, index=days)
    return X, p, y, members


def test_ood_score_only_uses_the_past():
    X, *_ = trust_data()
    shocked = X.copy()
    shocked.iloc[1000:] += 100
    a, b = ood_score(X), ood_score(shocked)
    assert np.allclose(a.iloc[:1000].fillna(-1), b.iloc[:1000].fillna(-1))
    assert b.iloc[1000] > 4                                      # the shock itself is flagged


def test_calibration_error_ignores_unknown_outcomes():
    X, p, y, _ = trust_data()
    h = 21
    flipped = y.copy()
    flipped.iloc[-h:] = 1 - flipped.iloc[-h:]                    # the last h outcomes are not known yet
    a = recent_calibration_error(y, p, h, 504)
    b = recent_calibration_error(flipped, p, h, 504)
    assert np.allclose(a.iloc[:-h].fillna(-1), b.iloc[:-h].fillna(-1))


def test_flags_summary_and_coverage_curve():
    X, p, y, members = trust_data(4000)
    trust_cfg = {"disagreement_threshold": 0.25, "ood_z_threshold": 4.0,
                 "calibration_window_days": 504, "near_threshold_band": 0.05}
    flags = abstain_flags(p, members, X, 0.5, trust_cfg, y=y, h=21)
    assert {"disagree", "ood", "near_threshold", "high_cal_error", "abstain"} <= set(flags)
    assert (flags["abstain"] >= flags["near_threshold"]).all()
    assert 0 <= abstain_summary(y, p, flags)["abstain_rate"] <= 1

    certainty = -(p - 0.5).abs()                                 # far from 50% = sure, so uncertainty is lower
    curve = coverage_risk_curve(y, p, certainty)
    assert curve["brier"].iloc[0] < curve["brier"].iloc[-1] and curve["n"].iloc[-1] == len(y)


def test_rule_a_settings_change_the_overlay():
    aligned = make_aligned()
    default = run_pipeline(WEIGHTS, 21, aligned=aligned, **FAST)
    eager = run_pipeline(WEIGHTS, 21, aligned=aligned, rule_a_threshold=0.2, rule_a_exposure=0.2, **FAST)
    name = "Rule A [M1]"
    assert eager.meta["rule_a_threshold"] == 0.2 and eager.meta["rule_a_exposure"] == 0.2
    assert eager.overlay_stats.loc[name, "avg_exposure"] < default.overlay_stats.loc[name, "avg_exposure"]
    assert eager.preds.equals(default.preds)          # only the overlay changes, not the forecasts
