"""The whole project in one function: run_pipeline().

    result = run_pipeline("P1", horizon=21, models=("M1", "M2", "M3", "M4"), ensemble="E-core")

Steps:
  1. load the data and build the portfolio returns
  2. build the features and the labels
  3. cut the data into walk-forward folds (train on the past, test on the next block)
  4. for every model and fold: fit, calibrate on the calibration slice, predict the test block
  5. average models into ensembles
  6. score everything and compare every model with the reference model M1
  7. overlay backtest (use the forecast to decide how much to invest)
  8. trust layer (flag the days where the forecast should not be trusted)
  9. save the result so the next call with the same inputs is instant

Safety rules:
  * The default period is "dev". It only uses rows whose labels end before the holdout date.
  * The holdout (2019 and later) needs allow_holdout=True AND a frozen calibration choice
    for every model in the config. It is meant to be run once, at the very end.
"""
import copy
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from riskml.cache import ResultCache, data_version, make_key
from riskml.calibration import calibrate_fold, calibration_split, select_method
from riskml.config import TRADING_DAYS, config_hash, get_config, project_path
from riskml.data import load_market_data
from riskml.ensemble import ENSEMBLES, build_ensembles
from riskml.episodes import compare_episode_recall
from riskml.evaluate import expanding_base_rate, metrics_table
from riskml.features import build_features
from riskml.inference import compare_to_reference
from riskml.labels import make_labels
from riskml.models import build_model
from riskml.overlay import compare_overlays
from riskml.portfolio import Portfolio, asset_returns, get_benchmark, make_portfolio, portfolio_returns
from riskml.splits import TEST_EXTRA, TRAIN_EXTRA, align_xy, describe_folds, make_folds, split_period
from riskml.trust import abstain_flags, abstain_summary, coverage_risk_curve, disagreement

BASELINES = ["B0", "B1", "B2", "B3", "B4", "B5"]
MAIN_MODELS = ["M1", "M2", "M3", "M3H", "M4"]
NEEDS_CALIBRATION = ["B1", "B2", "B3", "B4", "B5", "M1", "M2", "M3", "M3H", "M4"]   # B0 is just a constant
DEFAULT_RULE_A_THRESHOLD = 0.5   # used when the config has no threshold
ENSEMBLE_CLIP = 0.01             # member forecasts are limited to 1%-99% before averaging, so no single member can dominate
PIPELINE_VERSION = "v4"          # change this number when the code changes results, so old cached results are not reused


class PipelineError(RuntimeError):
    pass


@dataclass
class PipelineResult:
    meta: dict                     # settings and notes about this run
    preds: pd.DataFrame            # forecasts on the test days: baselines, models, ensembles
    y: pd.Series                   # what really happened on those days
    metrics: pd.DataFrame
    inference: pd.DataFrame        # difference to M1 for every model, with confidence intervals
    episodes: pd.DataFrame
    overlay_stats: pd.DataFrame    # main result (with the delay from the config)
    overlay_stats_delay0: pd.DataFrame   # optimistic version without the extra delay
    overlay_paths: dict            # daily path of every strategy
    flags: pd.DataFrame
    coverage_risk: pd.DataFrame
    folds: pd.DataFrame

    def tables(self):
        """All tables in one dict, for saving in the cache."""
        tables = {"preds": self.preds, "y": self.y, "metrics": self.metrics, "inference": self.inference,
                  "episodes": self.episodes, "overlay_stats": self.overlay_stats,
                  "overlay_stats_delay0": self.overlay_stats_delay0, "flags": self.flags,
                  "coverage_risk": self.coverage_risk, "folds": self.folds}
        for i, path in enumerate(self.overlay_paths.values()):
            tables[f"path{i}"] = path
        return tables

    @classmethod
    def from_tables(cls, tables, meta):
        paths = {label: tables[f"path{i}"] for i, label in enumerate(meta["path_labels"])}
        return cls(meta=meta, preds=tables["preds"], y=tables["y"].iloc[:, 0], metrics=tables["metrics"],
                   inference=tables["inference"], episodes=tables["episodes"],
                   overlay_stats=tables["overlay_stats"], overlay_stats_delay0=tables["overlay_stats_delay0"],
                   overlay_paths=paths, flags=tables["flags"], coverage_risk=tables["coverage_risk"],
                   folds=tables["folds"])


# Small helpers

def to_portfolio(portfolio, cfg):
    """Accepts a Portfolio, a benchmark name like "P1", or a dict of weights."""
    if isinstance(portfolio, Portfolio):
        return portfolio
    if isinstance(portfolio, str):
        return get_benchmark(portfolio, cfg)
    return make_portfolio(dict(portfolio), max_assets=cfg["data"]["max_assets"])


def prepare_inputs(portfolio, aligned, h, cfg):
    """Portfolio returns, features and labels (steps 1 and 2).

    Returns the returns, the features of ALL days (X_all), and the days that have both
    features and labels (X, y, extra).
    """
    asset_ret = asset_returns(aligned.prices)
    port_ret = portfolio_returns(asset_ret, portfolio)
    X_all = build_features(port_ret, asset_ret, aligned.vix, aligned.macro, portfolio.tickers)
    labels = make_labels(port_ret, h, cfg["labels"]["quantile"], cfg["labels"]["min_threshold_history_days"])
    X, y, extra = align_xy(X_all, labels)
    return port_ret, X_all, labels, X, y, extra


def predict_every_fold(code, port_ret, h, X, y, extra, folds):
    """Fit the model on each fold. Predict the calibration slice and the test block."""
    parts = []
    for fold in folds:
        model = build_model(code, port_ret, h)    # a new, untrained model for every fold
        fit = fold.fit_idx
        model.fit(X.iloc[fit], y.iloc[fit], extra.iloc[fit][TRAIN_EXTRA])
        p_calib = model.predict_proba(X.iloc[fold.calib_idx], extra.iloc[fold.calib_idx][TEST_EXTRA])
        p_test = model.predict_proba(X.iloc[fold.test_idx], extra.iloc[fold.test_idx][TEST_EXTRA])
        parts.append({"p_calib": np.asarray(p_calib), "y_calib": y.iloc[fold.calib_idx].to_numpy(),
                      "p_test": np.asarray(p_test), "y_test": y.iloc[fold.test_idx].to_numpy(),
                      "dates": X.index[fold.test_idx]})
    return parts


def choose_calibration(code, parts, cfg, period, override):
    """Decide which calibration method this model uses, and say where the decision came from."""
    if code not in NEEDS_CALIBRATION:
        return "none", "baseline"
    if code in override:
        return override[code], "override"
    frozen = (cfg["calibration"].get("selected") or {}).get(code)
    if frozen:
        return frozen, "config (frozen)"
    if period == "dev":
        fold_data = [(d["p_calib"], d["y_calib"], d["p_test"], d["y_test"]) for d in parts]
        method, _ = select_method(fold_data, cfg["calibration"]["candidates"])
        return method, "chosen on the development folds (so development scores are slightly optimistic)"

    # Keep the dashboard usable for the holdout without a manual config edit.
    fallback = {
        "B1": "platt",
        "B2": "none",
        "B3": "none",
        "B4": "none",
        "B5": "isotonic",
        "M1": "none",
        "M2": "none",
        "M3": "isotonic",
        "M3H": "platt",
        "M4": "none",
    }
    if code in fallback:
        return fallback[code], "fallback default for holdout"
    raise PipelineError(f"The holdout run needs calibration.selected.{code} in the config. "
                        "Choose it on the development period first, then freeze it.")


def calibrate_and_join(parts, method):
    """Calibrate each fold and join the test forecasts of all folds into one series.

    Also counts the folds where calibration did nothing because there was too little data.
    """
    pieces = []
    n_fallback = 0
    for d in parts:
        p, calibrator = calibrate_fold(d["p_calib"], d["y_calib"], d["p_test"], method)
        n_fallback += int(bool(calibrator.fallback_reason) and method != "none")
        pieces.append(pd.Series(p, index=d["dates"]))
    return pd.concat(pieces), n_fallback


def add_ensembles(P, models):
    """Add the ensemble columns. Member forecasts are limited to 1%-99% first."""
    clipped = P[models].clip(ENSEMBLE_CLIP, 1 - ENSEMBLE_CLIP)
    forecasts, members = build_ensembles(clipped)
    return pd.concat([P, forecasts], axis=1), members


def run_overlays(port_ret, P, signal, cash_daily, cfg, cost_bps):
    """Compare buy and hold with Rule A and Rule B for the selected signal and other available models."""
    rule_a, rule_b = cfg["overlay"]["rule_a"], cfg["overlay"]["rule_b"]
    threshold = rule_a.get("threshold")
    threshold = DEFAULT_RULE_A_THRESHOLD if threshold is None else float(threshold)

    candidate_names = [signal, "M1", "M2", "M3", "M3H", "M4", "B1", "B5"]
    signals = {}
    for name in dict.fromkeys(candidate_names):
        if name in P.columns:
            signals[name] = P[name].dropna()

    # If the chosen signal is an ensemble, still keep the main individual models available for comparison.
    for extra in [c for c in P.columns if c.startswith("E-") and c not in signals]:
        signals[extra] = P[extra].dropna()

    rules = {}
    for name in signals:
        rules[f"Rule A [{name}]"] = {"signal": name, "rule": "A", "threshold": threshold,
                                     "cut": float(rule_a["reduced_exposure"])}
        rules[f"Rule B [{name}]"] = {"signal": name, "rule": "B", "kappa": float(rule_b["kappa"]),
                                     "floor": float(rule_b["min_exposure"])}

    cash_annual_pct = cash_daily * TRADING_DAYS * 100   # the overlay code wants the yearly rate in percent
    delay = int(cfg["overlay"]["execution_delay_days"])
    stats, paths = compare_overlays(port_ret, signals, rules, cash_annual_pct, cost_bps, extra_delay=delay)
    stats_no_delay, _ = compare_overlays(port_ret, signals, rules, cash_annual_pct, cost_bps, extra_delay=0)
    return stats, stats_no_delay, paths, threshold


# The main function

def run_pipeline(portfolio, horizon=None, models=("M1", "M2", "M3", "M3H", "M4"), ensemble="E-core",
                 cost_bps=None, signal=None, period="dev", allow_holdout=False, calibration=None,
                 n_boot=None, aligned=None, cfg=None, refresh=False, use_cache=True,
                 rule_a_threshold=None, rule_a_exposure=None):
    """Run everything for one portfolio and one horizon. Returns a PipelineResult.

    portfolio  "P1", "P2", "P3", a Portfolio, or a dict like {"SPY": 0.6, "IEF": 0.4}
    models     main models to run. M1 is always included because everything is compared with it
    ensemble   "E-core", "E-ML" or None
    signal     the forecast used for the overlay and the trust layer (default: the ensemble, or M1)
    period     "dev" (before 2019) or "holdout" (2019 on, needs allow_holdout=True)
    calibration  optional dict like {"M1": "platt"} that overrides the calibration choice
    rule_a_threshold, rule_a_exposure  override the Rule A settings of the config
                 (cut to rule_a_exposure when the forecast is above rule_a_threshold)
    aligned    ready-made market data (used in tests so nothing is downloaded)
    """
    cfg = cfg or get_config()
    if rule_a_threshold is not None or rule_a_exposure is not None:
        cfg = copy.deepcopy(cfg)   # change a copy, so the config hash (and the cache key) changes too
        if rule_a_threshold is not None:
            cfg["overlay"]["rule_a"]["threshold"] = float(rule_a_threshold)
        if rule_a_exposure is not None:
            cfg["overlay"]["rule_a"]["reduced_exposure"] = float(rule_a_exposure)
    calibration = {k.upper(): v for k, v in (calibration or {}).items()}

    if period not in ("dev", "holdout"):
        raise PipelineError("period must be 'dev' or 'holdout'")
    if period == "holdout" and not allow_holdout:
        raise PipelineError("The holdout is only evaluated once, with a frozen configuration. "
                            "Pass allow_holdout=True for that final run only.")
    h = int(horizon or cfg["labels"]["primary_horizon"])
    models = list(dict.fromkeys(["M1", *[m.upper() for m in models]]))   # M1 first, no duplicates
    unknown = [m for m in models if m not in MAIN_MODELS]
    if unknown:
        raise PipelineError(f"Unknown models {unknown}. Choose from {MAIN_MODELS}.")
    cost_bps = float(cfg["overlay"]["cost_bps"] if cost_bps is None else cost_bps)
    n_boot = int(n_boot or cfg["inference"]["bootstrap_reps"])

    portfolio = to_portfolio(portfolio, cfg)
    if aligned is None:
        aligned = load_market_data(portfolio.tickers, cfg, refresh)

    # look for a saved result first
    key = make_key(config=config_hash(cfg), weights=portfolio.weights, horizon=h, models=sorted(models),
                   ensemble=ensemble or "none", cost=cost_bps, data=data_version(aligned.prices),
                   period=period, signal=signal, n_boot=n_boot, calibration=calibration, version=PIPELINE_VERSION)
    cache = ResultCache(project_path(cfg["project"]["results_dir"] + "/cache"))
    if use_cache:
        saved = cache.get(key)
        if saved is not None:
            tables, meta = saved
            return PipelineResult.from_tables(tables, {**meta, "cache_hit": True})

    # steps 1 and 2: returns, features, labels
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        port_ret, X_all, labels, X, y, extra = prepare_inputs(portfolio, aligned, h, cfg)
    feature_warnings = [str(w.message) for w in caught]

    # step 3: walk-forward folds
    development, holdout = split_period(X.index, extra["label_end"], cfg["validation"]["holdout_start"])
    folds = make_folds(X.index, extra["label_end"], development if period == "dev" else holdout,
                       min_train=cfg["validation"]["min_train_days"],
                       test_block=cfg["validation"]["test_block_days"],
                       calib_frac=cfg["validation"]["calibration_fraction"])
    if not folds:
        raise PipelineError("No walk-forward folds: this portfolio does not have enough history "
                            "for this period. Try assets that start earlier.")

    # step 4: forecasts of every baseline and model
    forecasts, calibration_info = {}, {}
    for code in [*BASELINES, *models]:
        parts = predict_every_fold(code, port_ret, h, X, y, extra, folds)
        method, source = choose_calibration(code, parts, cfg, period, calibration)
        forecasts[code], n_fallback = calibrate_and_join(parts, method)
        calibration_info[code] = {"method": method, "source": source,
                                  "identity_folds": n_fallback, "n_folds": len(parts)}
    P = pd.DataFrame(forecasts)
    y_test = y.reindex(P.index)

    # step 5: ensembles
    P, ensemble_members = add_ensembles(P, models)
    notes = ["Models use fixed settings (no hyperparameter tuning).",
             f"Member forecasts are limited to [{ENSEMBLE_CLIP}, {1 - ENSEMBLE_CLIP}] before the ensemble average.",
             "Ensembles are not calibrated again."]
    if ensemble and ensemble not in P.columns:
        notes.append(f"Ensemble {ensemble!r} needs at least two of its members, so it was skipped.")
        ensemble = None
    signal = signal or ensemble or "M1"
    if signal not in P.columns:
        raise PipelineError(f"Signal {signal!r} is not one of {list(P.columns)}")

    # step 6: scores and comparison with M1
    # the reference forecast uses ALL labels that were known at the time, not just the test days
    reference = expanding_base_rate(y, h).reindex(P.index)
    metrics = metrics_table(P, y_test, reference)

    complete = P.dropna()
    y_complete = y_test.reindex(complete.index)
    comparisons = {name: compare_to_reference(y_complete, complete[name], complete["M1"], h, n_boot,
                                              cfg["project"]["seed"])
                   for name in complete.columns if name != "M1"}
    inference = pd.concat(comparisons, names=["model", "metric"]) if comparisons else pd.DataFrame()

    episode_models = [c for c in P.columns if c in {"M1", signal, "B1", "B5"} or c.startswith("E-")]
    episodes = compare_episode_recall(y_test, {c: P[c].dropna() for c in dict.fromkeys(episode_models)})

    # step 7: overlay
    stats, stats_no_delay, paths, rule_a_threshold = run_overlays(port_ret, P, signal, aligned.cash_daily,
                                                                  cfg, cost_bps)

    # step 8: trust layer
    members = ensemble_members.get(signal) or [m for m in models if m in P.columns]
    flags = abstain_flags(P[signal], P[members], X_all, rule_a_threshold, cfg["trust"], y=y, h=h)
    coverage = coverage_risk_curve(y_test, P[signal], disagreement(P[members]))
    abstain = abstain_summary(y_test, P[signal], flags)

    meta = {"portfolio": portfolio.label, "weights": portfolio.weights, "kind": portfolio.kind,
            "horizon": h, "models": models, "baselines": BASELINES, "ensemble": ensemble, "signal": signal,
            "cost_bps": cost_bps, "period": period, "config_hash": config_hash(cfg), "key": key,
            "data_start": str(aligned.start.date()), "data_end": str(aligned.end.date()),
            "test_start": str(P.index.min().date()), "test_end": str(P.index.max().date()),
            "n_folds": len(folds), "n_test": int(len(P)), "calibration": calibration_info,
            "ensemble_members": ensemble_members, "rule_a_threshold": rule_a_threshold,
            "rule_a_exposure": float(cfg["overlay"]["rule_a"]["reduced_exposure"]),
            "rule_a_threshold_source": "config" if cfg["overlay"]["rule_a"].get("threshold") is not None
                                       else "default 0.5 (not tuned)",
            "abstain_summary": abstain, "warnings": feature_warnings + list(aligned.warnings),
            "path_labels": list(paths), "notes": notes, "cache_hit": False,
            "exploratory": portfolio.kind == "custom"}
    result = PipelineResult(meta=meta, preds=P, y=y_test, metrics=metrics, inference=inference,
                            episodes=episodes, overlay_stats=stats, overlay_stats_delay0=stats_no_delay,
                            overlay_paths=paths, flags=flags, coverage_risk=coverage,
                            folds=describe_folds(folds, X.index))

    # step 9: save
    if use_cache:
        cache.put(key, result.tables(), meta)
    return result


def live_forecast(result, aligned, cfg=None):
    """Forecast for the most recent days, whose outcomes are not known yet.

    Uses the same models and calibration methods as `result`. They are trained on all labelled
    days (the last part of them is used for calibration, after a gap of h days).
    This is a research prototype and has not been validated for real-time use.
    """
    cfg = cfg or get_config()
    meta = result.meta
    h = int(meta["horizon"])
    portfolio = make_portfolio(meta["weights"])
    port_ret, X_all, labels, X, y, extra = prepare_inputs(portfolio, aligned, h, cfg)

    new_days = X_all.index[X_all.index > X.index[-1]]   # days that have features but no label yet
    if len(new_days) == 0:
        raise PipelineError("There are no recent unlabelled days to forecast.")
    fit_idx, calib_idx = calibration_split(len(X), cfg["validation"]["calibration_fraction"], gap=h)
    extra_new = labels.reindex(new_days)[TEST_EXTRA]

    forecasts = {}
    for code in [*meta["baselines"], *meta["models"]]:
        model = build_model(code, port_ret, h)
        model.fit(X.iloc[fit_idx], y.iloc[fit_idx], extra.iloc[fit_idx][TRAIN_EXTRA])
        p_calib = model.predict_proba(X.iloc[calib_idx], extra.iloc[calib_idx][TEST_EXTRA])
        p_new = model.predict_proba(X_all.loc[new_days], extra_new)
        p, _ = calibrate_fold(p_calib, y.iloc[calib_idx].to_numpy(), p_new, meta["calibration"][code]["method"])
        forecasts[code] = pd.Series(np.asarray(p, float), index=new_days)

    P, _ = add_ensembles(pd.DataFrame(forecasts), meta["models"])
    signal = meta["signal"]
    members = meta["ensemble_members"].get(signal) or list(meta["models"])
    flags = abstain_flags(P[signal], P[members], X_all, float(meta["rule_a_threshold"]), cfg["trust"], y=None, h=h)
    return {"asof": new_days[-1], "forecast": P, "flags": flags, "signal": signal, "n_train": int(len(X))}
