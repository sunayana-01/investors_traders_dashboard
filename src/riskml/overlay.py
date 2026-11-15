"""Overlay backtest: use the forecast to decide how much of the portfolio to hold.

The rest of the money sits in cash. We compare that with simply holding the portfolio.

Timing (no peeking): a forecast made at the close of day t can only change the exposure
of day t+1 at the earliest. The main results wait one more day (extra_delay = 1),
so a forecast from the close of day t first affects the return of day t+2.

Every day:  net return = exposure * portfolio return + (1 - exposure) * cash return - trading cost
Trading cost = change in exposure * cost in basis points (1 bp = 0.01%).
The very first position is free, and the same for all strategies.

Two rules:
  Rule A: invest `cut` (for example 50%) when the probability is above a threshold, otherwise 100%
  Rule B: invest 1 - kappa * probability, but never less than `floor`
"""
import numpy as np
import pandas as pd

from riskml.config import TRADING_DAYS


def exposure_from_probs(p, rule, threshold=0.5, cut=0.5, kappa=1.0, floor=0.2):
    """Target exposure (0 to 1) for each day. Missing forecasts mean fully invested."""
    prob = p.to_numpy(float)
    if rule == "A":
        exposure = np.where(prob > threshold, cut, 1.0)
    elif rule == "B":
        exposure = np.clip(1 - kappa * prob, floor, 1.0)
    else:
        raise ValueError(f"Unknown rule {rule!r}")
    exposure = np.where(np.isnan(prob), 1.0, exposure)
    return pd.Series(exposure, index=p.index, name="target")


def daily_cash_return(rf_annual_pct, index):
    """Daily cash return from a yearly rate in percent (None means 0)."""
    if rf_annual_pct is None:
        return pd.Series(0.0, index=index)
    rate = rf_annual_pct.reindex(index.union(rf_annual_pct.index)).ffill().reindex(index).fillna(0.0)
    return rate / 100 / TRADING_DAYS


def run_overlay(port_ret, p, rule, rf_annual_pct=None, cost_bps=5.0, extra_delay=1, start=None, end=None,
                drawdown_limit=None, max_leverage=None, **rule_settings):
    """Backtest one strategy. rule = "A", "B" or "hold" (always fully invested).

    port_ret is indexed by the day of the return, p by the day the forecast was made.
    start and end limit the backtest to the days between them (inclusive).

    Two optional safety stops, both off by default:
      drawdown_limit  if the strategy has lost more than this from its peak (0.2 = 20%), go to cash
      max_leverage    if the wanted exposure is ever above this, go to cash
    Once the strategy has gone to cash it stays in cash for the rest of the backtest.
    """
    uses_forecast = rule != "hold" and p is not None
    if uses_forecast:
        first_forecast = p.first_valid_index()
        start = first_forecast if start is None else max(pd.Timestamp(start), first_forecast)

    days = port_ret.index
    if start is not None:
        days = days[days >= pd.Timestamp(start)]
    if end is not None:
        days = days[days <= pd.Timestamp(end)]
    returns = port_ret.reindex(days)

    if uses_forecast:
        target = exposure_from_probs(p.reindex(port_ret.index), rule, **rule_settings)
    else:
        target = pd.Series(1.0, index=port_ret.index)
    target = target.reindex(days)

    # the delay: the target of day t is used from day t + 1 + extra_delay
    wanted = target.shift(1 + int(extra_delay)).fillna(1.0).to_numpy(dtype=float)
    cash = daily_cash_return(rf_annual_pct, days)

    # go through the days one by one, because the safety stops depend on what happened before
    n = len(days)
    stock_returns, cash_returns = returns.to_numpy(dtype=float), cash.to_numpy(dtype=float)
    exposure, turnover, cost = np.ones(n), np.zeros(n), np.zeros(n)
    net_return, equity = np.zeros(n), np.ones(n)

    peak = 1.0              # highest equity so far
    previous_exposure = 1.0
    previous_equity = 1.0
    in_cash_for_good = False
    for i in range(n):
        held = wanted[i]
        if in_cash_for_good:
            held = 0.0
        elif max_leverage is not None and held > float(max_leverage):
            held = 0.0
            in_cash_for_good = True
        if i > 0 and drawdown_limit is not None and previous_equity / peak - 1.0 <= -float(drawdown_limit):
            held = 0.0
            in_cash_for_good = True

        exposure[i] = held
        turnover[i] = abs(held - previous_exposure)
        cost[i] = turnover[i] * cost_bps / 1e4
        net_return[i] = held * stock_returns[i] + (1 - held) * cash_returns[i] - cost[i]
        equity[i] = previous_equity * (1.0 + net_return[i])

        peak = max(peak, equity[i])
        previous_exposure = held
        previous_equity = equity[i]

    equity = pd.Series(equity, index=days)
    return pd.DataFrame({"ret_asset": returns, "target": target, "exposure": exposure,
                         "turnover": turnover, "cost": cost, "rf": cash, "net_ret": net_return,
                         "equity": equity, "drawdown": equity / equity.cummax() - 1.0}, index=days)


def perf_stats(path):
    """Summary numbers of one backtest."""
    n_days = len(path)
    years = n_days / TRADING_DAYS
    net, cash = path["net_ret"], path["rf"]

    volatility = float(net.std(ddof=1) * np.sqrt(TRADING_DAYS))
    cagr = float(path["equity"].iloc[-1] ** (1 / years) - 1)   # compound yearly growth rate
    max_drawdown = float(path["drawdown"].min())
    worst_21d = float(((1 + net).rolling(21).apply(np.prod, raw=True) - 1).min())
    sharpe = float((net - cash).mean() / net.std(ddof=1) * np.sqrt(TRADING_DAYS)) if volatility > 0 else np.nan

    return {"cagr": cagr, "ann_vol": volatility, "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "calmar": cagr / abs(max_drawdown) if max_drawdown < 0 else np.nan,
            "worst_21d": worst_21d,
            "avg_exposure": float(path["exposure"].mean()),
            "pct_days_reduced": float((path["exposure"] < 1 - 1e-9).mean()),
            "ann_turnover": float(path["turnover"].sum() / years),
            "ann_cost_drag": float(path["cost"].sum() / years),
            "n_days": n_days}


def compare_overlays(port_ret, signals, rules, rf_annual_pct=None, cost_bps=5.0, extra_delay=1):
    """Buy and hold against several overlay rules, all on the same dates.

    signals: {signal name: forecast Series}
    rules:   {label: {"signal": name, "rule": "A" or "B", ...settings}}
    Returns a table of summary numbers and a dict with the full path of every strategy.
    """
    # Only use the days where ALL signals exist. Before the first forecast and after the last one
    # there is no forecast, and a rule would just sit fully invested. That would make every
    # strategy look the same there (and mix in data the models were never tested on).
    common_start = max(signals[spec["signal"]].first_valid_index() for spec in rules.values())
    common_end = min(signals[spec["signal"]].last_valid_index() for spec in rules.values())

    paths = {"Buy & hold": run_overlay(port_ret, None, "hold", rf_annual_pct, cost_bps, extra_delay,
                                       common_start, common_end)}
    for label, spec in rules.items():
        settings = {k: v for k, v in spec.items() if k not in ("signal", "rule")}
        paths[label] = run_overlay(port_ret, signals[spec["signal"]], spec["rule"], rf_annual_pct, cost_bps,
                                   extra_delay, common_start, common_end, **settings)

    stats = pd.DataFrame({label: perf_stats(path) for label, path in paths.items()}).T
    stats["d_maxdd_vs_hold"] = stats["max_drawdown"] - stats.loc["Buy & hold", "max_drawdown"]   # above 0 = shallower
    stats["d_cagr_vs_hold"] = stats["cagr"] - stats.loc["Buy & hold", "cagr"]
    return stats, paths
