"""Looking back: which holding caused the risk? (Describes what happened, it is not a forecast.)

Risk contribution of asset i over a window:
    share_i = weight_i * covariance(asset i, portfolio) / variance(portfolio)
The shares add up to 1. A diversifier can have a negative share.

We use the same window as the label (days t+1 ... t+h), so the shares tell us who drove
the volatility that the models were trying to predict.
"""
import numpy as np
import pandas as pd

from riskml.config import TRADING_DAYS, get_config
from riskml.episodes import episode_report, find_episodes
from riskml.labels import make_labels
from riskml.portfolio import asset_returns, make_portfolio, portfolio_returns


def forward_risk_contrib(asset_ret, weights, h):
    """Risk share of each asset over the NEXT h days, shown on the day the forecast is made."""
    tickers = list(weights)
    w = pd.Series(weights, dtype=float)
    assets = asset_ret[tickers]
    port = (assets * w).sum(axis=1, min_count=len(tickers))
    variance = port.rolling(h).var()
    variance = variance.where(variance > 0)
    shares = pd.DataFrame({t: w[t] * assets[t].rolling(h).cov(port) / variance for t in tickers})
    return shares.shift(-h)   # the window that ENDS on t+h starts at t+1


def describe_flag(detected, lead):
    if not detected:
        return "missed"
    lead = int(lead)
    return f"flagged {lead}d early" if lead >= 0 else f"late by {-lead}d"


def build_context(result, aligned, cfg=None, far=0.10):
    """Extra tables for the dashboard: volatility series, risk shares and one row per episode."""
    cfg = cfg or get_config()
    h = int(result.meta["horizon"])
    portfolio = make_portfolio(result.meta["weights"])
    assets = asset_returns(aligned.prices)
    port_ret = portfolio_returns(assets, portfolio)

    labels = make_labels(port_ret, h, cfg["labels"]["quantile"], cfg["labels"]["min_threshold_history_days"])
    vol = pd.DataFrame({"fwd_rv": labels["fwd_rv"], "threshold": labels["threshold"], "y": labels["y"],
                        "rv_21": port_ret.rolling(21).std() * np.sqrt(TRADING_DAYS),
                        "vix": aligned.vix.reindex(port_ret.index) / 100})
    contrib = forward_risk_contrib(assets, portfolio.weights, h)

    y = result.y.dropna()
    episodes = find_episodes(y)
    reports = {name: episode_report(y, result.preds[name], far=far)["episodes"] for name in result.preds.columns}

    rows = []
    for k, episode in episodes.iterrows():
        days = y.index[int(episode.start_pos): int(episode.end_pos) + 1]
        shares = contrib.loc[days].mean()
        top = shares.idxmax()
        row = {"start": episode.start, "end": episode.end, "days": int(episode.length),
               "peak_vol": float(vol.loc[days, "fwd_rv"].max()),
               "top_driver": top, "top_share": float(shares[top])}
        row.update({f"share_{t}": float(shares[t]) for t in shares.index})
        for name, report in reports.items():
            r = report.iloc[k]
            row[f"flag_{name}"] = describe_flag(bool(r["detected"]), r["lead_days"] if r["detected"] else 0)
        rows.append(row)

    return {"vol": vol, "contrib": contrib, "episodes": pd.DataFrame(rows),
            "far": far, "assets": list(contrib.columns)}
