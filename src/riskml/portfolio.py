"""Portfolios: checking the weights and calculating daily returns.

A portfolio here is long-only (no negative weights) and is rebalanced every day
back to the same weights. With simple returns this means the portfolio return
of a day is just the weighted average of the asset returns of that day.
"""
import re
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Portfolio:
    name: str
    weights: dict                  # ticker -> weight, sorted by ticker, adds up to 1
    kind: str = "custom"           # "benchmark" (P1 to P3) or "custom" (typed in by the user)
    notes: list = field(default_factory=list)   # what we changed while cleaning the weights

    @property
    def tickers(self):
        return list(self.weights)

    @property
    def label(self):
        if self.kind == "custom":
            return f"{self.name} (exploratory, not validated)"
        return self.name


def clean_weights(weights, max_assets=10):
    """Check the weights and scale them so they add up to 1.

    Returns the clean weights and a list of notes about what was changed.
    """
    notes = []
    if len(weights) == 0:
        raise ValueError("No weights given.")

    totals = {}
    for raw_ticker, raw_weight in weights.items():
        ticker = str(raw_ticker).strip().upper()
        if not re.match(r"^[A-Z][A-Z0-9.\-]{0,11}$", ticker):
            raise ValueError(f"Invalid ticker {raw_ticker!r}.")
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError):
            raise ValueError(f"Weight for {ticker} is not a number: {raw_weight!r}.") from None
        if not np.isfinite(weight):
            raise ValueError(f"Weight for {ticker} is not a finite number.")
        if weight < 0:
            raise ValueError(f"Negative weight for {ticker}. Only long-only portfolios are supported.")
        if ticker in totals:
            notes.append(f"{ticker} was listed more than once, so its weights were added together.")
        totals[ticker] = totals.get(ticker, 0.0) + weight

    zero_weight = [t for t, w in totals.items() if w <= 1e-12]
    if zero_weight:
        notes.append("Dropped assets with zero weight: " + ", ".join(zero_weight) + ".")
        totals = {t: w for t, w in totals.items() if w > 1e-12}
    if not totals:
        raise ValueError("All weights are zero.")
    if len(totals) > max_assets:
        raise ValueError(f"At most {max_assets} assets are supported (got {len(totals)}).")

    total = sum(totals.values())
    if abs(total - 1.0) > 1e-6:
        notes.append(f"Weights added up to {total:.4f}, so they were scaled to add up to 1.")
    clean = {t: totals[t] / total for t in sorted(totals)}
    return clean, notes


def parse_weight_text(text):
    """Turn text like "SPY 60, IEF 30, GLD 10" into {"SPY": 60.0, "IEF": 30.0, "GLD": 10.0}.

    Entries are separated by commas, semicolons or new lines. The ticker and the
    number can be separated by a space, a colon or an equals sign.
    If the numbers add up to more than 1.5 we assume they are percentages.
    """
    if text is None or not str(text).strip():
        raise ValueError("The weight text is empty.")

    result = {}
    for entry in re.split(r"[,;\n]+", str(text)):
        entry = entry.strip().replace("%", "")
        if not entry:
            continue
        parts = re.split(r"[\s:=]+", entry)
        if len(parts) != 2:
            raise ValueError(f"Could not read {entry!r}. Use a format like 'SPY 60'.")
        ticker = parts[0].upper()
        try:
            weight = float(parts[1])
        except ValueError:
            raise ValueError(f"Could not read the number in {entry!r}.") from None
        if ticker in result:
            raise ValueError(f"Ticker {ticker} appears more than once.")
        result[ticker] = weight

    if not result:
        raise ValueError("No weights found in the text.")
    if sum(result.values()) > 1.5:
        result = {t: w / 100 for t, w in result.items()}
    return result


def make_portfolio(weights, name=None, kind="custom", max_assets=10):
    clean, notes = clean_weights(weights, max_assets)
    if name is None:
        name = "Custom: " + ", ".join(f"{t} {w:.0%}" for t, w in clean.items())
    return Portfolio(name=name, weights=clean, kind=kind, notes=notes)


def get_benchmark(pid, cfg):
    """One of the three fixed portfolios from the config ('P1', 'P2' or 'P3')."""
    pid = str(pid).strip().upper()
    if pid not in cfg["portfolios"]:
        raise KeyError(f"Unknown portfolio {pid!r}. Choose from {sorted(cfg['portfolios'])}.")
    entry = cfg["portfolios"][pid]
    return make_portfolio(entry["weights"], name=f"{pid}: {entry['label']}", kind="benchmark")


def asset_returns(prices):
    """Simple daily returns: today's price divided by yesterday's price, minus 1.

    Prices are not forward filled here. A missing price gives a missing return.
    """
    prices = prices.astype(float).where(lambda p: p > 0)   # zero or negative prices count as missing
    returns = prices / prices.shift(1) - 1
    return returns.iloc[1:]


def portfolio_returns(asset_ret, portfolio):
    """Daily portfolio return = sum of weight * asset return.

    The series starts on the first day where every asset has a return.
    """
    tickers = portfolio.tickers
    missing = [t for t in tickers if t not in asset_ret.columns]
    if missing:
        raise KeyError("No return data for: " + ", ".join(missing))

    assets = asset_ret[tickers]
    first_days = [assets[t].first_valid_index() for t in tickers]
    if any(day is None for day in first_days):
        raise ValueError("At least one asset has no valid returns.")
    assets = assets.loc[max(first_days):]

    has_gap = assets.isna().any(axis=1)
    if has_gap.any():
        warnings.warn(f"{int(has_gap.sum())} day(s) with a missing asset return were dropped "
                      f"for {portfolio.name}.", stacklevel=2)
        assets = assets.loc[~has_gap]

    weights = np.array([portfolio.weights[t] for t in tickers])
    return pd.Series(assets.to_numpy(dtype=float) @ weights, index=assets.index, name=portfolio.name)
