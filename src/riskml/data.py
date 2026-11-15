"""Getting the market data and putting it on one calendar.

Three steps:
  1. download prices (Yahoo Finance), the VIX (Yahoo) and macro series (FRED).
     Everything is saved as parquet files in data/raw, so the next run is fast and works offline.
  2. line everything up on the SPY trading calendar. Macro numbers are published with a
     delay, so we shift them by a few days. That way a model never sees a number before
     it was really available.
  3. run a few sanity checks on the result.
"""
import io
import re
import urllib.request
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from riskml.config import TRADING_DAYS, get_config, project_path

CALENDAR_TICKER = "SPY"   # the dates SPY traded on are the dates we use for everything
PRICE_FFILL_LIMIT = 3     # a missing price can be filled from the day before, but only for 3 days (holidays)


class DataUnavailableError(RuntimeError):
    pass


@dataclass
class AlignedData:
    prices: pd.DataFrame        # adjusted close prices for the tickers we need
    vix: pd.Series
    macro: pd.DataFrame         # curve_slope, credit_spread, fin_stress (already delayed)
    cash_daily: pd.Series       # daily return of cash, taken from the 3-month T-bill rate
    start: pd.Timestamp
    end: pd.Timestamp
    warnings: list = field(default_factory=list)   # data quality warnings (not errors)


# Step 1: download and save

def clean_index(data):
    """Dates without time zone and time of day, sorted, one row per date."""
    index = pd.DatetimeIndex(data.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    data = data.copy()
    data.index = index.normalize()
    data = data.sort_index()
    return data[~data.index.duplicated(keep="last")]


def file_for(kind, name):
    safe_name = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")   # "^VIX" becomes "VIX"
    folder = project_path(get_config()["project"]["data_dir"] + "/raw")
    return folder / f"{kind}__{safe_name}.parquet"


def save_series(file, series):
    series.rename("value").to_frame().to_parquet(file)


def read_series(file, name):
    series = pd.read_parquet(file)["value"]
    series.index = pd.DatetimeIndex(series.index)
    return series.rename(name)


def download_yahoo(tickers, start):
    """Adjusted close prices from Yahoo Finance for a list of tickers."""
    import yfinance as yf   # imported here so the rest of the code works without it

    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False, threads=False)
    if raw is None or raw.empty:
        raise DataUnavailableError(f"Yahoo returned no data for {tickers}")
    close = raw["Close"]
    if isinstance(close, pd.Series):   # older versions return a Series for one ticker
        close = close.to_frame(tickers[0])
    return clean_index(close)


def download_fred(series_id, base_url, start):
    """One macro series from FRED as a pandas Series (FRED gives a plain CSV file)."""
    url = f"{base_url}?id={series_id}&cosd={start}"
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            text = response.read().decode("utf-8")
    except Exception as error:
        raise DataUnavailableError(f"Could not download {series_id} from FRED: {error}") from error
    return parse_fred_csv(text, series_id)


def parse_fred_csv(text, series_id):
    table = pd.read_csv(io.StringIO(text))
    dates = pd.to_datetime(table.iloc[:, 0])
    column = series_id if series_id in table.columns else table.columns[1]
    values = pd.to_numeric(table[column], errors="coerce")   # FRED writes "." for a missing value
    series = clean_index(pd.Series(values.to_numpy(), index=dates, name=series_id)).dropna()
    if series.empty:
        raise DataUnavailableError(f"No observations found for FRED series {series_id}")
    return series


def load_prices(tickers, cfg, refresh=False):
    """Prices for each ticker. Uses the saved file if there is one, otherwise downloads."""
    tickers = list(dict.fromkeys(tickers))   # remove duplicates but keep the order
    series = {}
    to_download = []
    for ticker in tickers:
        file = file_for("price", ticker)
        if file.exists() and not refresh:
            series[ticker] = read_series(file, ticker)
        else:
            to_download.append(ticker)

    if to_download:
        try:
            fresh = download_yahoo(to_download, cfg["data"]["start_date"])
        except Exception as error:
            fresh = pd.DataFrame()
            print(f"Download failed ({error}). Trying the saved files instead.")
        for ticker in to_download:
            file = file_for("price", ticker)
            if ticker in fresh.columns and not fresh[ticker].dropna().empty:
                series[ticker] = fresh[ticker].dropna().rename(ticker)
                save_series(file, series[ticker])
            elif file.exists():
                series[ticker] = read_series(file, ticker)   # old copy is better than nothing
            else:
                raise DataUnavailableError(f"No data for ticker '{ticker}'. "
                                           "Check the ticker symbol and your internet connection.")

    prices = pd.concat([series[t] for t in tickers], axis=1)
    prices.columns = tickers
    return prices.sort_index()


def load_fred(name, spec, cfg, refresh=False):
    file = file_for("fred", spec["id"])
    if file.exists() and not refresh:
        return read_series(file, name)
    try:
        series = download_fred(spec["id"], cfg["data"]["fred_base_url"], cfg["data"]["start_date"])
        save_series(file, series)
        return series.rename(name)
    except DataUnavailableError:
        if file.exists():
            return read_series(file, name)
        raise


# Step 2: put everything on the same calendar

def place_with_delay(series, calendar, lag_days, ffill_limit):
    """Put a published series on the trading calendar, but only after it was really published.

    A value dated d is moved to the first trading day on or after d, then pushed
    lag_days further into the future. Gaps after that are filled with the last
    known value (never with a future one) for at most ffill_limit days.
    """
    series = series.dropna().sort_index()
    series = series[~series.index.duplicated(keep="last")]
    positions = calendar.searchsorted(series.index, side="left") + int(lag_days)
    on_calendar = positions < len(calendar)
    if not on_calendar.any():
        return pd.Series(np.nan, index=calendar, name=series.name)

    placed = pd.Series(series.to_numpy()[on_calendar], index=positions[on_calendar])
    placed = placed.groupby(level=0).last()   # if two values land on one day, keep the newest
    values = np.full(len(calendar), np.nan)
    values[placed.index.to_numpy()] = placed.to_numpy()
    return pd.Series(values, index=calendar, name=series.name).ffill(limit=int(ffill_limit))


def align_data(prices, vix, fred, portfolio_tickers, cfg):
    """Put prices, VIX and macro series on the SPY calendar and cut to the common start date."""
    if CALENDAR_TICKER not in prices.columns:
        raise ValueError(f"{CALENDAR_TICKER} prices are needed for the trading calendar")

    needed = list(dict.fromkeys([*portfolio_tickers, *cfg["data"]["market_reference_tickers"]]))
    calendar = prices[CALENDAR_TICKER].dropna().index
    prices = prices[needed].reindex(calendar).ffill(limit=PRICE_FFILL_LIMIT)

    # the usable history starts on the day when every needed ticker has a price
    first_day = prices.apply(lambda column: column.first_valid_index())
    if first_day.isna().any():
        raise ValueError(f"No price history for: {list(first_day[first_day.isna()].index)}")
    start = first_day.max()
    end = prices.dropna().index.max()

    years = (end - start).days / 365.25
    min_years = cfg["data"]["min_common_history_years"]
    if years < min_years:
        raise ValueError(f"Common history is only {years:.1f} years ({start.date()} to {end.date()}), "
                         f"less than the {min_years} years we need. "
                         f"The asset that starts last is {first_day.idxmax()}.")
    prices = prices.loc[start:end]

    # VIX is known at the close of its own day, so no delay
    vix = vix.reindex(calendar).ffill(limit=PRICE_FFILL_LIMIT).loc[start:end].rename("vix")

    macro = {}
    cash_daily = pd.Series(0.0, index=prices.index, name="cash_daily")
    for name, spec in cfg["data"]["fred_series"].items():
        series = place_with_delay(fred[name], calendar, spec["lag_days"], spec["ffill_limit"]).loc[start:end]
        if name == "tbill_3m":
            cash_daily = (series / 100 / TRADING_DAYS).rename("cash_daily")   # percent per year to daily
        else:
            macro[name] = series.rename(name)

    return AlignedData(prices=prices, vix=vix, macro=pd.DataFrame(macro),
                       cash_daily=cash_daily, start=start, end=end)


# Step 3: sanity checks

def check_data(aligned, cfg):
    """Raise an error for data we cannot use. Return a list of warnings for odd-looking data."""
    problems = []
    notes = []

    for ticker in aligned.prices.columns:
        prices = aligned.prices[ticker]
        if prices.isna().any():
            problems.append(f"{ticker}: {int(prices.isna().sum())} missing price(s)")
        if (prices <= 0).any():
            problems.append(f"{ticker}: prices that are zero or negative")

        daily_move = prices.pct_change(fill_method=None)
        big_moves = daily_move[daily_move.abs() > cfg["data"]["max_abs_return_flag"]].dropna()
        for day, move in big_moves.items():
            notes.append(f"{ticker}: daily move of {move:+.1%} on {day.date()}, please check it is real")

        same_as_yesterday = (prices.diff() == 0)
        if (same_as_yesterday.rolling(4).sum() == 4).any():
            notes.append(f"{ticker}: the same price 5 days in a row at least once")

    if aligned.vix.isna().any():
        problems.append("VIX: missing values")
    if (aligned.vix <= 0).any():
        problems.append("VIX: values that are zero or negative")

    for name in aligned.macro.columns:
        missing = int(aligned.macro[name].isna().sum())
        if missing:
            notes.append(f"{name}: {missing} day(s) without a value (before the first release or a long gap)")

    age_in_days = (pd.Timestamp.today().normalize() - aligned.end).days
    if age_in_days > 7:
        notes.append(f"the latest data is {age_in_days} days old (last date {aligned.end.date()})")

    if problems:
        raise ValueError("Data problems: " + "; ".join(problems[:5]))
    return notes


def load_market_data(portfolio_tickers, cfg=None, refresh=False):
    """Everything the pipeline needs for one portfolio, checked and on one calendar."""
    cfg = cfg or get_config()
    needed = list(dict.fromkeys([*portfolio_tickers, *cfg["data"]["market_reference_tickers"]]))

    prices = load_prices(needed, cfg, refresh)
    vix_ticker = cfg["data"]["vix_ticker"]
    vix = load_prices([vix_ticker], cfg, refresh)[vix_ticker]
    fred = {name: load_fred(name, spec, cfg, refresh) for name, spec in cfg["data"]["fred_series"].items()}

    aligned = align_data(prices, vix, fred, portfolio_tickers, cfg)
    aligned.warnings = check_data(aligned, cfg)
    return aligned
