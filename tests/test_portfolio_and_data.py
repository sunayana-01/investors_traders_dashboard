import numpy as np
import pandas as pd
import pytest

from riskml.data import place_with_delay
from riskml.portfolio import (asset_returns, clean_weights, make_portfolio, parse_weight_text,
                              portfolio_returns)


def test_parse_weight_text_formats():
    assert parse_weight_text("SPY 60, IEF 40") == {"SPY": 0.6, "IEF": 0.4}     # percentages are turned into fractions
    assert parse_weight_text("spy:0.5;ief=0.5") == {"SPY": 0.5, "IEF": 0.5}
    assert parse_weight_text("SPY 60%\nGLD 40%") == {"SPY": 0.6, "GLD": 0.4}


@pytest.mark.parametrize("text", ["", "SPY sixty", "SPY 50, SPY 50", "SPY"])
def test_parse_weight_text_rejects_bad_input(text):
    with pytest.raises(ValueError):
        parse_weight_text(text)


def test_clean_weights_scales_and_reports():
    weights, notes = clean_weights({"spy": 3, "IEF": 1})
    assert weights == {"IEF": 0.25, "SPY": 0.75} and notes      # sorted, scaled to 1, with a note


def test_clean_weights_rejects_short_selling_and_too_many_assets():
    with pytest.raises(ValueError):
        clean_weights({"SPY": 1.2, "IEF": -0.2})
    with pytest.raises(ValueError):
        clean_weights({f"A{i}": 1 for i in range(11)})


def test_portfolio_return_is_weighted_average():
    days = pd.bdate_range("2020-01-01", periods=5)
    prices = pd.DataFrame({"A": [100, 110, 110, 121, 121], "B": [50, 50, 55, 55, 60.5]}, index=days)
    portfolio = make_portfolio({"A": 0.5, "B": 0.5})
    returns = portfolio_returns(asset_returns(prices), portfolio)
    assert np.isclose(returns.iloc[0], 0.5 * 0.10 + 0.5 * 0.0)


def test_macro_values_are_delayed():
    days = pd.bdate_range("2020-01-01", periods=30)
    series = pd.Series([1.0], index=[days[5]])                    # one value dated day 5
    placed = place_with_delay(series, days, lag_days=3, ffill_limit=5)
    assert placed.iloc[:8].isna().all()                          # not visible before day 5 + 3
    assert placed.iloc[8] == 1.0                                 # visible from day 8
    assert placed.iloc[8 + 5] == 1.0 and np.isnan(placed.iloc[8 + 6])   # carried forward for 5 days only
