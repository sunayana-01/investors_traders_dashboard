"""Small Streamlit building blocks: checkbox lists and the portfolio chooser."""
import pandas as pd
import streamlit as st

from riskml.config import get_config
from riskml.portfolio import get_benchmark, make_portfolio, parse_weight_text

from .names import GROUPS, label

DEFAULT_CUSTOM_TEXT = "SPY 50, EFA 10, IEF 25, GLD 15"


def group_checkboxes(prefix, available, defaults, title="Choose what to show"):
    """Checkboxes for the forecasters, grouped (simple rules / models / ensembles). Returns the ticked codes."""
    chosen = []
    with st.expander(title):
        for group_name, codes in GROUPS.items():
            codes = [c for c in codes if c in available]
            if not codes:
                continue
            st.markdown(f"**{group_name}**")
            for code in codes:
                if st.checkbox(label(code), value=code in defaults, key=f"{prefix}_{code}"):
                    chosen.append(code)
    return chosen


def simple_checkboxes(prefix, options, defaults, title):
    """Checkboxes for options = {key: display text}. Returns the ticked keys."""
    chosen = []
    with st.expander(title):
        for key, text in options.items():
            if st.checkbox(text, value=key in defaults, key=f"{prefix}_{key}"):
                chosen.append(key)
    return chosen


def portfolio_chooser():
    """Sidebar control for the portfolio. Returns (portfolio, error text). One of them is None."""
    cfg = get_config()
    choices = {pid: f"{pid}: {p['label']}" for pid, p in cfg["portfolios"].items()}
    choices["custom"] = "Custom portfolio (exploratory)"
    choice = st.sidebar.selectbox("Portfolio", list(choices), format_func=choices.get, key="preset")

    try:
        if choice != "custom":
            portfolio = get_benchmark(choice, cfg)
        else:
            text = st.sidebar.text_area("Weights (ticker and percent; up to 10 long-only assets)",
                                        DEFAULT_CUSTOM_TEXT, key="custom_text",
                                        help="For example: SPY 60, IEF 30, GLD 10. Scaled to add up to 100%.")
            portfolio = make_portfolio(parse_weight_text(text), max_assets=cfg["data"]["max_assets"])
    except Exception as error:   # a typo in the weights should show a message, not crash the page
        return None, str(error)

    weights = ", ".join(f"{t} {w:.0%}" for t, w in portfolio.weights.items())
    st.sidebar.caption(f"Weights used: {weights}. Rebalanced daily to constant weights.")
    return portfolio, None
