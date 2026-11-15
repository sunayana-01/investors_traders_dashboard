"""Loads the market data for the dashboard and remembers it for a few hours."""
import streamlit as st

from riskml.config import get_config
from riskml.data import load_market_data


@st.cache_data(show_spinner="Loading market data...", ttl=6 * 3600)
def get_aligned(tickers, refresh=False):
    return load_market_data(list(tickers), get_config(), refresh)
