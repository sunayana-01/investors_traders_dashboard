"""Portfolio risk dashboard (course project, research prototype).

Run it with:  streamlit run app/Home.py
"""
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))
from components import charts                                                       # noqa: E402
from components.loaders import get_aligned                                          # noqa: E402
from components.names import METRICS, label, short, strategy_label, verdict        # noqa: E402
from components.widgets import group_checkboxes, portfolio_chooser, simple_checkboxes   # noqa: E402

from riskml.attribution import build_context                                        # noqa: E402
from riskml.config import get_config                                                # noqa: E402
from riskml.evaluate import reliability_table                                       # noqa: E402
from riskml.overlay import perf_stats, run_overlay                                  # noqa: E402
from riskml.pipeline import live_forecast, run_pipeline                             # noqa: E402
from riskml.portfolio import asset_returns, make_portfolio, portfolio_returns        # noqa: E402

cfg = get_config()
MODEL_LABELS = {"M1": "M1 HAR + VIX logistic (reference)", "M2": "M2 full-feature logistic",
                "M3": "M3 LightGBM", "M3H": "M3H HistGradientBoosting", "M4": "M4 GARCH probability"}

st.set_page_config(page_title="Portfolio risk dashboard", layout="wide")
st.markdown(
    """
    <style>
    .metric-card {
        background: rgba(255,255,255,0.92);
        color: #111827;
        border-radius: 14px;
        padding: 1rem 1rem 0.7rem 1rem;
        border: 1px solid rgba(17,24,39,0.08);
        box-shadow: 0 8px 18px rgba(15,23,42,0.08);
        margin-bottom: 0.75rem;
    }
    .metric-card .label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; color: #475569; }
    .metric-card .value { font-size: 1.35rem; font-weight: 700; margin-top: 0.2rem; }
    .metric-card .delta { font-size: 0.8rem; color: #334155; margin-top: 0.2rem; }
    .soft-panel {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(148,163,184,0.22);
        border-radius: 12px;
        padding: 0.8rem;
    }
    .compact-label { font-size: 0.75rem; color: #cbd5e1; letter-spacing: 0.04em; text-transform: uppercase; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Risk Monitor")
st.caption("Plain-English portfolio-risk monitoring. Open the Guide for context and model definitions.")


# 1. Sidebar controls

st.sidebar.header("Portfolio")
portfolio, portfolio_error = portfolio_chooser()

st.sidebar.header("Forecast")
horizons = list(cfg["labels"]["horizons"])
horizon = st.sidebar.selectbox("Horizon (trading days)", horizons,
                               index=horizons.index(cfg["labels"]["primary_horizon"]), key="horizon",
                               help="How far ahead we look. 21 trading days is about one month (the main choice).")
period = "dev"
models = st.sidebar.multiselect("Models", list(MODEL_LABELS), default=list(MODEL_LABELS),
                                format_func=MODEL_LABELS.get, key="models")
ensemble_choice = st.sidebar.selectbox("Ensemble (average of several models)", ["E-core", "E-ML", "E-M1M4", "None"],
                                       key="ensemble")
signal_choice = st.sidebar.selectbox("Main signal to show", ["Ensemble (or M1 if none)", *MODEL_LABELS],
                                     key="signal")

st.sidebar.header("Overlay")
cost_bps = st.sidebar.slider("Transaction cost (bps per unit of exposure change)", 0, 50,
                             int(cfg["overlay"]["cost_bps"]), key="cost",
                             help="1 bp = 0.01%. Paid each time the rule changes how much is invested.")
rule_a_default = cfg["overlay"]["rule_a"].get("threshold")
rule_a_threshold = st.sidebar.slider("Rule A: cut exposure when the chance is above", 10, 90,
                                     int(round(100 * (0.5 if rule_a_default is None else rule_a_default))), 5,
                                     key="rule_a_threshold", format="%d%%",
                                     help="Lower = the rule acts more often.")
rule_a_exposure = st.sidebar.slider("Rule A: share invested when cut", 0, 100,
                                    int(round(100 * cfg["overlay"]["rule_a"]["reduced_exposure"])), 5,
                                    key="rule_a_exposure", format="%d%%",
                                    help="The rest of the portfolio is held in cash.")
refresh = st.sidebar.checkbox("Refresh market data from the web", value=False, key="refresh")
run_clicked = st.sidebar.button("Run analysis", type="primary", key="run")
retrigger_clicked = st.sidebar.button("Re-trigger analysis", type="secondary", key="retrigger")
st.sidebar.caption("Using the development sample only for now.")


def force_refresh_output():
    st.session_state.pop("output", None)
    st.session_state.pop("output_key", None)
    st.session_state["force_recalc"] = True


# 2. Run the analysis when the controls change or when the button is pressed

def compute():
    aligned = get_aligned(tuple(portfolio.tickers), refresh)
    ensemble = None if ensemble_choice == "None" else ensemble_choice
    signal = None if signal_choice.startswith("Ensemble") else signal_choice
    result = run_pipeline(portfolio, horizon, models=tuple(models) or ("M1",), ensemble=ensemble,
                          cost_bps=float(cost_bps), signal=signal, period="dev", allow_holdout=False,
                          aligned=aligned, n_boot=500,
                          rule_a_threshold=rule_a_threshold / 100, rule_a_exposure=rule_a_exposure / 100)
    return {"result": result, "live": live_forecast(result, aligned), "context": build_context(result, aligned)}


def settings_key():
    ensemble = None if ensemble_choice == "None" else ensemble_choice
    signal = None if signal_choice.startswith("Ensemble") else signal_choice
    return (
        str(portfolio),
        int(horizon),
        tuple(models),
        ensemble,
        signal,
        period,
        float(cost_bps),
        int(rule_a_threshold),
        int(rule_a_exposure),
        bool(refresh),
    )


force_recalc = st.session_state.pop("force_recalc", False)

if retrigger_clicked:
    force_refresh_output()

if portfolio_error:
    st.error(f"Portfolio not usable: {portfolio_error}")
else:
    current_key = settings_key()
    if run_clicked or retrigger_clicked or force_recalc or "output" not in st.session_state or st.session_state.get("output_key") != current_key:
        with st.spinner("Fitting models with purged walk-forward validation..."):
            try:
                st.session_state["output"] = compute()
                st.session_state["output_key"] = current_key
            except Exception as error:
                st.session_state.pop("output", None)
                st.session_state.pop("output_key", None)
                st.error(f"Could not run the analysis: {error}")

output = st.session_state.get("output")
if output is None:
    st.stop()

result, live, context = output["result"], output["live"], output["context"]
meta, signal = result.meta, result.meta["signal"]
preds = result.preds
default_codes = list(dict.fromkeys([signal, "B5"]))   # forecasters ticked by default in the checkbox lists


# 3. Header

st.subheader(meta["portfolio"])
if meta["exploratory"]:
    st.warning("Custom portfolio: **exploratory, not validated.** The research claims were only tested on P1-P3.")

cards = st.columns(4)
with cards[0]:
    st.markdown(f"<div class='metric-card'><div class='label'>Horizon</div><div class='value'>{meta['horizon']} days</div><div class='delta'>forecast window</div></div>", unsafe_allow_html=True)
with cards[1]:
    st.markdown(f"<div class='metric-card'><div class='label'>Data as of</div><div class='value'>{live['asof'].date()}</div><div class='delta'>latest observed point</div></div>", unsafe_allow_html=True)
with cards[2]:
    st.markdown(f"<div class='metric-card'><div class='label'>Out-of-sample</div><div class='value'>{meta['test_start'][:4]}–{meta['test_end'][:4]}</div><div class='delta'>{meta['n_folds']} folds</div></div>", unsafe_allow_html=True)
with cards[3]:
    st.markdown(f"<div class='metric-card'><div class='label'>Main signal</div><div class='value'>{short(signal)}</div><div class='delta'>{signal}</div></div>", unsafe_allow_html=True)
st.caption("The dashboard refreshes when the controls change. Press **Run analysis** or **Re-trigger analysis** if you want to force a fresh calculation.")
if st.button("Force recalculation", key="page_retrigger", type="secondary"):
    force_refresh_output()


# 4. One function per tab

def tab_current_risk():
    if st.button("Recalculate this view", key="recalc_current_risk", type="secondary"):
        force_refresh_output()
    st.caption(f"What the selected {meta['period']} period says about the next trading days.")
    forecasts = preds.copy()
    flags = result.flags.reindex(forecasts.index)
    p_now = float(forecasts[signal].iloc[-1])
    base_rate = float(result.y.mean())

    with st.container():
        metric_cols = st.columns(2)
        with metric_cols[0]:
            st.markdown(f"<div class='soft-panel'><div class='compact-label'>Current forecast</div><div style='font-size:2rem;font-weight:700;'>{p_now:.0%}</div><div style='color:#cbd5e1;'>Chance of elevated volatility in {meta['horizon']} days</div></div>", unsafe_allow_html=True)
        with metric_cols[1]:
            st.markdown(f"<div class='soft-panel'><div class='compact-label'>Base rate</div><div style='font-size:2rem;font-weight:700;'>{base_rate:.0%}</div><div style='color:#cbd5e1;'>Average elevated-volatility rate in this period</div></div>", unsafe_allow_html=True)

    last_flags = flags.iloc[-1]
    if bool(last_flags["abstain"]):
        reasons = [k for k in ("disagree", "ood", "near_threshold", "high_cal_error") if bool(last_flags.get(k, False))]
        st.warning("Low confidence (" + ", ".join(reasons).replace("_", " ") + "). "
                  "See the Guide: do not lean on this forecast alone.")
    else:
        st.success("No warning sign fired: the models broadly agree and the selected period looks normal.")

    threshold = float(meta["rule_a_threshold"])
    rule_b = cfg["overlay"]["rule_b"]
    exposure_a = float(meta["rule_a_exposure"]) if p_now > threshold else 1.0
    exposure_b = min(max(1 - float(rule_b["kappa"]) * p_now, float(rule_b["min_exposure"])), 1.0)
    st.markdown(f"**Overlay view:** Rule A invests **{exposure_a:.0%}** (threshold {threshold:.0%}); Rule B invests **{exposure_b:.0%}**.")

    chart_col, controls_col = st.columns([3.2, 1.3])
    with controls_col:
        picks = group_checkboxes("now", forecasts.columns, set(default_codes), "Choose which forecasters to compare")
    with chart_col:
        if not picks:
            st.plotly_chart(charts.empty_chart("No forecast selected"), use_container_width=True)
        else:
            st.plotly_chart(charts.recent_path(forecasts, picks, signal, base_rate), use_container_width=True)
    st.caption("'Elevated' = realised volatility over the next horizon above the 75th percentile of this "
               "portfolio's own past. This uses the selected period's test window, not the most recent unlabeled days.")


def tab_volatility():
    if st.button("Recalculate this view", key="recalc_volatility", type="secondary"):
        force_refresh_output()
    st.caption("Top: what actually happened to the portfolio's volatility (red shading = it was 'elevated'). "
               "Bottom: what each forecaster said beforehand. A good forecaster is high in the red zones and low elsewhere.")
    chart_col, controls_col = st.columns([3.2, 1.3])
    with controls_col:
        show_rv21 = st.checkbox("Also show volatility over the PAST 21 days", key="vol_rv21")
        show_vix = st.checkbox("Also show the VIX (divided by 100)", key="vol_vix")
        picks = group_checkboxes("vol", preds.columns, set(default_codes), "Choose which forecasters to plot")
    with chart_col:
        chart_slot = st.empty()
        if not picks:
            with chart_slot:
                st.plotly_chart(charts.empty_chart("No forecast selected"), use_container_width=True)
        else:
            with chart_slot:
                st.plotly_chart(charts.vol_and_prob(context["vol"], preds, picks, meta["horizon"], show_rv21, show_vix),
                                use_container_width=True)
    st.caption("Volatility is plotted on the date the forecast is made, but it covers the days AFTER that date. "
               "The threshold is based on the past, so after a big crisis it can stay high and calm years rarely cross it.")


def tab_episodes():
    if st.button("Recalculate this view", key="recalc_episodes", type="secondary"):
        force_refresh_output()
    st.caption("After-the-fact view: for each stretch of elevated volatility, which holding drove the realised risk, "
               "and did each forecaster flag it in advance? The models do not forecast per holding.")
    assets = context["assets"]
    chosen_assets = simple_checkboxes("epa", {a: a for a in assets}, set(assets), "Choose which holdings to show")
    show_all = st.checkbox("Show all realised data (including 2019 onward, descriptive only; no model forecasts there)",
                           key="ep_full")
    contrib = context["contrib"] if show_all else context["contrib"].loc[preds.index.min():preds.index.max()]
    if chosen_assets:
        st.plotly_chart(charts.contrib_area(contrib.dropna(how="all"), chosen_assets, meta["horizon"]))

    episodes = context["episodes"]
    if len(episodes) == 0:
        st.info("No elevated-volatility episodes in the test window for this portfolio.")
        return

    table_col, side_col = st.columns([2.6, 1.4])
    with side_col:
        codes = group_checkboxes("epf", preds.columns, set(default_codes), "Choose which forecasters to show in the table")
        pick = st.selectbox("Look at one episode", list(episodes.index), key="ep_pick",
                           format_func=lambda i: f"{episodes.loc[i, 'start'].date()} to {episodes.loc[i, 'end'].date()}")
    with table_col:
        table = episodes[["start", "end", "days", "peak_vol", "top_driver", "top_share"]].copy()
        table["start"], table["end"] = table["start"].dt.date, table["end"].dt.date
        table = table.rename(columns={"days": "length (days)", "peak_vol": "peak realised vol",
                                     "top_driver": "main driver", "top_share": "driver's share"})
        for code in codes:
            table[short(code)] = episodes[f"flag_{code}"]
        st.dataframe(table.style.format({"peak realised vol": "{:.0%}", "driver's share": "{:.0%}"}))
        st.caption(f"'Flagged' uses an alarm level set so that each forecaster raises a false alarm on about "
                   f"{context['far']:.0%} of calm days, so forecasters are compared fairly. "
                   "'early' = days before the episode's first elevated day.")

    shares = episodes.loc[pick, [f"share_{a}" for a in assets]]
    shares.index = assets
    st.plotly_chart(charts.episode_bars(shares, "Share of realised portfolio risk during this episode"))
    flags_text = "  ".join(f"{short(c)}: {episodes.loc[pick, f'flag_{c}']}." for c in codes)
    st.markdown(f"Main driver: **{episodes.loc[pick, 'top_driver']}** ({episodes.loc[pick, 'top_share']:.0%}). {flags_text}")


def with_readable_index(table):
    """Replace model codes in the row labels with their plain-English names."""
    table = table.copy()
    if isinstance(table.index, pd.MultiIndex):
        table.index = pd.MultiIndex.from_tuples([(label(a), b) for a, b in table.index], names=table.index.names)
    else:
        table.index = [label(i) for i in table.index]
    return table


def tab_accuracy():
    if st.button("Recalculate this view", key="recalc_accuracy", type="secondary"):
        force_refresh_output()
    st.caption("How accurate each forecaster was on data it had never seen (walk-forward, development period only). "
               "Lower Brier score is better; higher PR-AUC is better.")
    st.info(verdict(result))
    show_all = st.toggle("Show all metrics", value=False, key="acc_detail")
    columns = ["brier", "pr_auc"]
    if show_all:
        columns = [c for c in ("brier", "brier_skill", "pr_auc", "roc_auc", "cal_slope", "ece") if c in result.metrics.columns]
    st.dataframe(with_readable_index(result.metrics[columns]).style.format("{:.3f}"))

    with st.expander("Model forecast correlation", expanded=False):
        st.caption("Pairwise correlation of the forecast probabilities on the common test days. Values near 1 mean the models move together; near 0 means they are largely independent.")
        model_codes = [c for c in ["M1", "M2", "M3", "M3H", "M4", "E-core", "E-ML", "E-M1M4"] if c in result.preds.columns]
        if model_codes:
            chosen = st.multiselect("Choose forecasters to compare", model_codes, default=model_codes, key="corr_models")
            corr = result.preds[chosen].corr().copy()
            corr.index = [label(i) for i in corr.index]
            corr.columns = [label(c) for c in corr.columns]
            st.dataframe(corr.style.format("{:.3f}"))
        else:
            st.info("No model forecasts are available yet for the selected run.")

    if show_all:
        st.markdown("**Difference vs the reference model M1** (positive = better; 95% interval; "
                   "the success rule needs both intervals above zero)")
        st.dataframe(with_readable_index(result.inference))
        st.markdown("**Episode detection at matched false-alarm rates**")
        st.dataframe(with_readable_index(result.episodes))


def tab_overlay():
    if st.button("Recalculate this view", key="recalc_overlay", type="secondary"):
        force_refresh_output()
    st.markdown("**What is an overlay?** A simple rule that uses the forecast to decide how much of the portfolio to "
                "keep invested (the rest sits in cash), compared with just holding it. It answers 'could someone act "
                "on this forecast?', which is a secondary question to 'is the forecast accurate?'.")
    st.caption(f"A signal at today's close first acts {1 + int(cfg['overlay']['execution_delay_days'])} trading days later; "
               f"cost {meta['cost_bps']:.0f} bps per unit of exposure change; idle cash earns the 3-month T-bill rate.")

    strategies = list(result.overlay_paths)
    model_names = sorted({
        match.group(1)
        for key in strategies
        if key != "Buy & hold"
        for match in [re.match(r"Rule [AB] \[(.+)\]", key)]
        if match
    }, key=lambda name: ("M" not in name, name))
    default_strategies = {"Buy & hold", f"Rule B [{signal}]", "Rule B [B5]"}

    def normalize_overlay_paths(paths, start=None, end=None):
        """Restrict to the chosen dates and normalize each equity path to 1 at the first date in-range."""
        out = {}
        for key, path in paths.items():
            p = path.copy()
            if start is not None:
                p = p.loc[p.index >= pd.Timestamp(start)]
            if end is not None:
                p = p.loc[p.index <= pd.Timestamp(end)]
            if p.empty:
                continue
            if "equity" in p.columns:
                first = float(p["equity"].iloc[0])
                if pd.notna(first) and first != 0:
                    p["equity"] = p["equity"] / first
                    p["drawdown"] = p["equity"] / p["equity"].cummax() - 1.0
            out[key] = p
        return out

    plot_paths = dict(result.overlay_paths)
    all_dates = sorted({pd.Timestamp(ts).date() for path in result.overlay_paths.values() for ts in path.index})
    default_from = all_dates[0]
    default_to = all_dates[-1]
    date_from = st.date_input("Overlay start date", value=default_from, min_value=all_dates[0], max_value=all_dates[-1], key="overlay_from")
    date_to = st.date_input("Overlay end date", value=default_to, min_value=all_dates[0], max_value=all_dates[-1], key="overlay_to")
    if date_to < date_from:
        st.warning("End date is before start date; swapped automatically.")
        date_from, date_to = date_to, date_from
    plot_paths = normalize_overlay_paths(result.overlay_paths, date_from, date_to)

    chart_col, control_col = st.columns([3.0, 1.3])
    with control_col:
        all_strategy_options = {k: strategy_label(k) for k in strategies}
        chosen = simple_checkboxes("ov", all_strategy_options, default_strategies,
                                 "Choose which strategies to compare")
        chart_kind = st.radio("Overlay view", ["Portfolio growth", "Peak-to-trough loss"],
                             horizontal=True, key="ov_kind")
    with chart_col:
        if chosen:
            mapped_kind = {"Portfolio growth": "Growth of 1",
                           "Peak-to-trough loss": "Drawdown"}[chart_kind]
            st.plotly_chart(charts.overlay_chart(plot_paths, chosen, mapped_kind, strategy_label), use_container_width=True)
        else:
            st.plotly_chart(charts.empty_chart("No strategy selected"), use_container_width=True)

    measures = {key: text for group in METRICS.values() for key, text in group.items()}
    measure_col, table_col = st.columns([1.3, 2.7])
    with measure_col:
        chosen_measures = simple_checkboxes("ovm", measures, {"cagr", "max_drawdown", "avg_exposure", "pct_days_reduced"},
                                           "Choose which measures to show (Return / Risk / Activity and cost)")
        optimistic = st.toggle("Show optimistic timing (no extra delay)", value=False, key="optimistic")
    with table_col:
        stats = result.overlay_stats_delay0 if optimistic else result.overlay_stats
        if chosen and chosen_measures:
            filtered_stats = pd.DataFrame({
                key: perf_stats(path)
                for key, path in plot_paths.items()
                if key in chosen
            }).T
            if not filtered_stats.empty:
                table = filtered_stats.loc[[k for k in chosen if k in filtered_stats.index],
                                         [m for m in chosen_measures if m in filtered_stats.columns]].copy()
                table.index = [strategy_label(i) for i in table.index]
                table.columns = [measures[c] for c in table.columns]
                st.dataframe(table.style.format("{:.3f}"))
    st.caption("The chart and summary are normalized to 1 at the selected start date, so the comparison is period-specific. "
                "Default: the full overlay path currently displayed in the app.")


def tab_trust():
    if st.button("Recalculate this view", key="recalc_trust", type="secondary"):
        force_refresh_output()
    st.caption("Can the forecasts be trusted? Left: when it said X%, did it happen about X% of the time? "
               "Right: does ignoring the least-sure days improve accuracy?")
    left, right = st.columns(2)
    table = reliability_table(result.y, preds[signal])
    if len(table):
        left.plotly_chart(charts.reliability(table))
    right.plotly_chart(charts.coverage_risk(result.coverage_risk))

    s = meta["abstain_summary"]
    st.markdown(f"The low-confidence flag fired on **{s['abstain_rate']:.0%}** of days. Brier score on unflagged days "
               f"**{s['brier_kept']:.3f}**, on flagged days **{s['brier_abstained']:.3f}**, overall {s['brier_all']:.3f}.")
    st.dataframe(result.flags.mean().rename("share of days").to_frame().style.format("{:.1%}"))

    st.markdown("Recalibration chosen per forecaster (on the development period):")
    st.dataframe(with_readable_index(pd.DataFrame(meta["calibration"]).T))


# 5. Show the tabs

tabs = st.tabs(["Current risk", "Volatility and forecasts", "Episodes and drivers", "Accuracy", "Overlay backtest", "Trust"])
for tab, show in zip(tabs, [tab_current_risk, tab_volatility, tab_episodes, tab_accuracy, tab_overlay, tab_trust]):
    with tab:
        show()
