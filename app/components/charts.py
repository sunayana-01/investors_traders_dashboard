"""Plotly charts. Each function takes pandas tables and returns a figure."""
import re

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from riskml.episodes import find_episodes

from .names import short

BASE_LAYOUT = dict(margin=dict(l=10, r=10, t=50, b=10), height=340, legend=dict(orientation="h", y=-0.25))
FORECAST_COLORS = {
    "B0": "#2E4057", "B1": "#5A9BD5", "B2": "#F4A261", "B3": "#7C9CBF",
    "B4": "#A78BFA", "B5": "#F7B267", "M1": "#2A9D8F", "M2": "#E76F51",
    "M3": "#E9C46A", "M3H": "#FF9F1C", "M4": "#8D99AE", "E-core": "#D62828",
    "E-ML": "#06D6A0", "E-M1M4": "#8338EC",
}


def dark_axes(fig):
    fig.update_layout(title_font=dict(color="#111827"), font=dict(color="#111827"))
    fig.update_xaxes(color="#111827", title_font=dict(color="#111827"), tickfont=dict(color="#111827"),
                     gridcolor="#e5e7eb", zerolinecolor="#d1d5db", linecolor="#111827")
    fig.update_yaxes(color="#111827", title_font=dict(color="#111827"), tickfont=dict(color="#111827"),
                     gridcolor="#e5e7eb", zerolinecolor="#d1d5db", linecolor="#111827")
    fig.update_annotations(font=dict(color="#111827"))
    for ann in fig.layout.annotations:
        ann.font = dict(color="#111827")
    return fig


def new_figure(title, y_title=None, **layout):
    merged = {**BASE_LAYOUT, **layout}
    if "legend" in merged:
        legend = merged["legend"]
        if isinstance(legend, dict):
            legend = {**legend, "font": {**legend.get("font", {}), "color": "#111827"}}
            merged["legend"] = legend
    else:
        merged["legend"] = dict(font=dict(color="#111827"), orientation="h", y=-0.25)
    fig = go.Figure()
    fig.update_layout(template="plotly_white", title=title,
                     paper_bgcolor="rgba(255,255,255,0.94)", plot_bgcolor="rgba(255,255,255,0.9)",
                     font=dict(color="#111827"),
                     title_font=dict(color="#111827"),
                     **merged)
    if y_title:
        fig.update_yaxes(title=y_title)
    return dark_axes(fig)


def empty_chart(title="No forecast selected"):
    fig = new_figure(title, "", height=420)
    fig.update_xaxes(showgrid=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, showticklabels=False)
    return fig


def color_for(code):
    return FORECAST_COLORS.get(code, "#4C78A8")


def recent_path(forecasts, codes, signal, base_rate):
    """The latest forecasts of the chosen forecasters (the main signal is drawn thicker)."""
    if not codes and signal not in forecasts.columns:
        return empty_chart("No forecast selected")
    fig = new_figure("Forecast for the most recent days (outcome not yet known)",
                     "chance of elevated volatility", height=620, yaxis_range=[0, 1], yaxis_tickformat=".0%")
    fig.update_xaxes(title="Date")
    for code in dict.fromkeys([*codes, signal]):
        if code in forecasts:
            fig.add_scatter(x=forecasts.index, y=forecasts[code], mode="lines", name=short(code),
                           line=dict(width=4 if code == signal else 1.8, color=color_for(code)))
    fig.add_hline(y=base_rate, line_dash="dot", annotation_text="how often this happened in the past",
                 line_color="#556", annotation_position="left top")
    return fig


def vol_and_prob(vol, preds, codes, h, show_rv21=False, show_vix=False):
    """Top: realised volatility and the threshold. Bottom: the forecasts. Red = elevated periods."""
    if not codes:
        return empty_chart("No forecast selected")
    vol = vol.loc[preds.index.min():preds.index.max()]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5], vertical_spacing=0.08,
                       subplot_titles=(f"What happened: realised portfolio volatility over the NEXT {h} trading days (annualised)",
                                       "What the forecasters said beforehand: chance of 'elevated' volatility"))
    fig.update_layout(template="plotly_white", paper_bgcolor="rgba(255,255,255,0.94)", plot_bgcolor="rgba(255,255,255,0.9)",
                     height=640, margin=dict(l=10, r=10, t=70, b=10),
                     font=dict(color="#111827"), legend=dict(font=dict(color="#111827"), orientation="h", y=-0.12))
    fig.add_scatter(x=vol.index, y=vol["fwd_rv"], name=f"Realised volatility, next {h} days",
                   line=dict(width=2, color="#1f77b4"), row=1, col=1)
    fig.add_scatter(x=vol.index, y=vol["threshold"], name="'Elevated' threshold (75th pct of the past)",
                   line=dict(dash="dash", shape="hv", color="#d62728"), row=1, col=1)
    if show_rv21:
        fig.add_scatter(x=vol.index, y=vol["rv_21"], name="Volatility over the PAST 21 days",
                       line=dict(width=1.5, color="#7f7f7f"), row=1, col=1)
    if show_vix:
        fig.add_scatter(x=vol.index, y=vol["vix"], name="VIX / 100 (market's expected volatility)",
                       line=dict(width=1.5, color="#9467bd"), row=1, col=1)

    episodes = find_episodes(vol["y"].fillna(0), min_len=1, merge_gap=5)
    for start, end in zip(episodes["start"], episodes["end"]):
        fig.add_vrect(x0=start, x1=end, fillcolor="red", opacity=0.12, line_width=0, row="all", col="all")

    for code in codes:
        if code in preds:
            fig.add_scatter(x=preds.index, y=preds[code], name=short(code), line=dict(width=1.8, color=color_for(code)), row=2, col=1)

    fig.update_yaxes(title_text="annualised volatility", tickformat=".0%", row=1, col=1)
    fig.update_yaxes(title_text="chance of elevated volatility", range=[0, 1], tickformat=".0%", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    buttons = [dict(count=1, label="1y", step="year", stepmode="backward"),
               dict(count=3, label="3y", step="year", stepmode="backward"), dict(step="all")]
    fig.update_xaxes(rangeselector=dict(buttons=buttons), row=1, col=1)
    fig.for_each_annotation(lambda a: a.update(font=dict(color="#111827")))
    return dark_axes(fig)


def contrib_area(contrib, assets, h):
    fig = new_figure(f"Which holding drove the realised risk over the next {h} days (after the fact)",
                     "share of portfolio variance", height=360, yaxis_tickformat=".0%")
    fig.update_xaxes(title="Date")
    for asset in assets:
        fig.add_scatter(x=contrib.index, y=contrib[asset], name=asset, stackgroup="one", line=dict(width=0.5))
    return fig


def episode_bars(shares, title):
    fig = new_figure(title, "share of realised risk", height=300, yaxis_tickformat=".0%")
    fig.update_xaxes(title="holding", tickfont=dict(color="#111827"), title_font=dict(color="#111827"))
    fig.update_yaxes(tickfont=dict(color="#111827"), title_font=dict(color="#111827"))
    fig.update_layout(legend=dict(font=dict(color="#111827")))
    colours = ["#d62728" if v == shares.max() else "#7f7f7f" for v in shares.values]
    fig.add_bar(x=list(shares.index), y=list(shares.values), marker_color=colours)
    return fig


def reliability(table):
    fig = new_figure("Reliability: what was forecast vs how often it happened", "how often it happened")
    fig.update_xaxes(title="average forecast", range=[0, 1], title_font=dict(color="#111827"), tickfont=dict(color="#111827"))
    fig.update_yaxes(title="observed rate", range=[0, 1], title_font=dict(color="#111827"), tickfont=dict(color="#111827"))
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect", line=dict(dash="dot"))
    fig.add_scatter(x=table["mean_p"], y=table["frac_pos"], mode="markers+lines", name="forecaster",
                   marker=dict(size=table["n"] / table["n"].max() * 20 + 5), text=table["n"].map(lambda n: f"n={n}"))
    return dark_axes(fig)


def coverage_risk(curve):
    fig = new_figure("Does ignoring the least-sure forecasts help? (lower = more accurate)", "Brier score")
    fig.add_scatter(x=curve["coverage"], y=curve["brier"], mode="lines+markers", name="Brier")
    fig.update_xaxes(title="share of days kept", title_font=dict(color="#111827"), tickfont=dict(color="#111827"))
    fig.update_yaxes(title_font=dict(color="#111827"), tickfont=dict(color="#111827"))
    return dark_axes(fig)


def overlay_chart(paths, chosen, kind, label_fn):
    if not chosen:
        return empty_chart("No strategy selected")
    column = {"Growth of 1": "equity", "Drawdown": "drawdown", "Share invested": "exposure"}[kind]
    title = {"equity": "Growth of 1 (after costs; idle cash earns the T-bill rate)",
             "drawdown": "Drawdown: loss from the previous peak",
             "exposure": "How much of the portfolio was invested"}[column]
    fig = new_figure(title, column, height=380, yaxis_tickformat=None if column == "equity" else ".0%")
    fig.update_xaxes(title="Date")
    for key in chosen:
        # key looks like "Rule A [M1]": the colour comes from the signal, Rule B is drawn dashed
        match = re.search(r"\[(.+)\]", key)
        colour = color_for(match.group(1)) if match else "#111827"   # buy and hold is dark grey
        fig.add_scatter(x=paths[key].index, y=paths[key][column], mode="lines", name=label_fn(key),
                        line=dict(color=colour, dash="dash" if key.startswith("Rule B") else "solid"))
    return fig
