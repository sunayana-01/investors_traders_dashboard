"""Plain-English names for the model codes and overlay strategies."""
import re

NAMES = {"B0": "Base rate (always the historical average)", "B1": "Recent-volatility rule",
         "B2": "EWMA volatility rule", "B3": "GARCH volatility rule", "B4": "HAR volatility regression",
         "B5": "VIX rule", "M1": "HAR + VIX logistic (the reference)", "M2": "All-feature logistic regression",
         "M3": "LightGBM decision trees", "M3H": "Histogram gradient boosting", "M4": "GARCH probability model",
         "E-core": "Ensemble (average of M1, M2, M3, M4)", "E-ML": "Ensemble (average of M2 and M3)",
         "E-M1M4": "Ensemble (average of M1, M3H and M4)"}

# how the forecasters are grouped in the checkbox lists
GROUPS = {"Simple rules": ["B0", "B1", "B2", "B3", "B4", "B5"],
          "Models": ["M1", "M2", "M3", "M3H", "M4"],
          "Ensembles (averages of models)": ["E-core", "E-ML", "E-M1M4"]}

# the overlay measures, grouped for the checkbox list
METRICS = {"Return": {"cagr": "Annual return (CAGR)"},
           "Risk": {"max_drawdown": "Worst peak-to-trough loss", "ann_vol": "Volatility",
                    "worst_21d": "Worst 21-day loss", "sharpe": "Sharpe ratio"},
           "Activity and cost": {"avg_exposure": "Average exposure", "pct_days_reduced": "Share of days cut back", "ann_turnover": "Yearly turnover",
                                 "ann_cost_drag": "Yearly cost drag"}}


def short(code):
    return NAMES.get(code, code)


def label(code):
    return f"{short(code)} [{code}]" if code in NAMES else code


def strategy_label(key):
    if key == "Buy & hold":
        return "Hold only (no overlay)"
    match = re.match(r"Rule ([AB]) \[(.+)\](?: \+ liquidation)?", key)
    if not match:
        return key
    rule, signal = match.groups()
    what = "Cut to 50% when risk is high" if rule == "A" else "Scale exposure down as risk rises"
    if "liquidation" in key:
        what += " with forced liquidation"
    return f"{what} - signal: {short(signal)}"


def verdict(result):
    """One short paragraph that summarises the accuracy table.

    It is built from the confidence intervals, so it cannot claim more than the numbers show.
    """
    signal, metrics = result.meta["signal"], result.metrics
    simple_rules = [c for c in ["B1", "B2", "B3", "B4", "B5"] if c in metrics.index]
    sentences = []

    if signal == "M1":
        sentences.append("The signal shown is the reference model itself, so there is nothing to compare it with.")
    else:
        try:
            brier = result.inference.loc[(signal, "brier")]
            pr_auc = result.inference.loc[(signal, "pr_auc")]
            if brier["ci_low"] > 0 and pr_auc["ci_low"] > 0:
                sentences.append(f"**{short(signal)}** beat the reference model (M1) on both accuracy measures, "
                                 "and both gaps are larger than chance.")
            elif brier["ci_high"] < 0 and pr_auc["ci_high"] < 0:
                sentences.append(f"**{short(signal)}** did *worse* than the reference model (M1) on both "
                                 "accuracy measures, by more than chance.")
            else:
                sentences.append(f"**{short(signal)}** is **not clearly different** from the reference model (M1): "
                                 "the gap is within what chance could produce.")
        except KeyError:
            sentences.append("No interval comparison against M1 is available for this signal.")

    if simple_rules:
        best = min(simple_rules, key=lambda c: metrics.loc[c, "brier"])
        sentences.append(f"Lowest Brier score among the simple rules: **{short(best)}** "
                         f"({metrics.loc[best, 'brier']:.3f}); the selected signal scored "
                         f"{metrics.loc[signal, 'brier']:.3f} (lower is better; no interval computed for this pair).")

    sentences.append("Development period only; the final holdout has not been used.")
    return " ".join(sentences)
