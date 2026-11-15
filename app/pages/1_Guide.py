"""Plain-English guide for readers who are new to the project."""
import streamlit as st

st.set_page_config(page_title="Guide", layout="wide")
st.title("Guide: what you are looking at")
st.caption("Written for someone with no finance or machine-learning background.")

st.header("In 60 seconds")
st.markdown(
    "- A portfolio is a mix of funds (for example stocks, bonds and gold).\n"
    "- Its **volatility** is how violently its value swings. Calm months have low volatility; crashes have high volatility.\n"
    "- This project asks: **can a computer, using only information available at the time, warn that the next month will be unusually "
    "volatile, and do it better than simple rules like 'the VIX is high' or 'it has been choppy lately'?**\n"
    "- Every forecaster outputs one number per day: the **chance that the next 21 trading days (about a month) will be 'elevated'**.\n"
    "- 'Elevated' means volatility over that month beats the 75th percentile of the portfolio's own past (roughly 'among its most "
    "volatile quarter of months so far').\n"
    "- A fair result can be negative: it is fine to find that simple rules are just as good.")

st.header("A worked example")
st.markdown(
    "On a given Friday a forecaster says **40%**. We then wait 21 trading days and measure the realised volatility. "
    "If it ended up above the threshold, the outcome is 1 ('elevated'), otherwise 0. Doing this on thousands of days, with the "
    "forecaster never allowed to see the days after the forecast, lets us measure honestly how good it is.")

st.header("How to read each tab")
for title, body in {
    "Current risk": "The latest forecast. The coloured box says whether any warning sign fired. The line chart compares forecasters.",
    "Volatility and forecasts": "Top panel: what really happened to volatility (red = elevated). Bottom panel: what each forecaster said "
                                "beforehand. Good forecasters are high in the red zones and low outside them.",
    "Episodes and drivers": "After the fact: for each elevated stretch, which holding contributed most to the realised swings, and did each "
                            "forecaster warn in advance? The models do not forecast per holding; this is a description of what happened.",
    "Accuracy": "Scores on data the models never saw. Lower Brier = better. Higher PR-AUC = better. The sentence at the top is generated "
                "from confidence intervals so it cannot claim more than the numbers support.",
    "Overlay backtest": "A rule that moves money to cash when risk looks high, versus just holding. Shows return, drawdown and how much "
                        "trading it takes. Costs and a delay between signal and action are included.",
    "Trust": "Reliability (did '30%' happen about 30% of the time?) and whether ignoring the least-sure days improves accuracy."}.items():
    with st.expander(title):
        st.write(body)

st.header("Glossary")
GLOSSARY = {
    "Volatility": "How much returns swing, usually measured as the standard deviation of daily returns, scaled to a yearly number.",
    "Drawdown": "How far the portfolio is below its previous peak. A drawdown of -20% means it lost 20% from the high point.",
    "Realised volatility": "Volatility actually measured over a stretch of past days.",
    "VIX": "A market index of how much volatility investors expect over the next month, built from stock-index option prices.",
    "Elevated volatility (the target)": "Volatility over the next horizon above the 75th percentile of the portfolio's own earlier history.",
    "Base rate": "How often 'elevated' happened in the past. A forecaster that always says this number knows nothing else.",
    "Probability forecast": "A number between 0 and 1 (or 0% to 100%), not a yes/no.",
    "Brier score": "Average squared gap between the forecast and what happened (0 or 1). Lower is better. Confident wrong forecasts are punished most.",
    "PR-AUC": "How well the forecaster's highest-risk days line up with the truly elevated days, balancing false alarms against misses. Higher is better.",
    "ROC-AUC": "Chance a randomly chosen elevated day gets a higher forecast than a randomly chosen calm day. 0.5 = coin flip.",
    "False alarm / miss": "False alarm: warned, but the month was calm. Miss: did not warn, but volatility spiked.",
    "Calibration": "Whether forecasts mean what they say: of all days forecast at 30%, about 30% should turn out elevated.",
    "Recalibration (none / Platt / isotonic)": "A final correction step that adjusts raw forecasts to match how often events happened recently.",
    "Out-of-sample / walk-forward": "Models are trained only on the past, then tested on the next block of days they never saw, then the window moves forward.",
    "Look-ahead / leakage": "Accidentally using future information. Everything here is built to prevent it, and tests check it.",
    "Development period vs holdout": "Before 2019 is used to build and compare. 2019 onward is kept untouched for one final test.",
    "Overlay": "A rule that changes how much of the portfolio is invested based on the forecast.",
    "Rule A / Rule B": "A: cut to 50% invested when the forecast is above a threshold. B: invest less the higher the forecast, never below 20%.",
    "Transaction cost (bps)": "A basis point is 0.01%. The cost paid each time exposure changes.",
    "Ensemble": "An average of several models' forecasts. E-core averages M1, M2, M3 and M4; E-ML averages M2 and M3; E-M1M4 averages M1, M3H and M4.",
    "Low-confidence flag (abstain)": "Raised when models disagree a lot, inputs look unusual, the forecast is near the action threshold, or recent forecasts were miscalibrated.",
    "B0-B5 (simple rules)": "B0 base rate; B1 recent volatility; B2 recent volatility weighting new days more; B3 GARCH statistical model; B4 HAR regression; B5 VIX.",
    "M1-M4 / M3H (models)": "M1 logistic regression on volatility and VIX (the reference to beat); M2 logistic regression with about 25 inputs; M3 boosted decision trees (LightGBM); M3H histogram gradient boosting; M4 GARCH-based probability.",
    "Episode": "A run of elevated days (at least 5, merging gaps of up to 10 days).",
    "Risk contribution": "A holding's share of the portfolio's realised variance: weight x its co-movement with the portfolio, divided by total variance. Shares add to 100%; a diversifier can be negative.",
}
for k, v in GLOSSARY.items():
    with st.expander(k):
        st.write(v)

st.header("What the results can and cannot show")
st.markdown(
    "- **Can show:** whether a forecaster separates calm from turbulent months better than simple rules, on data it never saw.\n"
    "- **Cannot show:** that it predicts returns, beats the market, or will keep working. Few independent crises occur in any sample, "
    "so conclusions are tentative.\n"
    "- Custom portfolios are exploratory: all research claims were tested only on three fixed benchmark portfolios.\n"
    "- The expanding threshold means the share of 'elevated' days is not 25% in practice (it depends on the portfolio and era).\n"
    "- Data are adjusted closes from free public sources (Yahoo Finance, FRED) and may contain errors.\n"
    "- This is a course project, not investment advice.")