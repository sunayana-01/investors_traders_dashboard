# Can machine learning flag elevated portfolio risk?

Course project. For a portfolio of funds (for example 60% SPY, 30% IEF, 10% GLD) we forecast the
chance that the next 21 trading days will be unusually volatile, and check honestly whether
machine learning does better than simple rules such as "the VIX is high".

## Run it

```
pip install -r requirements.txt
pip install -e .                      # makes `import riskml` work
streamlit run app/Home.py             # the dashboard
python scripts/smoke_test.py          # quick run on real data (about 10 seconds)
python scripts/run_research.py --stage dev     # all benchmark portfolios and horizons
pytest                                # tests (no internet needed)
```

Prices and macro data are saved in `data/raw`, so after the first run everything also works offline.
Tick "Refresh market data" in the dashboard to download fresh data.

## How the code is organised

The code follows the order in which the project works. `pipeline.py` calls the others in this order.

| Step | File | What it does |
|---|---|---|
| settings | `config/config.yaml`, `config.py` | every setting in one place; a hash of it is saved with each result |
| 1. data | `data.py` | downloads prices, VIX and macro data; puts them on one calendar; delays macro data by its publication lag; sanity checks |
| 2. portfolio | `portfolio.py` | checks the weights; computes the daily portfolio return |
| 3. features | `features.py` | 28 inputs for the models (volatility, trend, correlation, VIX, macro), all using past data only |
| 4. labels | `labels.py` | the target: 1 if the next h days are more volatile than the 75th percentile of what was known so far |
| 5. splits | `splits.py` | walk-forward folds: train on the past, test on the next ~6 months, with purging |
| 6. models | `models.py` | baselines B0-B5 and models M1-M4 (all with `fit` and `predict_proba`) |
| 7. calibration | `calibration.py` | corrects probabilities so "30%" means about 30% (none / Platt / isotonic) |
| 8. ensemble | `ensemble.py` | averages models in log-odds space |
| 9. evaluation | `evaluate.py`, `inference.py` | Brier, PR-AUC, ROC-AUC, calibration; bootstrap and Diebold-Mariano tests against M1 |
| 10. use | `overlay.py` | backtest: reduce exposure when the forecast is high, after costs and with a delay |
| extras | `episodes.py`, `trust.py`, `attribution.py` | did we catch the volatile episodes; when not to trust a forecast; which holding drove the risk |
| glue | `pipeline.py`, `cache.py` | runs all steps; saves finished results |

Dashboard: `app/Home.py` (one function per tab), `app/pages/1_Guide.py` (plain-English guide),
`app/components/` (charts, names, widgets).

## Rules that prevent cheating (all covered by tests in `tests/test_no_leakage.py`)

* Features at day t only use data up to day t.
* The label threshold at day t only uses volatility values that were already known at day t.
* A training row is only used if its label was finished before the test block starts (purging).
* Macro data is delayed by its publication lag.
* The holdout period (2019 on) is evaluated once, with calibration choices frozen in the config.
* A forecast made at the close of day t acts on exposure two days later in the main overlay result.

## Models in one line each

* **M1** logistic regression on log 5/21/63-day volatility and log VIX. The reference to beat.
* **M2** logistic regression on all 28 features.
* **M3** LightGBM (small boosted trees) on all 28 features.
* **M3H** scikit-learn histogram gradient boosting on all 28 features (a second boosted-tree model).
* **M4** GARCH(1,1) volatility forecast turned into a probability with a normal distribution.
* **E-core** average of M1, M2, M3 and M4. **E-ML** average of M2 and M3. **E-M1M4** average of M1, M3H and M4.
# traders_dasboard
