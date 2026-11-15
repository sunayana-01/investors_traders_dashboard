"""Diagnostics for a development-period run: is the ML gain ranking or just calibration level?"""
import warnings
import numpy as np
import pandas as pd
from riskml.evaluate import roc_auc
from riskml.pipeline import run_pipeline

warnings.filterwarnings("ignore")


def diagnose(res):
    P, y = res.preds, res.y
    cols = [c for c in ["B0", "B1", "B2", "B5", "M1", "M2", "M3", "M4", "E-core"] if c in P.columns]

    print("\n1) Label base rate and mean forecast by year (a drifting base rate favours recalibrated models)")
    by = pd.DataFrame({"base_rate": y.groupby(y.index.year).mean(),
                       **{f"mean_{c}": P[c].groupby(P.index.year).mean() for c in ["B1", "M1", "E-core"] if c in P}})
    print(by.round(2).assign(n=y.groupby(y.index.year).size()))

    print("\n2) Within-fold ROC-AUC (ranking only, no pooled-scale effects); mean over folds")
    rows = {}
    for f, r in res.folds.iterrows():
        sl = slice(r["test_start"], r["test_end"])
        yy = y.loc[sl]
        if yy.nunique() < 2:
            continue
        rows[f] = {c: roc_auc(yy, P[c].loc[sl]) for c in cols}
    auc = pd.DataFrame(rows).T
    print(f"   folds used: {len(auc)} of {len(res.folds)}")
    print(auc.mean().round(3).to_string())

    print("\n3) Share of days each abstain trigger fires")
    print(res.flags.mean().round(3).to_string())

    print("\n4) Fold table")
    print(res.folds[["train_end", "n_train", "n_calib", "test_start", "test_end", "n_test"]].to_string())


if __name__ == "__main__":
    res = run_pipeline("P1", 21, models=("M1", "M2", "M3", "M4"), ensemble="E-core", n_boot=200)
    diagnose(res)