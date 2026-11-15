"""Quick check with real data: portfolio P1 at h=21, development period only."""
import time
import warnings

from riskml.pipeline import run_pipeline

warnings.filterwarnings("ignore")
started = time.time()
result = run_pipeline("P1", 21, models=("M1", "M2", "M3", "M4"), ensemble="E-core", n_boot=200, use_cache=False)
meta = result.meta

print(f"{meta['portfolio']} | h={meta['horizon']} | folds={meta['n_folds']} | "
      f"test {meta['test_start']}..{meta['test_end']} | {time.time() - started:.0f}s")
print("data:", meta["data_start"], "->", meta["data_end"])
print("calibration:", {code: info["method"] for code, info in meta["calibration"].items()})
print(result.metrics[["brier", "brier_skill", "pr_auc", "roc_auc", "cal_slope", "ece"]].round(3))
print(result.inference.loc["E-core"].round(3))
print(result.overlay_stats[["cagr", "max_drawdown", "avg_exposure", "ann_cost_drag"]].round(3))
print("low-confidence summary:", {k: round(v, 3) for k, v in meta["abstain_summary"].items()})
