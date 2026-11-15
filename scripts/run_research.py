"""Runs the research experiments for the three benchmark portfolios.

  python scripts/run_research.py --stage dev
      P1-P3 at every horizon, development period only (before 2019).
      Prints the calibration choices we suggest to freeze in config.yaml.

  python scripts/run_research.py --stage holdout --confirm-frozen
      The final test on 2019 and later. It refuses to run unless
        (a) --confirm-frozen is given,
        (b) config.yaml has a calibration.selected entry for every model, and
        (c) it was not run before (results/holdout/RUN.lock).
"""
import argparse
import json
import time
import warnings
from collections import Counter
from pathlib import Path

import pandas as pd

from riskml.config import config_hash, get_config, project_path
from riskml.pipeline import BASELINES, MAIN_MODELS, NEEDS_CALIBRATION, PipelineError, run_pipeline

warnings.filterwarnings("ignore")


def save_result(result, folder):
    """Write every table of a result as csv/parquet files."""
    folder.mkdir(parents=True, exist_ok=True)
    result.preds.assign(y=result.y).to_parquet(folder / "preds.parquet")
    for name in ["metrics", "inference", "episodes", "overlay_stats", "overlay_stats_delay0",
                 "flags", "coverage_risk", "folds"]:
        getattr(result, name).to_csv(folder / f"{name}.csv")
    (folder / "meta.json").write_text(json.dumps(result.meta, indent=2, default=str))


def summary_row(pid, h, result):
    """One line of the summary table for a portfolio and horizon."""
    metrics, inference, overlay = result.metrics, result.inference, result.overlay_stats
    row = {"portfolio": pid, "h": h, "folds": result.meta["n_folds"],
           "test_start": result.meta["test_start"], "test_end": result.meta["test_end"],
           "base_rate": round(float(result.y.mean()), 3)}

    for code in ["B1", "B5", "M1", "E-core"]:
        if code in metrics.index:
            row[f"brier_{code}"] = round(float(metrics.loc[code, "brier"]), 4)
            row[f"pr_auc_{code}"] = round(float(metrics.loc[code, "pr_auc"]), 3)

    if "E-core" in inference.index.get_level_values(0):
        brier = inference.loc[("E-core", "brier")]
        pr_auc = inference.loc[("E-core", "pr_auc")]
        row.update(d_brier=round(float(brier["point"]), 4), d_brier_lo=round(float(brier["ci_low"]), 4),
                   d_brier_hi=round(float(brier["ci_high"]), 4),
                   d_prauc=round(float(pr_auc["point"]), 3), d_prauc_lo=round(float(pr_auc["ci_low"]), 3),
                   d_prauc_hi=round(float(pr_auc["ci_high"]), 3),
                   criterion_metrics=bool(brier["criterion"]) and bool(pr_auc["criterion"]))

    row["maxdd_hold"] = round(float(overlay.loc["Buy & hold", "max_drawdown"]), 3)
    rule_b = [name for name in overlay.index if name.startswith("Rule B")]
    if rule_b:
        row["maxdd_ruleB"] = round(float(overlay.loc[rule_b[0], "max_drawdown"]), 3)
        row["cagr_cost_ruleB"] = round(float(overlay.loc[rule_b[0], "cagr"] - overlay.loc["Buy & hold", "cagr"]), 4)
    return row


def suggest_calibration(results):
    """For each model, the calibration method chosen most often across the portfolios. Ties go to the simpler method."""
    votes = {}
    for result in results.values():
        for code, info in result.meta["calibration"].items():
            if code in NEEDS_CALIBRATION:
                votes.setdefault(code, Counter())[info["method"]] += 1
    simplicity = {"none": 0, "platt": 1, "isotonic": 2}
    return {code: sorted(counter.items(), key=lambda kv: (-kv[1], simplicity[kv[0]]))[0][0]
            for code, counter in votes.items()}


def run_dev(cfg, portfolios, horizons, n_boot, out_folder, aligned_for=None, use_cache=True):
    rows, primary = [], {}
    for pid in portfolios:
        for h in horizons:
            started = time.time()
            try:
                result = run_pipeline(pid, h, models=MAIN_MODELS, ensemble="E-core", n_boot=n_boot,
                                      aligned=aligned_for(pid) if aligned_for else None,
                                      use_cache=use_cache, cfg=cfg)
            except PipelineError as error:
                print(f"  {pid} h={h}: skipped ({error})")
                rows.append({"portfolio": pid, "h": h, "error": str(error)})
                continue
            save_result(result, out_folder / "dev" / pid / f"h{h}")
            rows.append(summary_row(pid, h, result))
            if h == cfg["labels"]["primary_horizon"]:
                primary[pid] = result
            print(f"  {pid} h={h}: {time.time() - started:.0f}s  "
                  f"Brier M1={rows[-1].get('brier_M1')}  E-core={rows[-1].get('brier_E-core')}")

    summary = pd.DataFrame(rows)
    (out_folder / "dev").mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_folder / "dev" / "summary.csv", index=False)
    suggestion = suggest_calibration(primary) if primary else {}
    (out_folder / "dev" / "suggested_calibration.json").write_text(json.dumps(suggestion, indent=2))
    return summary, suggestion


def run_holdout(cfg, portfolios, out_folder, n_boot, confirmed, force=False, aligned_for=None):
    lock_file = out_folder / "holdout" / "RUN.lock"
    if not confirmed:
        raise SystemExit("Refusing to run: pass --confirm-frozen to confirm the configuration is frozen.")

    frozen = cfg["calibration"].get("selected") or {}
    missing = [code for code in NEEDS_CALIBRATION if code not in frozen]
    if missing:
        raise SystemExit(f"Refusing to run: calibration.selected is missing {missing}. Run the dev stage, "
                         "copy suggested_calibration.json into config.yaml and try again.")
    if lock_file.exists() and not force:
        raise SystemExit(f"Refusing to run: the holdout was already evaluated ({lock_file.read_text().strip()}). "
                         "Use --force only for a documented rerun.")

    h = cfg["labels"]["primary_horizon"]   # the formal claim is made at the main horizon only
    rows = []
    for pid in portfolios:
        result = run_pipeline(pid, h, models=MAIN_MODELS, ensemble="E-core", n_boot=n_boot, period="holdout",
                              allow_holdout=True, aligned=aligned_for(pid) if aligned_for else None,
                              use_cache=False, cfg=cfg)
        save_result(result, out_folder / "holdout" / pid / f"h{h}")
        rows.append(summary_row(pid, h, result))

    summary = pd.DataFrame(rows)
    summary.to_csv(out_folder / "holdout" / "summary.csv", index=False)
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    lock_file.write_text(f"config_hash={config_hash(cfg)} at {pd.Timestamp.now().isoformat(timespec='seconds')}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["dev", "holdout"], default="dev")
    parser.add_argument("--portfolios", nargs="+", default=["P1", "P2", "P3"])
    parser.add_argument("--horizons", nargs="+", type=int, default=None)
    parser.add_argument("--n-boot", type=int, default=None)
    parser.add_argument("--confirm-frozen", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = get_config()
    out_folder = project_path(cfg["project"]["results_dir"])
    print(f"config hash {config_hash(cfg)} | stage {args.stage}")

    if args.stage == "dev":
        horizons = args.horizons or list(cfg["labels"]["horizons"])
        summary, suggestion = run_dev(cfg, args.portfolios, horizons, args.n_boot, out_folder)
        print("\n", summary.to_string(index=False))
        print(f"\nSuggested calibration.selected (majority over portfolios at h={cfg['labels']['primary_horizon']}):")
        print("calibration:\n  selected:")
        for code, method in suggestion.items():
            print(f"    {code}: {method}")
    else:
        summary = run_holdout(cfg, args.portfolios, out_folder, args.n_boot, args.confirm_frozen, args.force)
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
