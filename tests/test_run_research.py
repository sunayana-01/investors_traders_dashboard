import copy
import pathlib
import sys

import pytest
import yaml

from helpers import make_aligned
from riskml.config import DEFAULT_CONFIG, get_config, load_config

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import run_research as research   # noqa: E402

DATA = make_aligned()


def test_dev_stage_and_holdout_guards(tmp_path):
    cfg = get_config()
    summary, suggestion = research.run_dev(cfg, ["P1", "P2"], [21], 50, tmp_path,
                                           aligned_for=lambda pid: DATA, use_cache=False)
    assert len(summary) == 2 and {"brier_M1", "brier_E-core", "d_brier_lo", "maxdd_hold"} <= set(summary.columns)
    assert (tmp_path / "dev" / "P1" / "h21" / "preds.parquet").exists()
    assert set(suggestion) >= {"M1", "M3", "B1", "B5"}

    with pytest.raises(SystemExit):     # not confirmed
        research.run_holdout(cfg, ["P1"], tmp_path, 50, confirmed=False, aligned_for=lambda p: DATA)
    not_frozen = copy.deepcopy(cfg)     # a config without frozen calibration choices
    not_frozen["calibration"]["selected"] = {}
    with pytest.raises(SystemExit):
        research.run_holdout(not_frozen, ["P1"], tmp_path, 50, confirmed=True, aligned_for=lambda p: DATA)


def test_holdout_runs_only_once(tmp_path):
    raw = yaml.safe_load(open(DEFAULT_CONFIG))
    raw["calibration"]["selected"] = {code: "none" for code in research.NEEDS_CALIBRATION}
    frozen_file = tmp_path / "frozen.yaml"
    frozen_file.write_text(yaml.safe_dump(raw))
    cfg = load_config(frozen_file)

    summary = research.run_holdout(cfg, ["P1"], tmp_path, 50, confirmed=True, aligned_for=lambda p: DATA)
    assert summary.loc[0, "test_start"] >= "2019-01-01"
    assert (tmp_path / "holdout" / "RUN.lock").exists()
    with pytest.raises(SystemExit):     # a second run is refused
        research.run_holdout(cfg, ["P1"], tmp_path, 50, confirmed=True, aligned_for=lambda p: DATA)
