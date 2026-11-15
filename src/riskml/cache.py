"""Saves finished pipeline results so the dashboard does not have to recompute them.

Each result gets its own folder named after a key. The key is a hash of everything that
could change the result (config, weights, horizon, models, data, ...), so a changed input
always gives a new key and an old result is never shown by mistake.
Inside the folder: one parquet file per table and a meta.json.
"""
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


def data_version(prices):
    """Quick fingerprint of the price data: its shape, first and last date and the last row."""
    last_row = prices.iloc[-1].round(6).to_json() if len(prices) else ""
    text = f"{prices.shape}|{prices.index[0]}|{prices.index[-1]}|{last_row}"
    return hashlib.sha256(text.encode()).hexdigest()[:10]


def make_key(**parts):
    """Hash of all the named inputs. The order of the names does not matter."""
    text = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:20]


class ResultCache:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)

    def put(self, key, tables, meta):
        folder = self.folder / key
        folder.mkdir(parents=True, exist_ok=True)
        for name, table in tables.items():
            table = table.to_frame() if isinstance(table, pd.Series) else table
            table.to_parquet(folder / f"{name}.parquet")
        (folder / "meta.json").write_text(json.dumps(meta, default=str))   # written last: marks the entry as complete

    def get(self, key):
        """Returns (tables, meta), or None if this key is not saved (or is damaged)."""
        folder = self.folder / key
        if not (folder / "meta.json").exists():
            return None
        try:
            tables = {file.stem: pd.read_parquet(file) for file in folder.glob("*.parquet")}
            meta = json.loads((folder / "meta.json").read_text())
        except (OSError, ValueError):
            return None
        return tables, meta

    def clear(self):
        shutil.rmtree(self.folder, ignore_errors=True)
        self.folder.mkdir(parents=True, exist_ok=True)
