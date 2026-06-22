from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

from .config import DATA_CACHE_DIR, RESULTS_URL, SHOOTOUTS_URL


def _download(url: str, destination: Path, force: bool = False) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if force or not destination.exists():
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        try:
            urlretrieve(url, temporary)
            temporary.replace(destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    return destination


def load_results(force_download: bool = False) -> pd.DataFrame:
    path = _download(RESULTS_URL, DATA_CACHE_DIR / "results.csv", force_download)
    results = pd.read_csv(path, parse_dates=["date"])
    results["neutral"] = results["neutral"].astype(bool)
    return results.sort_values("date").reset_index(drop=True)


def filter_missing_scores(results: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    filtered = results.dropna(subset=["home_score", "away_score"]).copy()
    removed = len(results) - len(filtered)
    return filtered.reset_index(drop=True), removed


def load_shootouts(force_download: bool = False) -> pd.DataFrame:
    path = _download(SHOOTOUTS_URL, DATA_CACHE_DIR / "shootouts.csv", force_download)
    shootouts = pd.read_csv(path, parse_dates=["date"])
    return shootouts.sort_values("date").reset_index(drop=True)


def load_dataset(force_download: bool = False) -> pd.DataFrame:
    results = load_results(force_download)
    results, removed_missing_scores = filter_missing_scores(results)
    shootouts = load_shootouts(force_download)
    shootout_cols = ["date", "home_team", "away_team", "winner"]
    merged = results.merge(
        shootouts[shootout_cols],
        how="left",
        on=["date", "home_team", "away_team"],
    )
    dataset = merged.rename(columns={"winner": "shootout_winner"})
    dataset.attrs["removed_missing_scores"] = removed_missing_scores
    return dataset
