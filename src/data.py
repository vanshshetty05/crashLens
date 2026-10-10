"""Load the raw MoRTH CSV for CrashLens.

One row is one State/UT x one category (factor_value) x one year. The values are
aggregate counts, not individual crashes.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "india_road_accidents_2019_2024.csv"

TARGET = "severity_level"
SEVERITY_ORDER = ["Low", "Medium", "High"]

# Columns that determine the target. They must never be used as model features.
LEAKAGE_COLS = [
    "killed",
    "severity_index",
    "national_severity",
    "grievously_injured",
    "minor_injured",
    "total_injured",
]

# Every factor group is a separate breakdown of the same accidents, so summing
# across groups counts each accident up to nine times. National, regional and
# state totals must be taken from a single group. collision_type covers all
# accidents in every year, so it is the reference group for totals.
TOTALS_GROUP = "collision_type"

EXPECTED_COLS = [
    "year", "state", "state_harmonised", "state_type", "region", "annexure",
    "factor_group", "factor_value", "accidents", "killed", "grievously_injured",
    "minor_injured", "total_injured", "severity_index", "national_severity",
    "severity_level", "low_count_flag",
]


def seed_everything(seed: int = SEED) -> None:
    """Seed Python's and NumPy's global random generators.

    Every estimator, splitter and sampler in the project also takes SEED
    explicitly. This covers library code that falls back to the global generators.
    """
    random.seed(seed)
    np.random.seed(seed)


def load_data(path: str | Path = RAW_PATH, drop_low_count: bool = True) -> pd.DataFrame:
    """Read the raw CSV and return a clean DataFrame.

    Parameters
    ----------
    path : location of the raw CSV.
    drop_low_count : if True (the default, and the rule for all modelling), rows
        with ``low_count_flag`` set are removed. These rows have fewer than 10
        accidents and no ``severity_level``. Pass False only when summing
        ``accidents`` and ``killed`` into totals, where those small counts
        still belong in the sum.

    Returns
    -------
    DataFrame with ``severity_level`` as an ordered categorical
    (Low < Medium < High) and a fresh integer index.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place india_road_accidents_2019_2024.csv in data/raw/."
        )

    df = pd.read_csv(path)

    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")

    df["low_count_flag"] = df["low_count_flag"].astype(bool)
    df[TARGET] = pd.Categorical(df[TARGET], categories=SEVERITY_ORDER, ordered=True)

    if drop_low_count:
        df = df[~df["low_count_flag"]]
        if df[TARGET].isna().any():
            raise ValueError("Rows without a severity_level remain after dropping low-count rows.")

    return df.reset_index(drop=True)
