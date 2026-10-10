"""Build the model feature matrix, target and groups for CrashLens.

Leakage columns (killed, severity_index, national_severity and the injury
counts) define the target, so they are never allowed into the features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import LEAKAGE_COLS, TARGET

CATEGORICAL_FEATURES = ["factor_group", "factor_value", "region", "state_type", "state_harmonised"]
NUMERIC_FEATURES = ["year", "log_accidents"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
GROUP_COL = "state_harmonised"

TRAIN_YEARS = (2019, 2020, 2021, 2022)
TEST_YEARS = (2023, 2024)


def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return ``X``, ``y`` and ``groups`` for modelling.

    - Rows with ``low_count_flag`` set are dropped.
    - ``X`` holds exactly the columns in ``FEATURES``; ``log_accidents`` is
      ``log1p(accidents)``.
    - ``y`` is ``severity_level``.
    - ``groups`` is ``state_harmonised``, for GroupKFold.

    The three outputs share one index, which is the index of the kept rows of ``df``.
    """
    df = df[~df["low_count_flag"].astype(bool)]

    if df[TARGET].isna().any():
        raise ValueError("severity_level is missing for rows that are not flagged as low count.")

    X = df[CATEGORICAL_FEATURES + ["year"]].copy()
    X["log_accidents"] = np.log1p(df["accidents"].astype(float))
    X = X[FEATURES]

    leaked = sorted(set(X.columns) & set(LEAKAGE_COLS))
    if leaked:
        raise ValueError(f"Leakage columns found in the feature matrix: {leaked}")

    y = df[TARGET].copy()
    groups = df[GROUP_COL].copy()
    return X, y, groups


def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by time: train on 2019 to 2022, test on 2023 to 2024.

    Returns ``(train_df, test_df)``. Pass each one to ``prepare``.
    """
    train = df[df["year"].isin(TRAIN_YEARS)]
    test = df[df["year"].isin(TEST_YEARS)]
    return train, test


def make_preprocessor() -> ColumnTransformer:
    """One-hot encode the categorical features and standardise the numeric ones.

    ``handle_unknown="ignore"`` is required, not optional. Some categories exist
    only in some years: junction_type and traffic_control report "Others" only in
    2019 to 2021, and road_environment has three categories in 2024. A group-aware
    split also puts unseen states in the test fold. An unseen category is encoded
    as all zeros instead of raising an error.

    Fit this on the training data only.
    """
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ],
        remainder="drop",
    )
