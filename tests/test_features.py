"""Tests for src/features.py. The leakage tests must fail if a target-defining column reaches X."""
import numpy as np
import pandas as pd
import pytest

from src.data import LEAKAGE_COLS, RAW_PATH, TARGET, load_data
from src.features import (
    CATEGORICAL_FEATURES,
    FEATURES,
    TEST_YEARS,
    TRAIN_YEARS,
    make_preprocessor,
    prepare,
    time_split,
)

FORBIDDEN = {"killed", "severity_index", "national_severity",
             "grievously_injured", "minor_injured", "total_injured"}

needs_data = pytest.mark.skipif(not RAW_PATH.exists(), reason="raw CSV not in data/raw")


def make_frame() -> pd.DataFrame:
    """A small synthetic frame with the same columns as the raw CSV."""
    rows = []
    states = [("Kerala", "South", "State"), ("Bihar", "East", "State"), ("Delhi", "North", "UT")]
    for year in range(2019, 2025):
        for i, (state, region, stype) in enumerate(states):
            for j, value in enumerate(["Rural", "Urban"]):
                accidents = 5 if (i == 2 and j == 1) else 100 * (i + 1) + 10 * j + year - 2019
                low = accidents < 10
                killed = int(accidents * (0.1 + 0.3 * i))
                rows.append({
                    "year": year, "state": state, "state_harmonised": state, "state_type": stype,
                    "region": region, "annexure": 39, "factor_group": "area_type", "factor_value": value,
                    "accidents": accidents, "killed": killed, "grievously_injured": 3, "minor_injured": 4,
                    "total_injured": 7, "severity_index": 100 * killed / accidents, "national_severity": 36.0,
                    "severity_level": None if low else ["Low", "Medium", "High"][i], "low_count_flag": low,
                })
    return pd.DataFrame(rows)


@pytest.fixture
def frame():
    return make_frame()


# ---- leakage ---------------------------------------------------------------

def test_leakage_list_covers_every_forbidden_column():
    assert FORBIDDEN <= set(LEAKAGE_COLS)


def test_no_leakage_column_in_X(frame):
    X, _, _ = prepare(frame)
    assert not (set(X.columns) & FORBIDDEN), "a leakage column is in X"
    assert not (set(X.columns) & set(LEAKAGE_COLS)), "a leakage column is in X"
    assert not [c for c in X.columns if "injured" in c.lower()], "an injured column is in X"
    assert TARGET not in X.columns


def test_X_has_exactly_the_agreed_features(frame):
    X, _, _ = prepare(frame)
    assert list(X.columns) == ["factor_group", "factor_value", "region", "state_type",
                               "state_harmonised", "year", "log_accidents"]
    assert list(X.columns) == FEATURES


def test_X_does_not_depend_on_leakage_values(frame):
    """Scrambling every leakage column must leave X unchanged."""
    X_before, _, _ = prepare(frame)
    rng = np.random.default_rng(42)
    scrambled = frame.copy()
    for col in LEAKAGE_COLS:
        scrambled[col] = rng.permutation(scrambled[col].to_numpy()) * 3 + 1
    X_after, _, _ = prepare(scrambled)
    pd.testing.assert_frame_equal(X_before, X_after)


@needs_data
def test_no_leakage_column_in_X_on_real_data():
    X, _, _ = prepare(load_data(drop_low_count=False))
    assert not (set(X.columns) & FORBIDDEN)
    assert not [c for c in X.columns if "injured" in c.lower()]
    assert list(X.columns) == FEATURES


# ---- prepare ---------------------------------------------------------------

def test_prepare_drops_low_count_rows(frame):
    X, y, groups = prepare(frame)
    kept = frame[~frame["low_count_flag"]]
    assert frame["low_count_flag"].any()
    assert len(X) == len(y) == len(groups) == len(kept)
    assert X.index.equals(kept.index) and y.index.equals(X.index) and groups.index.equals(X.index)
    assert y.notna().all()


def test_log_accidents_is_log1p(frame):
    X, _, _ = prepare(frame)
    expected = np.log1p(frame.loc[X.index, "accidents"])
    np.testing.assert_allclose(X["log_accidents"], expected)
    assert "accidents" not in X.columns


def test_y_and_groups(frame):
    X, y, groups = prepare(frame)
    assert y.name == "severity_level"
    assert set(y.unique()) <= {"Low", "Medium", "High"}
    assert groups.name == "state_harmonised"
    assert (groups == X["state_harmonised"]).all()


# ---- time_split ------------------------------------------------------------

def test_time_split_years(frame):
    train, test = time_split(frame)
    assert set(train["year"]) == {2019, 2020, 2021, 2022} == set(TRAIN_YEARS)
    assert set(test["year"]) == {2023, 2024} == set(TEST_YEARS)
    assert len(train) + len(test) == len(frame)
    assert not set(train.index) & set(test.index)


@needs_data
def test_time_split_sizes_on_real_data():
    train, test = time_split(load_data())
    X_train, y_train, _ = prepare(train)
    X_test, y_test, _ = prepare(test)
    assert (len(X_train), len(X_test)) == (6327, 3005)
    assert X_train["year"].max() == 2022 and X_test["year"].min() == 2023
    assert set(y_train.unique()) == set(y_test.unique()) == {"Low", "Medium", "High"}


# ---- encoder ---------------------------------------------------------------

def test_encoder_ignores_unseen_categories(frame):
    train, test = time_split(frame)
    X_train, _, _ = prepare(train)
    X_test, _, _ = prepare(test)
    X_test = X_test.copy()
    X_test.loc[X_test.index[0], "factor_value"] = "Never seen in training"
    X_test.loc[X_test.index[1], "state_harmonised"] = "New State"

    pre = make_preprocessor().fit(X_train)
    out = pre.transform(X_test)          # must not raise
    assert out.shape[0] == len(X_test)
    assert pre.named_transformers_["cat"].handle_unknown == "ignore"


@needs_data
def test_real_data_has_categories_missing_from_some_years():
    """Documents why handle_unknown='ignore' is needed, and checks the encoder copes."""
    df = load_data()
    train, test = time_split(df)
    junction = df[df["factor_group"] == "junction_type"]
    assert set(junction.loc[junction["factor_value"] == "Others", "year"]) == {2019, 2020, 2021}
    env_2024 = df[(df["factor_group"] == "road_environment") & (df["year"] == 2024)]
    assert env_2024["factor_value"].nunique() == 3

    X_train, _, _ = prepare(train)
    X_test, _, _ = prepare(test)
    out = make_preprocessor().fit(X_train).transform(X_test)
    assert out.shape[0] == len(X_test)
    assert set(CATEGORICAL_FEATURES) <= set(X_train.columns)
