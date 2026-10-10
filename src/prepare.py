"""Prepare: check the raw CSV, build the features and write the time split.

Writes to data/processed/:

    train.csv            2019 to 2022, the seven features plus severity_level
    test.csv             2023 to 2024, same columns
    split_summary.csv    rows, states and class counts for each split

These files are a record of exactly what the models see. The later phases
rebuild the same frames from the raw CSV, so they do not depend on them.

Run from the project root:

    python -m src.prepare
"""
from __future__ import annotations

import pandas as pd

from src.data import LEAKAGE_COLS, PROJECT_ROOT, SEVERITY_ORDER, TARGET, load_data, seed_everything
from src.features import FEATURES, TEST_YEARS, TRAIN_YEARS, prepare, time_split

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def build(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The train and test frames: features and target, one row per modelling row."""
    train, test = time_split(df)
    frames = {}
    for name, part in [("train", train), ("test", test)]:
        X, y, _ = prepare(part)
        frames[name] = X.assign(**{TARGET: y.astype(str)})
    return frames


def check(frames: dict[str, pd.DataFrame]) -> None:
    """Fail loudly if the split or the feature set breaks a project rule."""
    train, test = frames["train"], frames["test"]
    assert list(train.columns) == FEATURES + [TARGET] == list(test.columns), "unexpected columns"
    assert not set(train.columns) & set(LEAKAGE_COLS), "a leakage column is in the features"
    assert set(train["year"]) == set(TRAIN_YEARS) and set(test["year"]) == set(TEST_YEARS), "years are mixed"
    for frame in frames.values():
        assert frame.notna().all().all(), "missing values in the prepared data"
        assert set(frame[TARGET]) == set(SEVERITY_ORDER), "a severity level is absent"


def summary(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, frame in frames.items():
        counts = frame[TARGET].value_counts()
        rows.append({
            "split": name, "years": f"{frame['year'].min()}-{frame['year'].max()}", "rows": len(frame),
            "states": frame["state_harmonised"].nunique(),
            **{level: int(counts[level]) for level in SEVERITY_ORDER},
            **{f"{level}_share": round(counts[level] / len(frame), 3) for level in SEVERITY_ORDER},
        })
    return pd.DataFrame(rows)


def main() -> int:
    seed_everything()
    raw = load_data(drop_low_count=False)
    df = load_data()
    frames = build(df)
    check(frames)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)
    table = summary(frames)
    table.to_csv(PROCESSED_DIR / "split_summary.csv", index=False)

    print(f"Raw rows: {len(raw):,}. Dropped as low count: {len(raw) - len(df):,}. Modelling rows: {len(df):,}.")
    print(table.to_string(index=False))
    print("saved data/processed/train.csv, test.csv and split_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
