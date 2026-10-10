"""Robustness checks on the Decision Tree.

Three experiments, all with the Decision Tree in its best configuration (the
strategy and hyperparameters chosen during tuning on 2019 to 2022). Nothing is
re-tuned here, so only the split or the feature set changes between rows.

1. Split design. Splits that keep a state on one side only (GroupKFold) against
   random row splits that put rows of the same state in train and test.
2. Leakage. The time split with the agreed features against the same split with
   severity_index or killed added as a feature.
3. Drift. The model trained on 2019 to 2022, tested on 2023 and on 2024 separately.

The leaky and random-split rows are deliberately wrong setups. They exist to
show the size of the error and must never be reported as model performance.

Run from the project root:

    python -m src.robustness
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import SEED, SEVERITY_ORDER, load_data, seed_everything
from src.evaluate import STRATEGY_NAMES, TAB_DIR, best_configurations
from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TEST_YEARS, TRAIN_YEARS, prepare
from src.models import STRATEGIES, model_path

MODEL = "dt"
N_SPLITS = 5
LEAKS = {"severity_index": ["severity_index"], "killed": ["killed"]}

METRICS = ["accuracy", "balanced_accuracy", "macro_f1", "recall_low", "recall_medium", "recall_high"]


def tuned_tree(extra_numeric: list[str] | None = None):
    """The unfitted Decision Tree pipeline with its tuned hyperparameters.

    ``extra_numeric`` adds columns to the encoder. It is used only by the leakage
    experiment, which is the one place a leakage column may enter a pipeline.
    """
    strategy = best_configurations()[MODEL]
    pipe = clone(joblib.load(model_path(MODEL, strategy)))
    if extra_numeric:
        pipe.set_params(encode=ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
                ("num", StandardScaler(), NUMERIC_FEATURES + list(extra_numeric)),
            ],
            remainder="drop",
        ))
    return pipe


def load_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """The modelling rows of every year: the raw frame, X, y and groups, on one index."""
    df = load_data()
    X, y, groups = prepare(df)
    return df, X, y.astype(str), groups


def score(y_true, y_pred) -> dict:
    recalls = recall_score(y_true, y_pred, labels=SEVERITY_ORDER, average=None, zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=SEVERITY_ORDER, average="macro", zero_division=0),
        "recall_low": recalls[0],
        "recall_medium": recalls[1],
        "recall_high": recalls[2],
    }


def fit_and_score(pipe, X, y, train, test) -> dict:
    """Fit on the ``train`` rows and score on the ``test`` rows (boolean masks or positions)."""
    X_train, X_test = (X.loc[train], X.loc[test]) if isinstance(train, pd.Series) else (X.iloc[train], X.iloc[test])
    y_train, y_test = (y.loc[train], y.loc[test]) if isinstance(train, pd.Series) else (y.iloc[train], y.iloc[test])
    pipe.fit(X_train, y_train)
    return {"n_train": len(y_train), "n_test": len(y_test), **score(y_test, pipe.predict(X_test))}


def cross_validate(X, y, splits) -> dict:
    """Mean of each metric over the folds, with the fold-to-fold SD of macro-F1."""
    folds = pd.DataFrame([fit_and_score(tuned_tree(), X, y, train, test) for train, test in splits])
    row = folds.mean().to_dict()
    row["macro_f1_sd"] = folds["macro_f1"].std(ddof=1)
    row["n_train"], row["n_test"] = round(row["n_train"]), round(row["n_test"])
    return row


# ---- experiments ------------------------------------------------------------

def split_experiment(X, y, groups, year) -> list[dict]:
    """Experiment 1: does the same state sit on both sides of the split?"""
    in_train, in_test = year.isin(TRAIN_YEARS), year.isin(TEST_YEARS)
    Xt, yt, gt = X[in_train], y[in_train], groups[in_train]

    grouped = cross_validate(Xt, yt, list(GroupKFold(n_splits=N_SPLITS).split(Xt, yt, gt)))
    random_cv = cross_validate(
        Xt, yt, list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(Xt, yt))
    )
    time = fit_and_score(tuned_tree(), X, y, in_train, in_test)
    # A random split of all six years with the same test size as the time split.
    train_idx, test_idx = train_test_split(
        np.arange(len(X)), test_size=int(in_test.sum()), stratify=y, random_state=SEED
    )
    random_rows = fit_and_score(tuned_tree(), X, y, train_idx, test_idx)

    return [
        {"setup": "GroupKFold(5) by state", "train": "2019-2022", "test": "held-out states",
         "state_in_both": False, "reference": True, **grouped},
        {"setup": "Random stratified 5-fold", "train": "2019-2022", "test": "random rows",
         "state_in_both": True, "reference": False, **random_cv},
        {"setup": "Time split", "train": "2019-2022", "test": "2023-2024",
         "state_in_both": True, "reference": True, **time},
        {"setup": "Random row split, all years", "train": "random rows", "test": "random rows",
         "state_in_both": True, "reference": False, **random_rows},
    ]


def leakage_experiment(df, X, y, year) -> list[dict]:
    """Experiment 2: the time split with a leakage column added to the features."""
    in_train, in_test = year.isin(TRAIN_YEARS), year.isin(TEST_YEARS)
    rows = [{"setup": "Agreed features", "reference": True,
             **fit_and_score(tuned_tree(), X, y, in_train, in_test)}]
    for name, cols in LEAKS.items():
        X_leaky = X.join(df[cols])
        rows.append({"setup": f"Agreed features + {name}", "reference": False,
                     **fit_and_score(tuned_tree(cols), X_leaky, y, in_train, in_test)})
    return [{"train": "2019-2022", "test": "2023-2024", "state_in_both": True, **r} for r in rows]


def drift_experiment(X, y, year) -> list[dict]:
    """Experiment 3: one model trained on 2019 to 2022, scored on each test year."""
    in_train = year.isin(TRAIN_YEARS)
    pipe = tuned_tree().fit(X[in_train], y[in_train])
    rows = []
    for label, years in [("2023-2024", TEST_YEARS), *[(str(yr), (yr,)) for yr in TEST_YEARS]]:
        mask = year.isin(years)
        rows.append({
            "setup": f"Tested on {label}", "train": "2019-2022", "test": label, "state_in_both": True,
            "reference": len(years) > 1, "n_train": int(in_train.sum()), "n_test": int(mask.sum()),
            **score(y[mask], pipe.predict(X[mask])),
        })
    return rows


def run() -> pd.DataFrame:
    df, X, y, groups = load_all()
    year = df["year"]
    parts = {
        "1 Split design": split_experiment(X, y, groups, year),
        "2 Leakage": leakage_experiment(df, X, y, year),
        "3 Drift over time": drift_experiment(X, y, year),
    }
    table = pd.concat(
        [pd.DataFrame(rows).assign(experiment=name) for name, rows in parts.items()], ignore_index=True
    )

    # Change in macro-F1 against the correct setup that comes before the row in its experiment.
    reference = table["macro_f1"].where(table["reference"]).groupby(table["experiment"]).ffill()
    table["macro_f1_change"] = (table["macro_f1"] - reference).where(~table["reference"])

    columns = ["experiment", "setup", "train", "test", "state_in_both", "n_train", "n_test",
               *METRICS, "macro_f1_sd", "macro_f1_change"]
    return table[columns]


def to_latex(table: pd.DataFrame, strategy: str) -> str:
    out = pd.DataFrame({
        "Experiment": table["experiment"].str[2:].where(~table["experiment"].duplicated(), ""),
        "Setup": table["setup"].str.replace("_", r"\_", regex=False),
        "Test rows": table["n_test"].map("{:,}".format).str.replace(",", "{,}", regex=False),
        "Acc.": table["accuracy"].map("{:.3f}".format),
        "Bal. acc.": table["balanced_accuracy"].map("{:.3f}".format),
        "Macro-F1": table["macro_f1"].map("{:.3f}".format),
        "Rec. High": table["recall_high"].map("{:.3f}".format),
        r"$\Delta$ Macro-F1": table["macro_f1_change"].map(lambda v: "" if pd.isna(v) else f"{v:+.3f}"),
    })
    # The first row of each experiment carries its name; put a rule above the later ones.
    lines = out.to_latex(
        index=False, escape=False, column_format="llrrrrrr",
        caption=(f"Robustness checks on the Decision Tree ({strategy}). $\\Delta$ Macro-F1 is the change "
                 "against the correct setup listed before the row in the same experiment. Cross-validated "
                 "rows report the mean over five folds."),
        label="tab:robustness", position="t",
    ).splitlines()
    first_rows = [i for i, line in enumerate(lines) if line[:1].isalpha()][2:]
    for i in reversed(first_rows):
        lines.insert(i, r"\midrule")
    return "\n".join(lines) + "\n"


def main() -> int:
    strategy = best_configurations()[MODEL]
    assert strategy in STRATEGIES
    seed_everything()
    table = run()

    TAB_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TAB_DIR / "robustness.csv", index=False)
    (TAB_DIR / "robustness.tex").write_text(to_latex(table, STRATEGY_NAMES[strategy]), encoding="utf-8")

    pd.set_option("display.width", 250)
    print(f"Decision Tree, strategy: {strategy}, tuned hyperparameters, seed {SEED}\n")
    print(table.drop(columns=["train", "state_in_both", "n_train"]).to_string(
        index=False, float_format="{:.3f}".format, na_rep=""))
    print("\nsaved reports/tables/robustness.csv and .tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
