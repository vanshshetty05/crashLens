"""Model pipelines, tuning and persistence for CrashLens.

Models: Decision Tree, AdaBoost (depth-2 trees), KNN, Categorical Naive Bayes and
a Random Forest that serves as a reference only.
Imbalance strategies: none, class weights, SMOTEN.

Every pipeline is an imbalanced-learn Pipeline in this order:

    resampling (SMOTEN only) -> encoding -> classifier

Tuning uses the 2019 to 2022 data only, GridSearchCV with GroupKFold(5) on
state_harmonised and macro-F1 as the selection metric. Seed 42 everywhere.

Run from the project root:

    python -m src.models --model dt
    python -m src.models --model ada --strategy class_weight
    python -m src.models                      # every model and strategy
"""
from __future__ import annotations

import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTEN
from imblearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.naive_bayes import CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import FunctionTransformer, KBinsDiscretizer, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

from src.data import PROJECT_ROOT, SEED, load_data, seed_everything
from src.features import CATEGORICAL_FEATURES, make_preprocessor, prepare, time_split

MODELS_DIR = PROJECT_ROOT / "models"
CV_RESULTS_PATH = PROJECT_ROOT / "reports" / "tables" / "cv_results.csv"

MODELS = ["dt", "ada", "knn", "nb", "rf"]
STRATEGIES = ["none", "class_weight", "smoten"]
MODEL_NAMES = {
    "dt": "Decision Tree",
    "ada": "AdaBoost",
    "knn": "KNN",
    "nb": "Categorical Naive Bayes",
    "rf": "Random Forest (reference)",
}
# KNN has no class-weight mechanism, so that combination is not run.
SKIPPED = {("knn", "class_weight")}

N_SPLITS = 5
REFIT_METRIC = "f1_macro"
SCORING = {
    "f1_macro": "f1_macro",
    "balanced_accuracy": "balanced_accuracy",
    "recall_high": make_scorer(recall_score, labels=["High"], average="macro", zero_division=0),
}

PARAM_GRIDS = {
    "dt": {"clf__max_depth": [4, 6, 10, None], "clf__min_samples_leaf": [1, 5, 20]},
    "ada": {"clf__n_estimators": [100, 300], "clf__learning_rate": [0.1, 0.5, 1.0]},
    "knn": {"clf__n_neighbors": [5, 11, 21], "clf__weights": ["uniform", "distance"]},
    "nb": {"clf__alpha": [0.1, 0.5, 1.0]},
    "rf": {"clf__max_depth": [10, None], "clf__min_samples_leaf": [1, 5]},
}


# ---- encoders ---------------------------------------------------------------

def shift_codes(X):
    """Add 1 to every ordinal code.

    The ordinal encoder marks a category it did not see in training as -1, and
    CategoricalNB cannot index a negative code. After the shift, 0 means "unseen"
    and known categories start at 1. Code 0 has no training rows, so an unseen
    category contributes only the smoothing term to each class.
    """
    return np.asarray(X) + 1


def make_nb_encoder() -> ColumnTransformer:
    """Ordinal codes for CategoricalNB: categories and year as codes, log_accidents in quartile bins.

    Must be followed by ``make_code_shift()`` so that unseen categories get a valid code.
    """
    return ColumnTransformer(
        transformers=[
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
             CATEGORICAL_FEATURES + ["year"]),
            ("bin", KBinsDiscretizer(n_bins=4, encode="ordinal", strategy="quantile", random_state=SEED), ["log_accidents"]),
        ],
        remainder="drop",
    )


def make_code_shift() -> FunctionTransformer:
    return FunctionTransformer(shift_codes)


# ---- pipelines --------------------------------------------------------------

def is_valid(model: str, strategy: str) -> bool:
    return model in MODELS and strategy in STRATEGIES and (model, strategy) not in SKIPPED


def combinations() -> list[tuple[str, str]]:
    """Every model and strategy pair that is run."""
    return [(m, s) for m in MODELS for s in STRATEGIES if is_valid(m, s)]


def make_classifier(model: str, strategy: str):
    """The unfitted classifier, with class weighting applied where the strategy asks for it."""
    balanced = "balanced" if strategy == "class_weight" else None
    if model == "dt":
        return DecisionTreeClassifier(class_weight=balanced, random_state=SEED)
    if model == "ada":
        # AdaBoost takes no class_weight. Balanced sample weights go to fit(); see fit_params().
        return AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=2, random_state=SEED), random_state=SEED
        )
    if model == "knn":
        return KNeighborsClassifier()
    if model == "nb":
        # A uniform class prior is the Naive Bayes counterpart of class weighting.
        return CategoricalNB(fit_prior=(strategy != "class_weight"))
    if model == "rf":
        return RandomForestClassifier(n_estimators=300, class_weight=balanced, random_state=SEED, n_jobs=1)
    raise ValueError(f"Unknown model: {model}")


def build_pipeline(model: str, strategy: str) -> Pipeline:
    """Build the pipeline: resampling (SMOTEN only), then encoding, then the classifier."""
    if not is_valid(model, strategy):
        raise ValueError(f"Combination not supported: model={model!r}, strategy={strategy!r}")

    steps = []
    if strategy == "smoten":
        steps.append(("resample", SMOTEN(random_state=SEED)))
    if model == "nb":
        # imbalanced-learn does not allow a nested pipeline, so the encoding takes two steps.
        steps += [("encode", make_nb_encoder()), ("shift", make_code_shift())]
    else:
        steps.append(("encode", make_preprocessor()))
    steps.append(("clf", make_classifier(model, strategy)))
    return Pipeline(steps)


def fit_params(model: str, strategy: str, y) -> dict:
    """Extra arguments for fit(). Only AdaBoost with class weights needs any."""
    if model == "ada" and strategy == "class_weight":
        return {"clf__sample_weight": compute_sample_weight("balanced", y)}
    return {}


# ---- tuning -----------------------------------------------------------------

def load_training_data() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """X, y and groups for 2019 to 2022. The test years are never loaded here."""
    train, _ = time_split(load_data())
    X, y, groups = prepare(train)
    return X, y.astype(str), groups


def tune(model: str, strategy: str, X, y, groups, n_jobs: int = -1) -> GridSearchCV:
    """Grid search with GroupKFold(5) by state, selecting on macro-F1."""
    search = GridSearchCV(
        estimator=build_pipeline(model, strategy),
        param_grid=PARAM_GRIDS[model],
        scoring=SCORING,
        refit=REFIT_METRIC,
        cv=GroupKFold(n_splits=N_SPLITS),
        n_jobs=n_jobs,
        error_score="raise",
    )
    search.fit(X, y, groups=groups, **fit_params(model, strategy, y))
    return search


def cv_table(search: GridSearchCV, model: str, strategy: str) -> pd.DataFrame:
    """One row per grid candidate, with its parameters and cross-validation scores."""
    res = search.cv_results_
    rows = []
    for i, params in enumerate(res["params"]):
        row = {
            "model": model,
            "strategy": strategy,
            "params": json.dumps({k.replace("clf__", ""): v for k, v in params.items()}, sort_keys=True),
            "is_best": i == search.best_index_,
            "rank_f1_macro": int(res["rank_test_f1_macro"][i]),
        }
        for metric in SCORING:
            row[f"mean_{metric}"] = res[f"mean_test_{metric}"][i]
            row[f"std_{metric}"] = res[f"std_test_{metric}"][i]
        for k in range(N_SPLITS):
            row[f"fold{k + 1}_f1_macro"] = res[f"split{k}_test_f1_macro"][i]
        row["mean_fit_time_s"] = res["mean_fit_time"][i]
        rows.append(row)
    return pd.DataFrame(rows)


def log_cv_results(table: pd.DataFrame) -> None:
    """Write the rows to cv_results.csv, replacing earlier rows for the same model and strategy."""
    CV_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if CV_RESULTS_PATH.exists():
        old = pd.read_csv(CV_RESULTS_PATH)
        done = set(zip(table["model"], table["strategy"]))
        keep = [pair not in done for pair in zip(old["model"], old["strategy"])]
        table = pd.concat([old[keep], table], ignore_index=True)
    table = table.assign(
        _m=table["model"].map(MODELS.index), _s=table["strategy"].map(STRATEGIES.index)
    ).sort_values(["_m", "_s", "rank_f1_macro"], kind="stable").drop(columns=["_m", "_s"])
    table.to_csv(CV_RESULTS_PATH, index=False)


def model_path(model: str, strategy: str):
    return MODELS_DIR / f"{model}_{strategy}.joblib"


def save_pipeline(pipeline: Pipeline, model: str, strategy: str):
    """Save a fitted pipeline to models/ and return its path.

    The resampler only acts during fit and is skipped at prediction time. Its fitted
    state holds a copy of the training data, so it is replaced with an unfitted copy
    to keep the file small. Predictions are unchanged.
    """
    if "resample" in pipeline.named_steps:
        pipeline.steps[0] = ("resample", clone(pipeline.named_steps["resample"]))
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = model_path(model, strategy)
    joblib.dump(pipeline, path, compress=3)
    return path


def run(model: str, strategy: str, X, y, groups, n_jobs: int = -1) -> GridSearchCV:
    """Tune one model and strategy, save the best pipeline and log the CV scores."""
    start = time.time()
    search = tune(model, strategy, X, y, groups, n_jobs=n_jobs)

    save_pipeline(search.best_estimator_, model, strategy)
    log_cv_results(cv_table(search, model, strategy))

    i = search.best_index_
    best = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
    print(
        f"{MODEL_NAMES[model]:<26} {strategy:<13} "
        f"macro-F1 {search.best_score_:.3f} (sd {search.cv_results_['std_test_f1_macro'][i]:.3f})  "
        f"bal. acc {search.cv_results_['mean_test_balanced_accuracy'][i]:.3f}  "
        f"High recall {search.cv_results_['mean_test_recall_high'][i]:.3f}  "
        f"{best}  [{time.time() - start:.0f}s]"
    )
    return search


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Tune CrashLens models on the 2019 to 2022 data.")
    parser.add_argument("--model", choices=MODELS + ["all"], default="all")
    parser.add_argument("--strategy", choices=STRATEGIES + ["all"], default="all")
    parser.add_argument("--n-jobs", type=int, default=-1, help="parallel workers for the grid search")
    args = parser.parse_args(argv)
    seed_everything()

    models = MODELS if args.model == "all" else [args.model]
    strategies = STRATEGIES if args.strategy == "all" else [args.strategy]

    X, y, groups = load_training_data()
    print(f"Training rows: {len(X):,} (2019 to 2022), {groups.nunique()} states, GroupKFold({N_SPLITS})")

    for model in models:
        for strategy in strategies:
            if not is_valid(model, strategy):
                print(f"{MODEL_NAMES[model]:<26} {strategy:<13} skipped (not applicable)")
                continue
            run(model, strategy, X, y, groups, n_jobs=args.n_jobs)

    print(f"Best pipelines saved to {MODELS_DIR}")
    print(f"CV scores logged to {CV_RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    # Import through the package so saved pipelines reference src.models, not __main__.
    from src.models import main as _main

    raise SystemExit(_main())
