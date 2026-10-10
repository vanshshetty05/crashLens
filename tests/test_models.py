"""Tests for src/models.py: pipeline structure, strategy handling and unseen categories."""
import numpy as np
import pandas as pd
import pytest
from imblearn.over_sampling import SMOTEN
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from src.features import FEATURES
from src.models import (
    MODELS,
    PARAM_GRIDS,
    STRATEGIES,
    build_pipeline,
    combinations,
    fit_params,
    is_valid,
)


def make_xy(n=240, seed=42):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "factor_group": rng.choice(["weather", "area_type"], n),
        "factor_value": rng.choice(["Rainy", "Sunny/Clear", "Rural", "Urban"], n),
        "region": rng.choice(["South", "East", "North"], n),
        "state_type": rng.choice(["State", "UT"], n),
        "state_harmonised": rng.choice(["Kerala", "Bihar", "Delhi", "Goa", "Punjab"], n),
        "year": rng.choice([2019, 2020, 2021, 2022], n),
        "log_accidents": rng.uniform(2.4, 11, n),
    })[FEATURES]
    y = pd.Series(rng.choice(["Low", "Medium", "High"], n, p=[0.25, 0.3, 0.45]), name="severity_level")
    return X, y


def test_combinations_cover_everything_except_knn_class_weight():
    combos = combinations()
    assert ("knn", "class_weight") not in combos
    assert len(combos) == len(MODELS) * len(STRATEGIES) - 1
    assert not is_valid("knn", "class_weight")
    with pytest.raises(ValueError):
        build_pipeline("knn", "class_weight")


@pytest.mark.parametrize("model,strategy", combinations())
def test_step_order_is_resample_then_encode_then_classifier(model, strategy):
    names = [name for name, _ in build_pipeline(model, strategy).steps]
    assert names[-1] == "clf"
    assert names.index("encode") < names.index("clf")
    if strategy == "smoten":
        assert names[0] == "resample"
        assert isinstance(build_pipeline(model, strategy).named_steps["resample"], SMOTEN)
    else:
        assert "resample" not in names


@pytest.mark.parametrize("model", MODELS)
def test_encoding_matches_the_model(model):
    encode = build_pipeline(model, "none").named_steps["encode"]
    cat = dict((name, t) for name, t, _ in encode.transformers)["cat"]
    if model == "nb":
        assert isinstance(cat, OrdinalEncoder)
        assert dict((name, t) for name, t, _ in encode.transformers)["bin"].n_bins == 4
    else:
        assert isinstance(cat, OneHotEncoder) and cat.handle_unknown == "ignore"


def test_class_weight_handling():
    assert build_pipeline("dt", "class_weight").named_steps["clf"].class_weight == "balanced"
    assert build_pipeline("rf", "class_weight").named_steps["clf"].class_weight == "balanced"
    assert build_pipeline("dt", "none").named_steps["clf"].class_weight is None
    assert build_pipeline("nb", "class_weight").named_steps["clf"].fit_prior is False
    assert build_pipeline("ada", "none").named_steps["clf"].estimator.max_depth == 2


def test_adaboost_class_weight_uses_balanced_sample_weight():
    _, y = make_xy()
    params = fit_params("ada", "class_weight", y)
    weights = params["clf__sample_weight"]
    totals = pd.Series(weights).groupby(y.to_numpy()).sum()
    np.testing.assert_allclose(totals, totals.iloc[0])      # every class carries the same total weight
    assert fit_params("ada", "none", y) == {}
    assert fit_params("dt", "class_weight", y) == {}


def test_param_grids_are_the_agreed_ones():
    assert PARAM_GRIDS["dt"] == {"clf__max_depth": [4, 6, 10, None], "clf__min_samples_leaf": [1, 5, 20]}
    assert PARAM_GRIDS["ada"] == {"clf__n_estimators": [100, 300], "clf__learning_rate": [0.1, 0.5, 1.0]}
    assert PARAM_GRIDS["knn"] == {"clf__n_neighbors": [5, 11, 21], "clf__weights": ["uniform", "distance"]}
    assert PARAM_GRIDS["nb"] == {"clf__alpha": [0.1, 0.5, 1.0]}
    assert PARAM_GRIDS["rf"] == {"clf__max_depth": [10, None], "clf__min_samples_leaf": [1, 5]}


@pytest.mark.parametrize("model,strategy", combinations())
def test_fit_and_predict_with_unseen_categories(model, strategy):
    """Every pipeline must predict on a year, state and category it never saw in training."""
    X, y = make_xy()
    pipe = build_pipeline(model, strategy)
    if model in ("ada", "rf"):
        pipe.set_params(clf__n_estimators=10)
    pipe.fit(X, y, **fit_params(model, strategy, y))

    X_new = X.head(20).copy()
    X_new["year"] = 2024
    X_new.loc[X_new.index[0], "factor_value"] = "Never seen"
    X_new.loc[X_new.index[1], "state_harmonised"] = "New State"
    X_new.loc[X_new.index[2], "log_accidents"] = 15.0
    pred = pipe.predict(X_new)
    assert len(pred) == 20
    assert set(pred) <= {"Low", "Medium", "High"}
