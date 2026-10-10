"""SHAP-based explanations and figures.

Four parts:

1. Permutation importance of the best interpretable model on the 2023 to 2024
   test set (macro-F1, 10 repeats). The seven raw features are permuted, not
   the encoded columns, so each feature gets one score.
2. A shallow Decision Tree (max_depth 4, min_samples_leaf 30, balanced class
   weights) trained on 2019 to 2022. Its rules are exported in full, and the
   five most useful ones are rewritten in plain English.
3. A SHAP summary plot for the best ensemble.
4. ``explain_one(row)``: the top three factors behind one prediction, for the API.

"Best" always means the highest tuning macro-F1 in cv_results.csv, which holds
2019 to 2022 scores only. The test years never pick a model.

Run from the project root:

    python -m src.explain
"""
from __future__ import annotations

import sys
from functools import lru_cache

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text

from src.data import SEED, SEVERITY_ORDER, seed_everything
from src.evaluate import FIG_DIR, INK, STRATEGY_NAMES, TAB_DIR, load_test_data, save_fig, set_style
from src.features import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES
from src.models import CV_RESULTS_PATH, MODEL_NAMES, load_training_data, model_path

# KNN is left out of the interpretable group: it has no global structure to read.
INTERPRETABLE = ["dt", "nb"]
ENSEMBLES = ["ada", "rf"]

N_REPEATS = 10
N_RULES = 5
N_FACTORS = 3
SHAP_SAMPLE = 500          # test rows used for the SHAP summary plot
SHAP_MAX_DISPLAY = 15

BLUE = "#2a78d6"
FEATURE_LABELS = {
    "factor_group": "Factor group", "factor_value": "Factor value", "region": "Region",
    "state_type": "State or UT", "state_harmonised": "State", "year": "Year",
    "log_accidents": "Number of accidents",
}


def best_of(models: list[str]) -> tuple[str, str]:
    """The (model, strategy) pair with the highest tuning macro-F1 among ``models``."""
    cv = pd.read_csv(CV_RESULTS_PATH)
    best = cv[cv["is_best"] & cv["model"].isin(models)]
    row = best.loc[best["mean_f1_macro"].idxmax()]
    return row["model"], row["strategy"]


def describe(model: str, strategy: str) -> str:
    return f"{MODEL_NAMES[model]}, {STRATEGY_NAMES[strategy].lower()}"


# ---- 1. permutation importance ----------------------------------------------

def permutation_table(model: str, strategy: str) -> pd.DataFrame:
    """Drop in test macro-F1 when one raw feature is shuffled, over 10 repeats."""
    pipe = joblib.load(model_path(model, strategy))
    X_test, y_test = load_test_data()
    result = permutation_importance(
        pipe, X_test, y_test, scoring="f1_macro", n_repeats=N_REPEATS, random_state=SEED, n_jobs=1
    )
    return pd.DataFrame({
        "feature": FEATURES,
        "importance_mean": result.importances_mean,
        "importance_sd": result.importances_std,
    }).sort_values("importance_mean", ascending=False, ignore_index=True)


def plot_permutation(table: pd.DataFrame, model: str, strategy: str) -> None:
    t = table.iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.barh([FEATURE_LABELS[f] for f in t["feature"]], t["importance_mean"], xerr=t["importance_sd"],
            height=0.6, color=BLUE, zorder=2, error_kw=dict(ecolor=INK, elinewidth=1, capsize=2.5))
    span = (t["importance_mean"] + t["importance_sd"]).max()
    for i, (m, s) in enumerate(zip(t["importance_mean"], t["importance_sd"])):
        ax.text(max(m, 0) + s + span * 0.02, i, f"{m:.3f}", va="center", fontsize=8, color=INK)
    ax.axvline(0, color=INK, linewidth=0.8, zorder=3)
    ax.set_xlim(min(0, (t["importance_mean"] - t["importance_sd"]).min()) - span * 0.02, span * 1.14)
    ax.set_xlabel(f"Drop in macro-F1 when the feature is shuffled (mean of {N_REPEATS} repeats, bar = 1 SD)")
    ax.set_title(f"Permutation importance on the 2023 to 2024 test set\n{describe(model, strategy)}")
    ax.grid(axis="x")
    ax.tick_params(length=0)
    ax.spines["bottom"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, "fig09_permutation_importance")


# ---- 2. shallow tree and its rules ------------------------------------------

def fit_shallow_tree(X, y) -> SkPipeline:
    """One-hot categories, numeric features left in their own units so thresholds stay readable."""
    encode = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", "passthrough", NUMERIC_FEATURES),
        ],
        remainder="drop", verbose_feature_names_out=False,
    )
    clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=30, class_weight="balanced", random_state=SEED)
    return SkPipeline([("encode", encode), ("clf", clf)]).fit(X, y)


def split_column(column: str) -> tuple[str, str | None]:
    """Map an encoded column back to (raw feature, category). The category is None for numeric features."""
    if column in NUMERIC_FEATURES:
        return column, None
    for feature in CATEGORICAL_FEATURES:
        if column.startswith(feature + "_"):
            return feature, column[len(feature) + 1:]
    raise ValueError(f"Unknown encoded column: {column}")


def rule_paths(clf, columns) -> list[tuple[list[int], list[tuple[str, bool, float]]]]:
    """The tree as a list of rules: (leaves covered, conditions as (column, went_right, threshold)).

    A split whose two sides predict the same class changes no prediction, so the
    walk stops above it and the rule covers every leaf below. The rules still
    reproduce the tree's predictions exactly.
    """
    tree = clf.tree_

    def leaves_under(node):
        if tree.children_left[node] == -1:
            return [node]
        return leaves_under(tree.children_left[node]) + leaves_under(tree.children_right[node])

    rules = []

    def walk(node, path):
        leaves = leaves_under(node)
        if len({tree.value[leaf][0].argmax() for leaf in leaves}) == 1:
            rules.append((leaves, path))
            return
        column, threshold = columns[tree.feature[node]], tree.threshold[node]
        walk(tree.children_left[node], path + [(column, False, threshold)])
        walk(tree.children_right[node], path + [(column, True, threshold)])

    walk(0, [])
    return rules


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " or " + items[-1]


def plain_english(path: list[tuple[str, bool, float]], prediction: str) -> str:
    """Rewrite one root-to-leaf path as a sentence.

    A one-hot column above 0.5 means "is this category" and below means "is not".
    Once a feature is pinned to a category, its "is not" conditions say nothing
    more and are dropped. A named state also fixes its region and whether it is
    a State or UT, so those conditions are dropped too. Bounds on a numeric
    feature are merged into one range.
    """
    equal, excluded, low, high = {}, {}, {}, {}
    for column, went_right, threshold in path:
        feature, category = split_column(column)
        if category is not None:
            if went_right:
                equal[feature] = category
            else:
                excluded.setdefault(feature, []).append(category)
        elif went_right:
            low[feature] = max(threshold, low.get(feature, -np.inf))
        else:
            high[feature] = min(threshold, high.get(feature, np.inf))

    is_phrase = {
        "factor_group": "the factor group is {}", "factor_value": "the category is '{}'",
        "region": "in the {} region", "state_type": "in a {}", "state_harmonised": "in {}",
    }
    not_phrase = {
        "factor_group": "the factor group is not {}", "factor_value": "the category is not {}",
        "region": "outside the {} region", "state_type": "not in a {}", "state_harmonised": "outside {}",
    }
    parts = []
    for feature in CATEGORICAL_FEATURES:
        if feature in ("region", "state_type") and "state_harmonised" in equal:
            continue
        if feature in equal:
            value = equal[feature].replace("_", " ") if feature == "factor_group" else equal[feature]
            parts.append(is_phrase[feature].format(value))
        elif feature in excluded:
            values = excluded[feature]
            if feature == "factor_value":
                values = [f"'{v}'" for v in values]
            parts.append(not_phrase[feature].format(_join(values)))

    if "year" in low or "year" in high:
        first = int(np.floor(low["year"])) + 1 if "year" in low else None
        last = int(np.floor(high["year"])) if "year" in high else None
        parts.append(f"from {first} to {last}" if first and last else f"from {first} onwards" if first
                     else f"up to {last}")
    if "log_accidents" in low or "log_accidents" in high:
        # log_accidents is log1p(accidents), and accidents is a whole number.
        more = int(np.floor(np.expm1(low["log_accidents"]))) if "log_accidents" in low else None
        most = int(np.floor(np.expm1(high["log_accidents"]))) if "log_accidents" in high else None
        parts.append(f"between {more + 1:,} and {most:,} accidents" if more is not None and most is not None
                     else f"more than {more:,} accidents" if more is not None else f"at most {most:,} accidents")

    return (", ".join(parts) if parts else "any row") + f" → {prediction} severity"


def rule_table(tree: SkPipeline, X_train, y_train, X_test, y_test) -> pd.DataFrame:
    """One row per rule, ranked by usefulness.

    Usefulness is weighted relative accuracy: coverage x (precision - base rate
    of the predicted class), on the training rows. A rule ranks high when it
    covers many rows and is much more often right than the class frequency alone.
    Test-set precision is reported next to it as a check, not used for ranking.
    """
    encode, clf = tree.named_steps["encode"], tree.named_steps["clf"]
    paths = rule_paths(clf, encode.get_feature_names_out())
    leaf_train, leaf_test = clf.apply(encode.transform(X_train)), clf.apply(encode.transform(X_test))
    y_train, y_test = np.asarray(y_train), np.asarray(y_test)

    rows = []
    for leaves, path in paths:
        prediction = clf.classes_[clf.tree_.value[leaves[0]][0].argmax()]
        in_train, in_test = np.isin(leaf_train, leaves), np.isin(leaf_test, leaves)
        precision = (y_train[in_train] == prediction).mean()
        base_rate = (y_train == prediction).mean()
        rows.append({
            "prediction": prediction, "n_leaves": len(leaves), "n_conditions": len(path),
            "train_rows": int(in_train.sum()), "train_precision": precision, "base_rate": base_rate,
            "usefulness": in_train.mean() * (precision - base_rate),
            "test_rows": int(in_test.sum()),
            "test_precision": (y_test[in_test] == prediction).mean() if in_test.any() else np.nan,
            "rule": plain_english(path, prediction),
        })
    table = pd.DataFrame(rows).sort_values("usefulness", ascending=False, ignore_index=True)
    table.insert(0, "rank", table.index + 1)
    return table


# ---- 3. SHAP summary for the best ensemble ----------------------------------

@lru_cache(maxsize=1)
def _ensemble():
    """The best ensemble pipeline, its encoded column names and a SHAP explainer. Loaded once."""
    model, strategy = best_of(ENSEMBLES)
    pipe = joblib.load(model_path(model, strategy))
    columns = list(pipe.named_steps["encode"].get_feature_names_out())
    return model, strategy, pipe, columns, shap.TreeExplainer(pipe.named_steps["clf"])


def load_model() -> str:
    """Load the pipeline and explainer behind ``explain_one`` now, and return the model's name.

    The API calls this at startup so the first request does not pay for the load.
    """
    model, strategy, *_ = _ensemble()
    return describe(model, strategy)


def _encode(pipe, X) -> np.ndarray:
    encoded = pipe.named_steps["encode"].transform(X)
    return encoded.toarray() if hasattr(encoded, "toarray") else np.asarray(encoded)


def _shap_values(explainer, encoded) -> np.ndarray:
    """SHAP values as (rows, encoded columns, classes), in predicted-probability units."""
    values = explainer.shap_values(encoded, check_additivity=False)
    return np.stack(values, axis=-1) if isinstance(values, list) else np.asarray(values)


def _raw_feature(column: str) -> str:
    return split_column(column.split("__", 1)[1])[0]


def _display_name(column: str) -> str:
    feature, category = split_column(column.split("__", 1)[1])
    label = FEATURE_LABELS[feature]
    if category is None:
        return label + (" (log scale)" if feature == "log_accidents" else "")
    return f"{label}: {category.replace('_', ' ') if feature == 'factor_group' else category}"


def _shap_chunk(clf, chunk) -> np.ndarray:
    return _shap_values(shap.TreeExplainer(clf), chunk)


def _shap_values_parallel(clf, encoded, n_jobs: int = -1) -> np.ndarray:
    """Exact SHAP values, one chunk of rows per worker.

    The forest's trees are deep (one-hot columns give paths of 60 or more splits),
    so exact TreeSHAP takes over a second per row on one core.
    """
    chunks = np.array_split(encoded, max(1, min(len(encoded) // 10, 4 * joblib.cpu_count())))
    return np.concatenate(joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(_shap_chunk)(clf, c) for c in chunks))


def shap_summary(target: str = "High") -> pd.DataFrame:
    """Beeswarm of SHAP values for one class on a sample of the test set.

    Returns mean |SHAP| per raw feature and class, with one-hot columns summed
    back to the feature they encode.
    """
    model, strategy, pipe, columns, _ = _ensemble()
    X_test, _ = load_test_data()
    sample = X_test.sample(n=min(SHAP_SAMPLE, len(X_test)), random_state=SEED)
    encoded = _encode(pipe, sample)
    values = _shap_values_parallel(pipe.named_steps["clf"], encoded)
    classes = list(pipe.classes_)

    shap.summary_plot(
        values[:, :, classes.index(target)], encoded, feature_names=[_display_name(c) for c in columns],
        max_display=SHAP_MAX_DISPLAY, show=False, plot_size=(8.4, 6.2), rng=np.random.default_rng(SEED),
    )
    fig = plt.gcf()
    fig.axes[0].set_xlabel(f"SHAP value (change in the predicted probability of {target} severity)")
    fig.axes[0].set_title(
        f"What drives a {target} prediction: {describe(model, strategy)}\n"
        f"{len(sample)} rows sampled from the 2023 to 2024 test set. For a category, red = the row is in it.",
        loc="left", fontsize=10, fontweight="bold",
    )
    save_fig(fig, "fig10_shap_summary")

    raw = [_raw_feature(c) for c in columns]
    by_feature = pd.DataFrame(index=FEATURES, columns=classes, dtype=float)
    for k, cls in enumerate(classes):
        # Sum within a row first (SHAP values add up), then average the absolute value over rows.
        summed = pd.DataFrame(values[:, :, k], columns=raw).T.groupby(level=0).sum().T
        by_feature[cls] = summed.abs().mean()
    by_feature = by_feature[SEVERITY_ORDER]
    by_feature["mean_all_classes"] = by_feature.mean(axis=1)
    return by_feature.sort_values("mean_all_classes", ascending=False).rename_axis("feature").reset_index()


# ---- 4. one prediction, for the API -----------------------------------------

def _to_features(row) -> pd.DataFrame:
    """A one-row feature frame from a dict or Series. ``accidents`` is converted to ``log_accidents``."""
    row = dict(row)
    if "log_accidents" not in row:
        if "accidents" not in row:
            raise ValueError("row needs either 'accidents' or 'log_accidents'.")
        row["log_accidents"] = float(np.log1p(float(row["accidents"])))
    missing = [f for f in FEATURES if f not in row]
    if missing:
        raise ValueError(f"row is missing features: {missing}")
    return pd.DataFrame([{f: row[f] for f in FEATURES}])


def explain_one(row) -> dict:
    """Predict one row with the best ensemble and return the top three factors behind it.

    Parameters
    ----------
    row : dict or Series with factor_group, factor_value, region, state_type,
        state_harmonised, year and either accidents or log_accidents.

    Returns
    -------
    dict with
      ``prediction``     the predicted severity level;
      ``probabilities``  the predicted probability of each level;
      ``factors``        three dicts, largest effect first, each with ``feature``,
                         ``label``, ``value``, ``contribution`` (SHAP value: the
                         change in the probability of the predicted level, summed
                         over the feature's one-hot columns) and ``effect``
                         ("supports" or "opposes" the prediction).

    The values are plain Python types, so the result can go straight to ``jsonify``.
    """
    _, _, pipe, columns, explainer = _ensemble()
    X = _to_features(row)
    encoded = _encode(pipe, X)
    proba = pipe.named_steps["clf"].predict_proba(encoded)[0]
    k = int(proba.argmax())

    contribution = pd.Series(_shap_values(explainer, encoded)[0, :, k]).groupby(
        [_raw_feature(c) for c in columns]).sum()
    top = contribution.reindex(contribution.abs().sort_values(ascending=False).index).head(N_FACTORS)

    def shown(feature):
        if feature == "log_accidents":
            return int(round(float(np.expm1(X.at[0, feature]))))
        value = X.at[0, feature]
        return int(value) if feature == "year" else str(value)

    return {
        "prediction": str(pipe.classes_[k]),
        "probabilities": {level: float(proba[list(pipe.classes_).index(level)]) for level in SEVERITY_ORDER},
        "factors": [
            {"feature": "accidents" if f == "log_accidents" else f, "label": FEATURE_LABELS[f],
             "value": shown(f), "contribution": float(c), "effect": "supports" if c > 0 else "opposes"}
            for f, c in top.items()
        ],
    }


# ---- entry point ------------------------------------------------------------

def main() -> int:
    seed_everything()
    set_style()
    sys.stdout.reconfigure(errors="replace")   # the rule arrow is not in every Windows console code page
    pd.set_option("display.width", 250, "display.max_colwidth", 200)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    model, strategy = best_of(INTERPRETABLE)
    importance = permutation_table(model, strategy)
    importance.to_csv(TAB_DIR / "permutation_importance.csv", index=False)
    print(f"Permutation importance, {describe(model, strategy)}")
    print(importance.to_string(index=False, float_format="{:.4f}".format))
    plot_permutation(importance, model, strategy)

    X_train, y_train, _ = load_training_data()
    X_test, y_test = load_test_data()
    tree = fit_shallow_tree(X_train, y_train)
    encode, clf = tree.named_steps["encode"], tree.named_steps["clf"]
    (TAB_DIR / "shallow_tree_rules.txt").write_text(
        export_text(clf, feature_names=list(encode.get_feature_names_out()), show_weights=False),
        encoding="utf-8",
    )
    rules = rule_table(tree, X_train, y_train, X_test, y_test)
    rules.to_csv(TAB_DIR / "shallow_tree_rules.csv", index=False)
    print(f"\nShallow tree: {clf.get_n_leaves()} leaves, {len(rules)} distinct rules, test macro-F1 "
          f"{f1_score(y_test, tree.predict(X_test), average='macro'):.3f}")
    print(rules.head(N_RULES).drop(columns=["n_leaves", "n_conditions", "base_rate"]).to_string(
        index=False, float_format="{:.3f}".format))
    print("saved reports/tables/shallow_tree_rules.txt and .csv")

    by_feature = shap_summary()
    by_feature.to_csv(TAB_DIR / "shap_mean_abs_by_feature.csv", index=False)
    print(f"\nMean |SHAP| by feature, {describe(*best_of(ENSEMBLES))}")
    print(by_feature.to_string(index=False, float_format="{:.4f}".format))

    example = {**X_test.iloc[0].to_dict()}
    print("\nexplain_one on the first test row:", example)
    print(explain_one(example))
    return 0


if __name__ == "__main__":
    # Import through the package so the explainer cache is shared with other importers.
    from src.explain import main as _main

    raise SystemExit(_main())
