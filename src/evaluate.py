"""Evaluation: macro-F1, balanced accuracy and per-class recall, with accuracy as a secondary metric.

Two parts:

1. Test-set evaluation. Every pipeline saved in models/ is scored once on the
   2023 to 2024 data, next to a baseline that always predicts the most common
   training class. Nothing is tuned or selected on these rows.
2. Statistical comparison. Each model's best configuration (the strategy with the
   highest tuning macro-F1 in cv_results.csv, with its tuned hyperparameters) is
   re-run on the 2019 to 2022 data with repeated group k-fold (5 folds x 2 repeats,
   grouped by state_harmonised). The per-fold macro-F1 scores go to a Friedman
   test and a Nemenyi post-hoc test.

Run from the project root:

    python -m src.evaluate
    python -m src.evaluate --skip-cv          # test-set tables and figures only
"""
from __future__ import annotations

import argparse

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
from joblib import Parallel, delayed
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import friedmanchisquare, studentized_range
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import label_binarize

from src.data import PROJECT_ROOT, SEED, SEVERITY_ORDER, load_data, seed_everything
from src.features import prepare, time_split
from src.models import (
    CV_RESULTS_PATH,
    MODEL_NAMES,
    MODELS,
    STRATEGIES,
    combinations,
    fit_params,
    load_training_data,
    model_path,
)

FIG_DIR = PROJECT_ROOT / "reports" / "figures"
TAB_DIR = PROJECT_ROOT / "reports" / "tables"

N_SPLITS = 5
N_REPEATS = 2
ALPHA = 0.05

BASELINE = "baseline"
BASELINE_NAME = "Majority class baseline"
STRATEGY_NAMES = {"none": "No resampling", "class_weight": "Class weights", "smoten": "SMOTEN"}
SHORT_NAMES = {"dt": "Decision Tree", "ada": "AdaBoost", "knn": "KNN", "nb": "Naive Bayes",
               "rf": "Random Forest\n(reference)"}

METRICS = ["accuracy", "balanced_accuracy", "macro_f1", "recall_low", "recall_medium", "recall_high",
           "roc_auc_ovr", "pr_auc_ovr"]
METRIC_LABELS = {
    "accuracy": "Acc.", "balanced_accuracy": "Bal. acc.", "macro_f1": "Macro-F1",
    "recall_low": "Rec. Low", "recall_medium": "Rec. Med.", "recall_high": "Rec. High",
    "roc_auc_ovr": "ROC-AUC", "pr_auc_ovr": "PR-AUC",
}

# Same palette as notebooks/01_eda.ipynb. The three strategy colours are categorical.
STRATEGY_COLORS = {"none": "#2a78d6", "class_weight": "#eb6834", "smoten": "#1baf7a"}
SEQ = LinearSegmentedColormap.from_list("blue_seq", ["#cde2fb", "#6da7ec", "#2a78d6", "#0d366b"])
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def set_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "font.family": "sans-serif", "font.size": 9,
        "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 9, "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.grid": False, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "legend.frameon": False, "legend.fontsize": 8,
        "lines.linewidth": 2, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save_fig(fig, name: str) -> None:
    """Save a figure to reports/figures as PNG (300 dpi) and PDF."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved reports/figures/{name}.png and .pdf")


# ---- test-set evaluation ----------------------------------------------------

def load_test_data() -> tuple[pd.DataFrame, pd.Series]:
    """X and y for 2023 to 2024."""
    _, test = time_split(load_data())
    X, y, _ = prepare(test)
    return X, y.astype(str)


def score(y_true, y_pred, proba, classes) -> dict:
    """All test metrics for one classifier.

    ``proba`` has one column per entry of ``classes``, in that order. ROC-AUC and
    PR-AUC are one-vs-rest, macro-averaged over the three classes.
    """
    recalls = recall_score(y_true, y_pred, labels=SEVERITY_ORDER, average=None, zero_division=0)
    onehot = label_binarize(y_true, classes=list(classes))
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=SEVERITY_ORDER, average="macro", zero_division=0),
        "recall_low": recalls[0],
        "recall_medium": recalls[1],
        "recall_high": recalls[2],
        "roc_auc_ovr": roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=list(classes)),
        "pr_auc_ovr": average_precision_score(onehot, proba, average="macro"),
    }


def best_configurations() -> dict[str, str]:
    """For each model, the strategy with the highest tuning macro-F1.

    Read from cv_results.csv, which holds only 2019 to 2022 scores, so the choice
    never looks at the test years.
    """
    cv = pd.read_csv(CV_RESULTS_PATH)
    best = cv[cv["is_best"]]
    chosen = best.loc[best.groupby("model")["mean_f1_macro"].idxmax()].set_index("model")["strategy"]
    return {m: chosen[m] for m in MODELS if m in chosen.index}


def evaluate_test() -> tuple[pd.DataFrame, dict]:
    """Score the baseline and every saved pipeline on the test years.

    Returns the results table and the test predictions keyed by (model, strategy).
    """
    _, y_train, _ = load_training_data()
    X_test, y_test = load_test_data()
    best = best_configurations()

    dummy = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(y_train), 1)), y_train)
    blank = np.zeros((len(y_test), 1))
    rows = [{
        "model": BASELINE, "model_name": BASELINE_NAME, "strategy": f"always {dummy.classes_[dummy.class_prior_.argmax()]}",
        "cv_best": False, "n_test": len(y_test),
        **score(y_test, dummy.predict(blank), dummy.predict_proba(blank), dummy.classes_),
    }]

    predictions = {}
    for model, strategy in combinations():
        path = model_path(model, strategy)
        if not path.exists():
            print(f"{MODEL_NAMES[model]:<26} {strategy:<13} no saved pipeline, skipped")
            continue
        pipe = joblib.load(path)
        y_pred = pipe.predict(X_test)
        predictions[(model, strategy)] = y_pred
        rows.append({
            "model": model, "model_name": MODEL_NAMES[model], "strategy": strategy,
            "cv_best": best.get(model) == strategy, "n_test": len(y_test),
            **score(y_test, y_pred, pipe.predict_proba(X_test), pipe.classes_),
        })
    return pd.DataFrame(rows), {"y_true": y_test, "pred": predictions}


def to_latex(results: pd.DataFrame) -> str:
    """The results table as LaTeX (booktabs). The best pipeline in each column is bold."""
    pipelines = results[results["model"] != BASELINE]
    top = pipelines[METRICS].max().round(3)

    table = pd.DataFrame({
        "Model": results["model_name"].str.replace(" (reference)", r"$^\dagger$", regex=False),
        "Strategy": results["strategy"].map(STRATEGY_NAMES).fillna(results["strategy"].str.replace("always", "Always")),
    })
    for metric in METRICS:
        table[METRIC_LABELS[metric]] = [
            rf"\textbf{{{v:.3f}}}" if model != BASELINE and round(v, 3) == top[metric] else f"{v:.3f}"
            for model, v in zip(results["model"], results[metric])
        ]
    n = f"{int(results['n_test'].iloc[0]):,}".replace(",", "{,}")
    return table.to_latex(
        index=False, escape=False, column_format="ll" + "r" * len(METRICS),
        caption=(f"Test-set performance on the 2023 to 2024 data ($n = {n}$). ROC-AUC and PR-AUC are "
                 r"one-vs-rest, macro-averaged. Bold marks the best pipeline in each column. "
                 r"$^\dagger$Reference model only."),
        label="tab:test_results", position="t",
    )


def save_results(results: pd.DataFrame) -> None:
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(TAB_DIR / "test_results.csv", index=False)
    (TAB_DIR / "test_results.tex").write_text(to_latex(results), encoding="utf-8")
    print("saved reports/tables/test_results.csv and .tex")


# ---- figures ----------------------------------------------------------------

def best_two(results: pd.DataFrame) -> pd.DataFrame:
    """The two models with the highest test macro-F1, each in its best configuration.

    The configuration of each model is the one chosen on the training years
    (``cv_best``), so the test set ranks models but does not pick strategies.
    """
    return results[results["cv_best"]].sort_values("macro_f1", ascending=False).head(2)


def plot_confusion_matrices(results: pd.DataFrame, test: dict) -> None:
    top = best_two(results)
    fig, axes = plt.subplots(1, len(top), figsize=(3.9 * len(top), 3.9), squeeze=False)
    for ax, row in zip(axes[0], top.itertuples()):
        counts = confusion_matrix(test["y_true"], test["pred"][(row.model, row.strategy)], labels=SEVERITY_ORDER)
        share = counts / counts.sum(axis=1, keepdims=True)
        im = ax.imshow(share, cmap=SEQ, vmin=0, vmax=1)
        ax.set_xticks(range(3), SEVERITY_ORDER)
        ax.set_yticks(range(3), SEVERITY_ORDER)
        ax.set_xticks(np.arange(-0.5, 3), minor=True)
        ax.set_yticks(np.arange(-0.5, 3), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="both", length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{share[i, j]:.0%}\n({counts[i, j]:,})", ha="center", va="center", fontsize=8.5,
                        color="white" if share[i, j] > 0.45 else INK)
        ax.set_xlabel("Predicted severity level")
        ax.set_title(f"{MODEL_NAMES[row.model]}\n{STRATEGY_NAMES[row.strategy]}, macro-F1 {row.macro_f1:.3f}",
                     fontsize=9.5)
    axes[0, 0].set_ylabel("True severity level")
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.03)
    cb.set_label("Share of the true class (recall on the diagonal)")
    cb.outline.set_visible(False)
    fig.suptitle("Confusion matrices on the 2023 to 2024 test set, best two models",
                 x=0.01, y=1.04, ha="left", fontsize=11, fontweight="bold")
    save_fig(fig, "fig07_confusion_matrices")


def plot_f1_and_high_recall(results: pd.DataFrame) -> None:
    """Grouped bars, one panel per metric. Every strategy keeps its slot, so KNN has a gap."""
    base = results[results["model"] == BASELINE].iloc[0]
    pipelines = results[results["model"] != BASELINE].set_index(["model", "strategy"])
    width = 0.25
    panels = [("macro_f1", "Macro-F1"), ("recall_high", "High-class recall")]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9), sharey=True)
    for ax, (metric, title) in zip(axes, panels):
        for k, strategy in enumerate(STRATEGIES):
            xs = [i + (k - 1) * width for i, m in enumerate(MODELS) if (m, strategy) in pipelines.index]
            ys = [pipelines.loc[(m, strategy), metric] for m in MODELS if (m, strategy) in pipelines.index]
            ax.bar(xs, ys, width=width, color=STRATEGY_COLORS[strategy], edgecolor="white", linewidth=1.5,
                   label=STRATEGY_NAMES[strategy], zorder=2)
            for x, y in zip(xs, ys):
                ax.text(x, y + 0.012, f"{y:.2f}", ha="center", va="bottom", fontsize=6.5, color=INK2, zorder=4)
        ax.axhline(base[metric], color=INK, linewidth=1, linestyle=(0, (4, 3)), zorder=3,
                   label=f"Majority class baseline (macro-F1 {base['macro_f1']:.2f}, "
                         f"High-class recall {base['recall_high']:.2f})")
        ax.set_title(title)
        ax.set_xticks(range(len(MODELS)), [SHORT_NAMES[m] for m in MODELS])
        ax.set_xlim(-0.55, len(MODELS) - 0.45)
        ax.set_ylim(0, 1.08)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.grid(axis="y")
        ax.tick_params(length=0)
    if ("knn", "class_weight") not in pipelines.index:
        for ax in axes:
            ax.text(MODELS.index("knn"), 0.02, "n/a", ha="center", va="bottom", fontsize=7, color=MUTED)
    axes[0].set_ylabel("Score on the test set")
    handles, labels = axes[0].get_legend_handles_labels()
    order = [1, 2, 3, 0]   # strategies first, baseline line last
    fig.legend([handles[i] for i in order], [labels[i] for i in order], ncol=4, loc="upper left",
               bbox_to_anchor=(0.005, 0.94), handlelength=1.6, columnspacing=1.4)
    fig.suptitle("Macro-F1 and High-class recall by model and imbalance strategy, 2023 to 2024 test set",
                 x=0.01, y=0.995, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save_fig(fig, "fig08_macro_f1_high_recall")


# ---- repeated group k-fold and significance tests ---------------------------

def repeated_group_folds(X, y, groups) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    """5 folds x 2 repeats, grouped by state. Each repeat shuffles the states with its own seed.

    scikit-learn has no RepeatedGroupKFold, so this repeats a shuffled GroupKFold.
    """
    folds = []
    for repeat in range(N_REPEATS):
        splitter = GroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED + repeat)
        for fold, (train, test) in enumerate(splitter.split(X, y, groups)):
            folds.append((repeat + 1, fold + 1, train, test))
    return folds


def _score_fold(model, strategy, X, y, repeat, fold, train, test) -> dict:
    pipe = clone(joblib.load(model_path(model, strategy)))   # unfitted, with the tuned hyperparameters
    y_train = y.iloc[train]
    pipe.fit(X.iloc[train], y_train, **fit_params(model, strategy, y_train))
    y_pred = pipe.predict(X.iloc[test])
    y_true = y.iloc[test]
    return {
        "model": model, "strategy": strategy, "repeat": repeat, "fold": fold,
        "macro_f1": f1_score(y_true, y_pred, labels=SEVERITY_ORDER, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "recall_high": recall_score(y_true, y_pred, labels=["High"], average="macro", zero_division=0),
    }


def repeated_cv(n_jobs: int = -1) -> pd.DataFrame:
    """Per-fold scores of each model's best configuration on the training years.

    Every model is scored on the same ten folds, so the folds are the blocks of
    the Friedman test.
    """
    X, y, groups = load_training_data()
    folds = repeated_group_folds(X, y, groups)
    jobs = [
        delayed(_score_fold)(model, strategy, X, y, *fold)
        for model, strategy in best_configurations().items()
        for fold in folds
    ]
    return pd.DataFrame(Parallel(n_jobs=n_jobs)(jobs))


def significance_tests(scores: pd.DataFrame, metric: str = "macro_f1") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Friedman test and Nemenyi post-hoc test on the per-fold scores.

    Returns the average ranks (1 = best), the Friedman summary and the matrix of
    Nemenyi p-values.
    """
    wide = scores.pivot(index=["repeat", "fold"], columns="model", values=metric)
    wide = wide[[m for m in MODELS if m in wide.columns]]
    n, k = wide.shape

    stat, p = friedmanchisquare(*[wide[m] for m in wide.columns])
    ranks = wide.rank(axis=1, ascending=False).mean()
    # Nemenyi critical difference: two models differ at ALPHA if their average ranks differ by more than this.
    cd = studentized_range.ppf(1 - ALPHA, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * n))

    strategy = scores.drop_duplicates("model").set_index("model")["strategy"]
    average_ranks = pd.DataFrame({
        "model_name": [MODEL_NAMES[m] for m in wide.columns],
        "strategy": strategy[wide.columns].values,
        f"mean_{metric}": wide.mean().values,
        f"std_{metric}": wide.std(ddof=1).values,
        "average_rank": ranks.values,
    }, index=wide.columns).sort_values("average_rank")

    friedman = pd.DataFrame([{
        "metric": metric, "n_models": k, "n_folds": n, "friedman_chi2": stat, "dof": k - 1, "p_value": p,
        "alpha": ALPHA, "nemenyi_critical_difference": cd,
    }])

    nemenyi = sp.posthoc_nemenyi_friedman(wide.to_numpy())
    nemenyi.index = nemenyi.columns = [MODEL_NAMES[m] for m in wide.columns]
    return average_ranks, friedman, nemenyi


def save_significance(scores, average_ranks, friedman, nemenyi) -> None:
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_csv(TAB_DIR / "repeated_cv_scores.csv", index=False)
    average_ranks.to_csv(TAB_DIR / "friedman_average_ranks.csv", index_label="model")
    friedman.to_csv(TAB_DIR / "friedman_test.csv", index=False)
    nemenyi.to_csv(TAB_DIR / "nemenyi_p_values.csv", index_label="model")
    print("saved reports/tables/repeated_cv_scores.csv, friedman_average_ranks.csv, "
          "friedman_test.csv and nemenyi_p_values.csv")


# ---- entry point ------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the saved CrashLens pipelines.")
    parser.add_argument("--skip-cv", action="store_true", help="skip the repeated CV and significance tests")
    parser.add_argument("--n-jobs", type=int, default=-1, help="parallel workers for the repeated CV")
    args = parser.parse_args(argv)

    seed_everything()
    set_style()
    pd.set_option("display.width", 200)

    results, test = evaluate_test()
    save_results(results)
    print(f"\nTest set, 2023 to 2024 ({int(results['n_test'].iloc[0]):,} rows)")
    print(results.drop(columns=["model", "n_test"]).to_string(index=False, float_format="{:.3f}".format))
    print()
    plot_confusion_matrices(results, test)
    plot_f1_and_high_recall(results)

    if not args.skip_cv:
        scores = repeated_cv(n_jobs=args.n_jobs)
        average_ranks, friedman, nemenyi = significance_tests(scores)
        save_significance(scores, average_ranks, friedman, nemenyi)
        f = friedman.iloc[0]
        print(f"\nRepeated group k-fold ({N_SPLITS} folds x {N_REPEATS} repeats) on 2019 to 2022, macro-F1")
        print(average_ranks.to_string(index=False, float_format="{:.3f}".format))
        print(f"\nFriedman chi2({int(f['dof'])}) = {f['friedman_chi2']:.3f}, p = {f['p_value']:.4g}; "
              f"Nemenyi critical difference at {ALPHA} = {f['nemenyi_critical_difference']:.3f}")
        print("\nNemenyi p-values")
        print(nemenyi.to_string(float_format="{:.4f}".format))
    return 0


if __name__ == "__main__":
    # Import through the package so the saved pipelines resolve src.models, not __main__.
    from src.evaluate import main as _main

    raise SystemExit(_main())
