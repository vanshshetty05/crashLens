"""Single-column charts for the IEEE paper, drawn from the result tables.

The figures made by the other scripts are sized for a full page or a slide.
These four are drawn at the width of one IEEE column (3.5 in) with type that
stays readable at that size. Every value shown comes from reports/tables, so
run this after the evaluate, robustness and explain phases.

    fig14_paper_model_comparison   macro-F1 and High recall by model and strategy
    fig15_paper_generalisation     time-split test score against held-out-State CV
    fig16_paper_robustness         split design, leakage and drift on the Decision Tree
    fig17_paper_importance         permutation importance and mean |SHAP| by feature

Run from the project root:

    python -m src.paper_figures
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluate import (
    AXIS, BASELINE, GRID, INK, INK2, STRATEGY_COLORS, STRATEGY_NAMES, TAB_DIR, save_fig, set_style,
)
from src.models import MODELS, STRATEGIES

COLUMN = 3.5      # width of one IEEE column, in inches
BLUE, ORANGE = "#2a78d6", "#eb6834"
SHORT = {"dt": "DT", "ada": "AdaBoost", "knn": "KNN", "nb": "NB", "rf": "RF (ref.)"}
MARKERS = {"none": "o", "class_weight": "s", "smoten": "^"}     # shape as well as colour
FEATURE_LABELS = {
    "state_harmonised": "State", "region": "Region", "state_type": "State or UT",
    "factor_value": "Factor value", "factor_group": "Factor group", "log_accidents": "Accidents", "year": "Year",
}


def paper_style() -> None:
    set_style()
    plt.rcParams.update({
        "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.linewidth": 0.6, "grid.linewidth": 0.5,
    })


def model_comparison() -> None:
    """Dot plot: one column of dots per model, one dot per strategy. The scale does not start at zero."""
    results = pd.read_csv(TAB_DIR / "test_results.csv")
    base = results[results["model"] == BASELINE].iloc[0]
    pipes = results[results["model"] != BASELINE].set_index(["model", "strategy"])
    panels = [("macro_f1", "Macro-F1"), ("recall_high", "High-class recall")]

    fig, axes = plt.subplots(2, 1, figsize=(COLUMN, 3.15), sharex=True)
    for ax, (metric, title) in zip(axes, panels):
        for k, strategy in enumerate(STRATEGIES):
            xs = [i + (k - 1) * 0.27 for i, m in enumerate(MODELS) if (m, strategy) in pipes.index]
            ys = [pipes.loc[(m, strategy), metric] for m in MODELS if (m, strategy) in pipes.index]
            ax.scatter(xs, ys, s=22, marker=MARKERS[strategy], color=STRATEGY_COLORS[strategy],
                       edgecolor="white", linewidth=0.6, zorder=3, label=STRATEGY_NAMES[strategy])
            for x, y in zip(xs, ys):
                ax.text(x, y + 0.006, f"{y:.3f}", ha="center", va="bottom", rotation=90, fontsize=5.5, color=INK)
        low, high = pipes[metric].min(), pipes[metric].max()
        ax.set_ylim(low - 0.015, high + 0.05)
        ax.set_title(title)
        ax.grid(axis="y")
        ax.tick_params(length=0)
        for i in range(1, len(MODELS)):
            ax.axvline(i - 0.5, color=GRID, linewidth=0.5)
    axes[1].set_xticks(range(len(MODELS)), [SHORT[m] for m in MODELS])
    axes[1].set_xlim(-0.5, len(MODELS) - 0.5)
    axes[0].legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.13), handletextpad=0.2, columnspacing=1.0)
    fig.text(0.5, 0.005, f"Majority-class baseline (off the scale): macro-F1 {base['macro_f1']:.3f}, "
             f"High recall {base['recall_high']:.3f}", ha="center", fontsize=6, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    save_fig(fig, "fig14_paper_model_comparison")


def generalisation() -> None:
    """Each model's selected configuration: held-out-State CV mean against the time-split test score."""
    cv = pd.read_csv(TAB_DIR / "friedman_average_ranks.csv").set_index("model")
    test = pd.read_csv(TAB_DIR / "test_results.csv")
    test = test[test["cv_best"]].set_index("model")["macro_f1"]
    order = list(cv.sort_values("mean_macro_f1").index)
    y = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(COLUMN, 1.9))
    ax.hlines(y, cv.loc[order, "mean_macro_f1"], test[order], color=AXIS, linewidth=1.5, zorder=1)
    ax.errorbar(cv.loc[order, "mean_macro_f1"], y, xerr=cv.loc[order, "std_macro_f1"], fmt="none",
                ecolor=INK2, elinewidth=0.7, capsize=2, zorder=2)
    ax.scatter(cv.loc[order, "mean_macro_f1"], y, s=26, marker="D", color=ORANGE, edgecolor="white",
               linewidth=0.6, zorder=3, label="Held-out States (grouped CV, mean and SD)")
    ax.scatter(test[order], y, s=28, marker="o", color=BLUE, edgecolor="white", linewidth=0.6, zorder=3,
               label="Known States, later years (test set)")
    for yi, m in zip(y, order):
        ax.text(cv.at[m, "mean_macro_f1"], yi + 0.2, f"{cv.at[m, 'mean_macro_f1']:.3f}", ha="center",
                va="bottom", fontsize=5.8, color=INK)
        ax.text(test[m] + 0.012, yi, f"{test[m]:.3f}", ha="left", va="center", fontsize=5.8, color=INK)
    ax.set_yticks(y, [f"{SHORT[m]}" for m in order])
    ax.set_ylim(-0.6, len(order) - 0.3)
    ax.set_xlim(0.25, 0.86)
    ax.set_xlabel("Macro-F1")
    ax.grid(axis="x")
    ax.tick_params(length=0)
    ax.spines["left"].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=1, handletextpad=0.3, labelspacing=0.25)
    fig.tight_layout()
    save_fig(fig, "fig15_paper_generalisation")


def robustness() -> None:
    """Macro-F1 of the Decision Tree under each control setup. Flawed setups are hatched."""
    table = pd.read_csv(TAB_DIR / "robustness.csv")
    rows = [                       # (experiment, setup in the table, label, flawed?)
        ("1 Split design", "GroupKFold(5) by state", "Split by State (5-fold)", False),
        ("1 Split design", "Random stratified 5-fold", "Random rows (5-fold)", True),
        ("2 Leakage", "Agreed features", "Time split, agreed features", False),
        ("1 Split design", "Random row split, all years", "Random rows, all years", True),
        ("2 Leakage", "Agreed features + severity_index", "+ severity index (leak)", True),
        ("2 Leakage", "Agreed features + killed", "+ persons killed (leak)", True),
        ("3 Drift over time", "Tested on 2023", "Tested on 2023 only", False),
        ("3 Drift over time", "Tested on 2024", "Tested on 2024 only", False),
    ]
    keyed = table.set_index(["experiment", "setup"])["macro_f1"]
    values = [keyed[(e, s)] for e, s, _, _ in rows]
    y = np.arange(len(rows))[::-1]

    fig, ax = plt.subplots(figsize=(COLUMN, 2.2))
    for yi, v, (_, _, _, flawed) in zip(y, values, rows):
        ax.barh(yi, v, height=0.62, color=ORANGE if flawed else BLUE, hatch="////" if flawed else None,
                edgecolor="white", linewidth=0.4, zorder=2)
        ax.text(v + 0.012, yi, f"{v:.3f}", va="center", fontsize=6, color=INK)
    ax.axhline(y[1] - 0.5, color=AXIS, linewidth=0.6)      # between the cross-validated and the test-set rows
    ax.set_yticks(y, [label for _, _, label, _ in rows])
    ax.set_xlim(0, 1.12)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("Macro-F1 of the Decision Tree")
    ax.grid(axis="x")
    ax.tick_params(length=0)
    ax.spines["bottom"].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=BLUE, edgecolor="white"),
               plt.Rectangle((0, 0), 1, 1, facecolor=ORANGE, edgecolor="white", hatch="////")]
    ax.legend(handles, ["Correct setup", "Flawed setup (inflated)"], ncol=2, loc="lower center",
              bbox_to_anchor=(0.3, 1.0), handlelength=1.4, columnspacing=1.0)
    fig.tight_layout()
    save_fig(fig, "fig16_paper_robustness")


def importance() -> None:
    """Two attribution methods on two models, side by side, with one feature order."""
    perm = pd.read_csv(TAB_DIR / "permutation_importance.csv").set_index("feature")["importance_mean"]
    shap = pd.read_csv(TAB_DIR / "shap_mean_abs_by_feature.csv").set_index("feature")["mean_all_classes"]
    order = list(shap.sort_values().index)
    y = np.arange(len(order))
    panels = [(perm, "Permutation importance\n(NB, drop in macro-F1)"),
              (shap, "Mean |SHAP|\n(RF, probability)")]

    fig, axes = plt.subplots(1, 2, figsize=(COLUMN, 1.85), sharey=True)
    for ax, (values, title) in zip(axes, panels):
        v = values[order]
        ax.barh(y, v, height=0.62, color=BLUE, zorder=2)
        for yi, x in zip(y, v):
            ax.text(x + v.max() * 0.04, yi, f"{x:.3f}", va="center", fontsize=6, color=INK)
        ax.set_xlim(0, v.max() * 1.38)
        ax.set_title(title, fontsize=6.2, fontweight="normal", color=INK2)
        ax.grid(axis="x")
        ax.tick_params(length=0)
        ax.spines["bottom"].set_visible(False)
        ax.locator_params(axis="x", nbins=3)
    axes[0].set_yticks(y, [FEATURE_LABELS[f] for f in order])
    fig.tight_layout(w_pad=0.6)
    save_fig(fig, "fig17_paper_importance")


def main() -> int:
    paper_style()
    model_comparison()
    generalisation()
    robustness()
    importance()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
