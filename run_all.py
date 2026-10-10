"""Run the whole CrashLens pipeline and regenerate every table and figure.

    prepare -> train -> evaluate -> robustness -> explain -> insights

Run from the project root, inside the virtual environment:

    python run_all.py                      # everything (about 25 minutes)
    python run_all.py --from evaluate      # this phase and the ones after it
    python run_all.py --only explain insights
    python run_all.py --list

Each phase runs in its own Python process with the same seed settings, and the
run stops at the first phase that fails. After a full run, every generated file
is checked to have been rewritten; any that was not is listed as stale.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEED = 42

# Each phase is a list of commands, given as arguments to the Python interpreter.
PHASES: dict[str, dict] = {
    "prepare": {
        "what": "check the raw CSV and write the train and test split",
        "commands": [["-m", "src.prepare"]],
    },
    "train": {
        "what": "tune and save every model and imbalance strategy",
        "commands": [["-m", "src.models"]],
    },
    "evaluate": {
        "what": "test-set metrics, repeated grouped CV, Friedman and Nemenyi tests",
        "commands": [["-m", "src.evaluate"]],
    },
    "robustness": {
        "what": "split design, leakage and drift checks on the Decision Tree",
        "commands": [["-m", "src.robustness"]],
    },
    "explain": {
        "what": "permutation importance, shallow-tree rules and SHAP",
        "commands": [["-m", "src.explain"]],
    },
    "insights": {
        "what": "EDA notebook, maps, deadliest conditions, state profile, frontend data and paper charts",
        "commands": [
            ["-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
             "--ExecutePreprocessor.timeout=900", "notebooks/01_eda.ipynb"],
            ["-m", "src.maps"],
            ["-m", "src.paper_figures"],
        ],
    },
}

# Folders that hold generated files, and the patterns that count.
OUTPUTS = [
    ("data/processed", "*.csv"), ("models", "*.joblib"), ("reports/tables", "*.*"), ("reports/figures", "*.*"),
    ("frontend/data", "*.*"), ("frontend/img", "*.png"),
]


def child_env() -> dict:
    env = os.environ.copy()
    env.update({
        "PYTHONHASHSEED": str(SEED),      # fixed hash order for sets and dicts of strings
        "MPLBACKEND": "Agg",              # figures are written to files, never shown
        "PYTHONIOENCODING": "utf-8",      # the rule tables contain an arrow character
        "PYTHONUNBUFFERED": "1",
    })
    return env


def run_phase(name: str) -> float:
    start = time.time()
    print(f"\n{'=' * 78}\n{name}: {PHASES[name]['what']}\n{'=' * 78}", flush=True)
    for command in PHASES[name]["commands"]:
        result = subprocess.run([sys.executable, *command], cwd=ROOT, env=child_env())
        if result.returncode != 0:
            raise SystemExit(f"\nPhase '{name}' failed (exit code {result.returncode}): python {' '.join(command)}")
    return time.time() - start


def stale_outputs(started: float) -> list[str]:
    """Generated files that this run did not rewrite."""
    stale = []
    for folder, pattern in OUTPUTS:
        for path in sorted((ROOT / folder).glob(pattern)):
            if path.is_file() and path.name != ".gitkeep" and path.stat().st_mtime < started:
                stale.append(path.relative_to(ROOT).as_posix())
    return stale


def main(argv=None) -> int:
    names = list(PHASES)
    parser = argparse.ArgumentParser(description="Run the CrashLens pipeline.")
    parser.add_argument("--from", dest="start", choices=names, help="start at this phase")
    parser.add_argument("--only", nargs="+", choices=names, help="run only these phases, in pipeline order")
    parser.add_argument("--list", action="store_true", help="list the phases and exit")
    args = parser.parse_args(argv)

    if args.list:
        for name in names:
            print(f"{name:<11} {PHASES[name]['what']}")
        return 0
    if args.start and args.only:
        parser.error("use --from or --only, not both")

    selected = [n for n in names if n in args.only] if args.only else names[names.index(args.start or names[0]):]
    raw = ROOT / "data" / "raw" / "india_road_accidents_2019_2024.csv"
    if not raw.exists():
        raise SystemExit(f"Dataset not found at {raw}.")

    started = time.time()
    durations = {name: run_phase(name) for name in selected}

    print(f"\n{'=' * 78}\nDone in {(time.time() - started) / 60:.1f} minutes (seed {SEED})")
    for name, seconds in durations.items():
        print(f"  {name:<11} {seconds / 60:5.1f} min")

    if selected == names:
        stale = stale_outputs(started)
        if stale:
            print(f"\n{len(stale)} generated file(s) were NOT rewritten by this run:")
            for path in stale:
                print(f"  {path}")
            return 1
        print("\nEvery generated table, figure, model and frontend file was rewritten by this run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
