"""
Turn every seed sweep CSV in this folder into a Markdown report with the same name.
CSVs that already have a Markdown report are skipped.

    python results/csv_to_md.py
"""

import csv
import statistics
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent

SEED_SWEEP_COLUMNS = {
    "seed",
    "train_datasets",
    "test_dataset",
    "weight_dir",
    "ensemble",
    "accuracy",
}


def percent(value):
    return f"{value:.2%}"


def table(header, rows, align=None):
    """Markdown table, `align` holds "l" or "r" for each column (default: left)."""
    align = align or ["l"] * len(header)
    separator = [":---" if a == "l" else "---:" for a in align]
    lines = [header, separator, *rows]
    return "\n".join("| " + " | ".join(str(c) for c in line) + " |" for line in lines)


def read_sweep(csv_path):
    with open(csv_path, newline="") as stream:
        reader = csv.DictReader(stream)
        if not SEED_SWEEP_COLUMNS.issubset(reader.fieldnames or []):
            return None
        rows = list(reader)

    for row in rows:
        row["seed"] = int(row["seed"])
        row["accuracy"] = float(row["accuracy"])
    return rows


def build_report(title, rows):
    # Training combinations in the order they appear, e.g. ("ALL_UM", "AGFD")
    scenarios = list(dict.fromkeys((r["train_datasets"], r["test_dataset"]) for r in rows))
    seeds = list(dict.fromkeys(r["seed"] for r in rows))

    means = {
        (r["seed"], r["train_datasets"], r["test_dataset"]): r["accuracy"]
        for r in rows
        if r["ensemble"] == "mean"
    }
    ensembles = [r for r in rows if r["ensemble"] != "mean"]

    lines = [f"# {title}", ""]
    lines.append(
        f"{len(seeds)} seed(s), {len(scenarios)} training combination(s). "
        "Each model set is tested on the dataset left out of training. "
        "Accuracies are the average over the ensembles of a seed unless stated otherwise."
    )
    lines.append("")

    # Summary over seeds
    lines += ["## Summary over seeds", ""]
    summary_rows = []
    for train, test in scenarios:
        values = [means[(s, train, test)] for s in seeds if (s, train, test) in means]
        if not values:
            continue
        std = statistics.stdev(values) if len(values) > 1 else None
        summary_rows.append(
            [
                train,
                test,
                len(values),
                percent(statistics.mean(values)),
                percent(std) if std is not None else "–",
                percent(min(values)),
                percent(max(values)),
            ]
        )
    lines.append(
        table(
            ["Training", "Test", "Seeds", "Mean", "Std", "Min", "Max"],
            summary_rows,
            ["l", "l", "r", "r", "r", "r", "r"],
        )
    )
    lines.append("")

    # Mean per seed, one column per training combination
    lines += ["## Mean accuracy per seed", ""]
    per_seed_rows = []
    for seed in seeds:
        cells = []
        for train, test in scenarios:
            value = means.get((seed, train, test))
            cells.append(percent(value) if value is not None else "–")
        per_seed_rows.append([seed, *cells])
    lines.append(
        table(
            ["Seed", *[f"{train} → {test}" for train, test in scenarios]],
            per_seed_rows,
            ["l"] + ["r"] * len(scenarios),
        )
    )
    lines.append("")

    # Every ensemble individually
    lines += ["## Individual ensembles", ""]
    for train, test in scenarios:
        scenario_rows = [
            r for r in ensembles if r["train_datasets"] == train and r["test_dataset"] == test
        ]
        if not scenario_rows:
            continue
        lines += [f"### {train} → {test}", ""]
        lines.append(
            table(
                ["Seed", "Weights", "Ensemble", "Accuracy"],
                [
                    [r["seed"], f"`{r['weight_dir']}`", r["ensemble"], percent(r["accuracy"])]
                    for r in scenario_rows
                ],
                ["l", "l", "r", "r"],
            )
        )
        lines.append("")

    return "\n".join(lines)


def main():
    for csv_path in sorted(RESULTS_DIR.glob("*.csv")):
        md_path = csv_path.with_suffix(".md")
        if md_path.exists():
            print(f"Skipping {csv_path.name}, {md_path.name} already exists")
            continue

        rows = read_sweep(csv_path)
        if rows is None:
            print(f"Skipping {csv_path.name}, not a seed sweep CSV")
            continue
        if not rows:
            print(f"Skipping {csv_path.name}, no results yet")
            continue

        md_path.write_text(build_report(csv_path.stem, rows) + "\n")
        print(f"Wrote {md_path.name}")


if __name__ == "__main__":
    main()
