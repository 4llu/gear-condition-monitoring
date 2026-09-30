import csv
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import numpy as np

import test
from src.utils.init import setup_config, setup_device
from src.utils.logging import setup_logging
from src.training.training import CHECKPOINT_INTERVAL, last_checkpoint_num
from train_repeated import train_ensemble

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Training datasets -> left out dataset used for testing
SCENARIOS = {
    "ALL_UM": "AGFD",
    "ALL_UA": "MCC5",
    "ALL_MA": "UNSW",
}

CSV_COLUMNS = [
    "seed",
    "train_datasets",
    "test_dataset",
    "weight_dir",
    "ensemble",
    "accuracy",
]


def parse_args():
    parser = ArgumentParser(
        description=(
            "For every seed, train an ensemble on each of ALL_UM, ALL_UA and ALL_MA and "
            "test it with test.py on the dataset left out of training. Results are "
            "written to a CSV, one row per ensemble plus their mean."
        )
    )
    parser.add_argument(
        "seeds",
        type=int,
        nargs="+",
        help="Base seeds. Model i of a seed is trained with seed + i.",
    )
    parser.add_argument(
        "--config",
        default="FINAL",
        help="Config name in configs/, without .yaml (default: FINAL).",
    )
    parser.add_argument(
        "--models",
        type=int,
        default=9,
        help="Models trained per seed and training dataset combination (default: 9).",
    )
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=test.ENSEMBLE_SIZE,
        help=f"Models per ensemble when testing (default: test.py's {test.ENSEMBLE_SIZE}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV file to write (default: results/seed_sweep_<timestamp>.csv).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.models < args.ensemble_size:
        raise SystemExit(
            f"--models ({args.models}) must be at least --ensemble-size ({args.ensemble_size})"
        )

    output = args.output or (
        RESULTS_DIR / f"seed_sweep_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    device = setup_device(override="cpu")
    training_config = setup_config(args.config)
    log = setup_logging(log_to_file=training_config["log"])

    # Check before training that the trainings will save the checkpoint test.py loads
    last_checkpoint = last_checkpoint_num(training_config["max_batches"])
    if test.ENSEMBLE_CHECKPOINT_NUM > last_checkpoint:
        raise SystemExit(
            f"test.py loads checkpoint {test.ENSEMBLE_CHECKPOINT_NUM}.pth, but with "
            f"max_batches: {training_config['max_batches']} in {args.config}.yaml the "
            f"last checkpoint saved is {last_checkpoint}.pth (one every "
            f"{CHECKPOINT_INTERVAL} batches). Set ENSEMBLE_CHECKPOINT_NUM in test.py to "
            f"at most {last_checkpoint}, or max_batches to more than "
            f"{test.ENSEMBLE_CHECKPOINT_NUM * CHECKPOINT_INTERVAL}."
        )

    # test.py reads its ensemble size from this module level setting
    test.ENSEMBLE_SIZE = args.ensemble_size
    leftover = args.models % args.ensemble_size
    if leftover:
        log.warning(
            f"{args.models} models don't split evenly into ensembles of "
            f"{args.ensemble_size}, {leftover} per training will be unused in testing"
        )

    log.info(f"Writing results to {output}")
    with open(output, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for seed in args.seeds:
            for train_datasets, test_dataset in SCENARIOS.items():
                log.info("")
                log.info("####")
                log.info(
                    f"# SEED {seed}: train {train_datasets}, test {test_dataset}"
                )
                log.info("####")

                # Fresh config for every training, so nothing leaks between them
                config = setup_config(args.config)
                config["train_datasets"] = [train_datasets]

                weight_dir = train_ensemble(config, args.models, seed, device, log)

                # Uses the settings stored with the weights
                accuracies = test.evaluate(weight_dir, test_dataset)

                row = {
                    "seed": seed,
                    "train_datasets": train_datasets,
                    "test_dataset": test_dataset,
                    "weight_dir": weight_dir.name,
                }
                for ensemble_i, accuracy in enumerate(accuracies, start=1):
                    writer.writerow({**row, "ensemble": ensemble_i, "accuracy": accuracy})
                writer.writerow(
                    {**row, "ensemble": "mean", "accuracy": np.mean(accuracies)}
                )
                # Keep results of finished trainings if a later one fails
                stream.flush()

    log.info("")
    log.info(f"Done, results in {output}")


if __name__ == "__main__":
    main()
