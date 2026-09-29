from argparse import ArgumentParser
from pathlib import Path
from pprint import pformat

from src.data.data import setup_data
from src.training.training import run_single_training
from src.utils.init import setup_config, setup_device
from src.utils.logging import setup_logging

MODEL_WEIGHT_DIR = Path(__file__).resolve().parent / "model_weights"

# Letter used for each dataset in the output directory name, in the order they appear there
DATASET_CODES = {"UNSW": "U", "MCC5": "M", "AGFD": "A", "ASD": "S"}


def get_dataset_code(train_datasets):
    """Name the training dataset combination, e.g. [ALL_A] -> "A", [UNSW, MCC5] -> "UM"."""
    letters = set()
    for name in train_datasets:
        if name.startswith("ALL_"):
            # Same parsing as get_dataloader: ALL_<letters>
            letters.update(name.split("_")[1])
        elif name in DATASET_CODES:
            letters.add(DATASET_CODES[name])
        else:
            raise SystemExit(f"Unknown training dataset: {name}")

    return "".join(code for code in DATASET_CODES.values() if code in letters)


def get_output_dir(dataset_code):
    """First free directory out of <code>, <code>_1, <code>_2, ..."""
    output_dir = MODEL_WEIGHT_DIR / dataset_code
    suffix = 1
    while output_dir.exists():
        output_dir = MODEL_WEIGHT_DIR / f"{dataset_code}_{suffix}"
        suffix += 1

    return output_dir


def parse_args():
    parser = ArgumentParser(
        description=(
            "Train a model several times with the same config. The weights of every run "
            "go into model_weights/<datasets>, e.g. model_weights/UM for UNSW + MCC5, "
            "or <datasets>_1, <datasets>_2, ... if that already exists."
        )
    )
    parser.add_argument(
        "config", help="Config name in configs/, without .yaml (e.g. FINAL)."
    )
    parser.add_argument(
        "--runs", type=int, default=5, help="Number of training runs (default: 5)."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = setup_device(override="cpu")
    config = setup_config(args.config)
    log = setup_logging(log_to_file=config["log"])

    if not config["save"]:
        # Saving the weights is the whole point of this script
        log.warning("Config has save: false, overriding to true")
        config["save"] = True

    output_dir = get_output_dir(get_dataset_code(config["train_datasets"]))

    log.info("")
    log.info("##")
    log.info(f"# REPEATED RUNS: {config['name']} - {config['model']} x {args.runs}")
    log.info(f"# Saving weights to: {output_dir}")
    log.info("##")
    log.info("")
    log.info(pformat(config, width=1, sort_dicts=False))

    for run_i in range(args.runs):
        log.info("")
        log.info(f"# RUN {run_i + 1}/{args.runs}")
        log.info("")

        # Fresh data loaders for every run
        train_loaders, validation_loader, test_loader = setup_data(config, device)

        run_single_training(
            train_loaders,
            validation_loader,
            test_loader,
            config,
            device,
            model_weight_dir=output_dir,
        )

    log.info("")
    log.info(f"Finished {args.runs} runs, weights in {output_dir}")


if __name__ == "__main__":
    main()
