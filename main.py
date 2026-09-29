import sys
import traceback
import warnings
from pprint import pformat

from src.data.data import setup_data
from src.training.training import run_single_training
from src.utils.init import (
    get_arguments,
    resolve_seed,
    seed_everything,
    setup_config,
    setup_device,
)
from src.utils.logging import setup_logging


def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file, "write") else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))


def main():
    # warnings.showwarning = warn_with_traceback
    # warnings.simplefilter("always")

    # INITIALIZATION
    ################

    # Get terminal arguments
    args = get_arguments()

    # Read config
    config = setup_config(args.config)

    # Seed from the command line, then the config, otherwise a random one
    # * Always stored in the config, so it gets logged and saved with the weights
    seed = resolve_seed(args.seed if args.seed is not None else config.get("seed"))
    config["seed"] = seed
    seed_everything(seed)

    # Setup logging
    log = setup_logging(log_to_file=config["log"])

    # Determine device
    # Override if necessary
    device = setup_device(override="cpu")
    # device = setup_device()

    log.info("")
    log.info("")
    log.info("")
    log.info("")
    log.info("")
    log.info("##")
    log.info(f"# RUN: {config['name']} - {config['model']}")
    log.info("##")
    log.info("")

    log.info(pformat(config, width=1, sort_dicts=False))

    # RUN A TRAINING RUN
    ####################

    # Initialize data
    train_loaders, validation_loader, test_loader = setup_data(
        config, device, seed=seed
    )

    # Run training
    run_single_training(
        train_loaders,
        validation_loader,
        test_loader,
        config,
        device,
    )


if __name__ == "__main__":
    main()
