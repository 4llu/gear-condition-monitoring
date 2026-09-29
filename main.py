import sys
import traceback
import warnings
from pprint import pformat

from src.data.data import setup_data
from src.training.training import run_single_training
from src.utils.init import get_arguments, setup_config, setup_device
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

    # Determine device
    # Override if necessary
    device = setup_device(override="cpu")
    # device = setup_device()

    # Read config
    config = setup_config(args.config)

    # Setup logging
    log = setup_logging(log_to_file=config["log"])

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
    train_loaders, validation_loader, test_loader = setup_data(config, device)

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
