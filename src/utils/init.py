from argparse import ArgumentParser
from pathlib import Path

import torch
from ruamel.yaml import YAML

import logging

log = logging.getLogger("gear-cm")


def get_arguments():
    # Init arguments
    parser = ArgumentParser()
    parser.add_argument(f"--config", default="base", type=str)

    # Parse args
    args = parser.parse_args()

    return args


def setup_config(config_name, config_override_name=None):
    config_dir = Path(__file__).resolve().parent.parent.parent / "configs"
    yaml = YAML()

    # Read config
    config = None
    config_path = config_dir / (config_name + ".yaml")
    with open(config_path) as stream:
        config = yaml.load(stream)

    # Use override config if specified
    if not config_override_name is None:

        # Read override config
        config_override = None
        override_path = config_dir / (config_override_name + ".yaml")
        with open(override_path) as stream:
            config_override = yaml.load(stream)

        # Update config with values from override
        config.update(config_override)

    return dict(config)


def setup_device(override=None):
    # Enable cuDNN autotuner (speed optimization)
    torch.backends.cudnn.benchmark = True

    # Set torch download directory (relevant for weights of pretrained networks)
    torch.hub.set_dir("./torch_downloads")

    # Setup device
    device_type = "cpu"
    if override is None:
        if torch.cuda.is_available():
            device_type = "cuda"
            torch.set_float32_matmul_precision("high")
        elif torch.backends.mps.is_built():
            device_type = "mps"
            # device_type = "cpu"

        device = torch.device(device_type)
    else:
        device = torch.device(override)

    log.info("")
    log.info(f"Using device: {device}")
    # Additional info when using cuda
    if device.type == "cuda":
        log.info(torch.cuda.get_device_name(0))

    return device
