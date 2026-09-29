import random
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from ruamel.yaml import YAML

import logging

log = logging.getLogger("gear-cm")


def get_arguments():
    # Init arguments
    parser = ArgumentParser()
    parser.add_argument(f"--config", default="base", type=str)
    parser.add_argument(
        "--seed", default=None, type=int, help="Overrides the config's seed"
    )

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


def resolve_seed(seed=None):
    """Return the given seed, or a fresh random one if None, so it can always be recorded."""
    if seed is None:
        seed = int(np.random.SeedSequence().generate_state(1)[0])
    return int(seed)


def seed_everything(seed):
    """Seed every random number generator used during training."""
    random.seed(seed)
    # Safety net for any code still using the global NumPy RNG
    np.random.seed(seed)
    # Model initialization and dropout
    torch.manual_seed(seed)
    # Warn instead of failing if an op has no deterministic implementation
    torch.use_deterministic_algorithms(True, warn_only=True)


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
