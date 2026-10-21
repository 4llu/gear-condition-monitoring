import logging

import numpy as np

from src.data.sampler_ALL import get_ALL_data

log = logging.getLogger("gear-cm")

DATASET_LETTERS = {"U": "UNSW", "M": "MCC5", "A": "AGFD"}


def get_dataloader(name, split, rng, config, device):
    """
    Dataloader for `ALL_<letters>`, e.g. `ALL_UM` for UNSW + MCC5 (U: UNSW, M: MCC5,
    A: AGFD). The `<dataset>_<split>_*` entries of the config define each dataset.
    """
    if not name.startswith("ALL_") or not set(name[4:]) <= set(DATASET_LETTERS):
        s = f"No such data configuration as '{name}', use ALL_<letters> with {list(DATASET_LETTERS)}"
        log.error(s)
        raise ValueError(s)

    letters = name[4:]
    return get_ALL_data(
        split,
        rng,
        config,
        device,
        UNSW_included="U" in letters,
        MCC5_included="M" in letters,
        AGFD_included="A" in letters,
    )


def setup_data(config, device, seed=None):
    """
    Training loaders for every entry of `train_datasets` and one validation loader for
    `validation_datasets` (None if the list is empty or missing).

    Validation draws from its own generator, so the validation setup doesn't change the
    training samples or augmentations of a seed.
    """
    log.debug("")
    log.debug("PREPARING DATA")

    train_rng = np.random.default_rng(seed)
    validation_rng = np.random.default_rng(None if seed is None else [seed, 1])

    train_loaders = [
        iter(get_dataloader(name, "train", train_rng, config, device))
        for name in config["train_datasets"]
    ]

    validation_datasets = config.get("validation_datasets") or []
    if len(validation_datasets) > 1:
        raise ValueError("Only one validation dataset is supported")
    validation_loader = None
    if validation_datasets:
        validation_loader = iter(
            get_dataloader(
                validation_datasets[0], "validation", validation_rng, config, device
            )
        )

    log.debug("PREPARING DATA DONE")

    return train_loaders, validation_loader
