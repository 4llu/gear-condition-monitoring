import logging

import numpy as np

from src.data.sampler_AGFD import get_AGFD_data
from src.data.sampler_ALL import get_ALL_data
from src.data.sampler_MCC5 import get_MCC5_data
from src.data.sampler_UNSW import get_UNSW_data

log = logging.getLogger("gear-cm")


def get_dataloader(name, split, rng, config, device):
    if name == "UNSW":
        return get_UNSW_data(split, rng, config, device)
    elif name == "AGFD":
        return get_AGFD_data(split, rng, config, device)
    elif name == "MCC5":
        return get_MCC5_data(split, rng, config, device)
    # ALL -> final configuration
    elif "ALL_" in name:
        incl_data = name.split("_")[1]
        return get_ALL_data(
            split,
            rng,
            config,
            device,
            UNSW_included="U" in incl_data,
            MCC5_included="M" in incl_data,
            AGFD_included="A" in incl_data,
            ASD_included="S" in incl_data,
        )
    # elif name == "ALL_UMA":
    #     return get_ALL_data(
    #         split,
    #         rng,
    #         config,
    #         device,
    #         UNSW_included=True,
    #         MCC5_included=True,
    #         AGFD_included=True,
    #     )
    # elif name == "ALL_MA":
    #     return get_ALL_data(
    #         split,
    #         rng,
    #         config,
    #         device,
    #         UNSW_included=False,
    #         MCC5_included=True,
    #         AGFD_included=True,
    #     )
    # elif name == "ALL_UA":
    #     return get_ALL_data(
    #         split,
    #         rng,
    #         config,
    #         device,
    #         UNSW_included=True,
    #         MCC5_included=False,
    #         AGFD_included=True,
    #     )
    # elif name == "ALL_UM":
    #     return get_ALL_data(
    #         split,
    #         rng,
    #         config,
    #         device,
    #         UNSW_included=True,
    #         MCC5_included=True,
    #         AGFD_included=False,
    #     )
    # elif name == "ALL_U":
    #     return get_ALL_data(
    #         split,
    #         rng,
    #         config,
    #         device,
    #         UNSW_included=True,
    #         MCC5_included=False,
    #         AGFD_included=False,
    #     )
    # elif name == "ALL_M":
    #     return get_ALL_data(
    #         split,
    #         rng,
    #         config,
    #         device,
    #         UNSW_included=False,
    #         MCC5_included=True,
    #         AGFD_included=False,
    #     )
    # elif name == "ALL_A":
    #     return get_ALL_data(
    #         split,
    #         rng,
    #         config,
    #         device,
    #         UNSW_included=False,
    #         MCC5_included=False,
    #         AGFD_included=True,
    #     )
    # OLD
    # elif name == "ALL":
    #     return get_ALL_data(split, rng, config, device, UNSW_included=False)
    # elif name == "ALL+UNSW":
    #     return get_ALL_data(split, rng, config, device, UNSW_included=True)
    # elif name == "ALL-AGFD":
    #     return get_ALL_data(split, rng, config, device, AGFD_included=False)
    # elif name == "ALL-MCC5":
    #     return get_ALL_data(split, rng, config, device, MCC5_included=False)
    else:
        s = f"No such data configuration as '{name}'!"
        log.error(s)
        raise Exception(s)


def setup_data(config, device):
    log.debug("")
    log.debug("PREPARING DATA")

    rng = np.random.default_rng()

    train_loaders = []
    validation_loaders = []
    test_loaders = []

    for dataset_name in config["train_datasets"]:
        train_loaders.append(
            iter(get_dataloader(dataset_name, "train", rng, config, device))
        )
    for dataset_name in config["validation_datasets"]:
        validation_loaders.append(
            iter(get_dataloader(dataset_name, "validation", rng, config, device))
        )
    for dataset_name in config["test_datasets"]:
        test_loaders.append(
            iter(get_dataloader(dataset_name, "test", rng, config, device))
        )

    log.debug("PREPARING DATA DONE")

    return train_loaders, validation_loaders[0], test_loaders[0]
