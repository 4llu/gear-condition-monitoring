import itertools
import logging
from functools import partial
from pathlib import Path

import pandas as pd
from torch.utils.data import BatchSampler, DataLoader, Dataset

from src.data.preprocessing_batch import BatchPreprocessor, collate_all
from src.data.preprocessing_individual import IndividualPreprocessor
from src.data.sampler_AGFD import AGFD_Dataset
from src.data.sampler_MCC5 import MCC5_Dataset
from src.data.sampler_UNSW import UNSW_Dataset

log = logging.getLogger("gear-cm")

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

# The only severity of missing tooth in MCC5-THU
MCC5_MISSING_TOOTH_SEVERITY = "S"


def MCC5_class_severities(config, config_prefix, class_):
    """
    Severities used for an MCC5 fault class: `<prefix>class_severities` can set them
    per class, otherwise `<prefix>severities` applies. Missing tooth only has one.
    """
    if class_ == "missing_tooth":
        return [MCC5_MISSING_TOOTH_SEVERITY]
    return config.get(f"{config_prefix}class_severities", {}).get(
        class_, config[f"{config_prefix}severities"]
    )


class ALL_Dataset(Dataset):
    NAME = "ALL"

    def __init__(self, dataset_AGFD, dataset_MCC5, dataset_UNSW, rng, config, device):
        self.dataset_AGFD = dataset_AGFD
        self.dataset_MCC5 = dataset_MCC5
        self.dataset_UNSW = dataset_UNSW
        self.rng = rng
        self.config = config
        self.device = device

    def __getitem__(self, index):
        if index[0] == "AGFD":
            return self.dataset_AGFD[tuple(index[1:])]
        elif index[0] == "MCC5":
            return self.dataset_MCC5[tuple(index[1:])]
        elif index[0] == "UNSW":
            return self.dataset_UNSW[tuple(index[1:])]
        else:
            raise ValueError(f"Unknown dataset type: {index[0]}")


class ALL_BatchSampler(BatchSampler):
    def __init__(self, dataset, split, rng, config):
        self.dataset = dataset
        self.rng = rng
        self.config = config

        self.config_prefix_AGFD = f"AGFD_{split}_"
        self.config_prefix_MCC5 = f"MCC5_{split}_"
        self.config_prefix_UNSW = f"UNSW_{split}_"
        self.config_prefix_ALL = f"ALL_{split}_"

        self.AGFD_included = self.dataset.dataset_AGFD is not None
        self.MCC5_included = self.dataset.dataset_MCC5 is not None
        self.UNSW_included = self.dataset.dataset_UNSW is not None

        if self.AGFD_included:
            self.AGFD_options = list(
                itertools.product(
                    self.config[f"{self.config_prefix_AGFD}speeds"],
                    self.config[f"{self.config_prefix_AGFD}loads"],
                    self.config[f"{self.config_prefix_AGFD}sensors"],
                )
            )
            print(f"Batch options AGFD: {len(self.AGFD_options)}")  # 15

        if self.MCC5_included:
            self.MCC5_options = list(
                itertools.product(
                    self.config[f"{self.config_prefix_MCC5}speeds"],
                    self.config[f"{self.config_prefix_MCC5}loads"],
                    self.config[f"{self.config_prefix_MCC5}sensors"],
                    ["torque", "speed"],
                )
            )
            # 5 and 15 Nm loads exist only during torque circulation
            self.MCC5_options = filter(
                lambda x: not (x[3] == "speed" and x[1] in [5, 15]), self.MCC5_options
            )
            # 1500 and 2500 rpm speeds exist only during speed circulation
            self.MCC5_options = list(
                filter(
                    lambda x: not (x[3] == "torque" and x[0] in [1500, 2500]),
                    self.MCC5_options,
                )
            )
            print(f"Batch options MCC5: {len(self.MCC5_options)}")  # 22

        if self.UNSW_included:
            # [speed, load] pairs to leave out, e.g. UNSW_train_excluded_conditions
            excluded_UNSW_options = {
                tuple(condition)
                for condition in self.config.get(
                    f"{self.config_prefix_UNSW}excluded_conditions", []
                )
            }
            self.UNSW_options = [
                option
                for option in itertools.product(
                    self.config[f"{self.config_prefix_UNSW}speeds"],
                    self.config[f"{self.config_prefix_UNSW}loads"],
                )
                if option not in excluded_UNSW_options
            ]
            print(f"Batch options UNSW: {len(self.UNSW_options)}")  # 12

    def num_query_groups(self, config_prefix, class_):
        """
        Number of query groups (installations, severities or healthy gears) of
        `num_queries` samples for a class in one option.

        With `balance_healthy_like`, classes labelled like healthy (e.g. wear used as
        extra healthy units) take one of healthy's groups each instead of adding
        their own, so every label has the same number of samples in a batch.
        """
        num_groups = self.config[f"{self.config_prefix_ALL}num_installations"]
        if not self.config.get("balance_healthy_like", False):
            return num_groups

        healthy_label = self.config["class_map"]["healthy"]
        healthy_like = [
            c
            for c in self.config[f"{config_prefix}classes"]
            if c != "healthy" and self.config["class_map"][c] == healthy_label
        ]
        if class_ in healthy_like:
            return 1
        if class_ == "healthy":
            assert len(healthy_like) < num_groups, (
                f"Need more than {len(healthy_like)} query groups to balance healthy-like classes"
            )
            return num_groups - len(healthy_like)
        return num_groups

    def __iter__(self):
        while True:
            batch_options_AGFD = None
            batch_options_MCC5 = None
            batch_options_UNSW = None

            if self.AGFD_included:
                batch_options_AGFD = self.rng.choice(
                    self.AGFD_options,
                    size=self.config[f"{self.config_prefix_ALL}num_options"],
                    replace=False,
                )
            if self.MCC5_included:
                batch_options_MCC5 = self.rng.choice(
                    self.MCC5_options,
                    size=self.config[f"{self.config_prefix_ALL}num_options"],
                    replace=False,
                )
            if self.UNSW_included:
                batch_options_UNSW = self.rng.choice(
                    self.UNSW_options,
                    size=self.config[f"{self.config_prefix_ALL}num_options"],
                    replace=False,
                )

            batch = []

            # UNSW samples
            ##

            if self.UNSW_included:
                for option in batch_options_UNSW:
                    group = []
                    speed, load = option
                    speed = int(speed)
                    load = int(load)

                    anchor_severity, healthy_query_severity = self.rng.permutation(
                        ["H1", "H2"],
                    )

                    for class_ in self.config[f"{self.config_prefix_UNSW}classes"]:
                        # HEALTHY
                        ##

                        if class_ == "healthy":
                            # Anchor
                            for _ in range(
                                self.config[f"{self.config_prefix_ALL}num_anchors"]
                            ):
                                # idx: (class, speed, load, severity, OT_method, TSA_size)
                                group.append((
                                    "UNSW",
                                    "healthy",
                                    speed,
                                    load,
                                    anchor_severity,
                                    self.config[f"{self.config_prefix_UNSW}OT_method"],
                                    self.config[f"{self.config_prefix_UNSW}TSA_size"],
                                ))

                            # Query healthy
                            # NOTE: This is added to keep number of healthy samples in balance with
                            # the number of samples per faulty class.
                            for _ in range(
                                self.config[
                                    f"{self.config_prefix_ALL}num_installations"
                                ]
                            ):
                                for _ in range(
                                    self.config[f"{self.config_prefix_ALL}num_queries"]
                                ):
                                    # idx: (class, speed, load, severity, installation, healthy_GP, OT_method, TSA_size, sensor)
                                    group.append((
                                        "UNSW",
                                        "healthy",
                                        speed,
                                        load,
                                        healthy_query_severity,
                                        self.config[
                                            f"{self.config_prefix_UNSW}OT_method"
                                        ],
                                        self.config[
                                            f"{self.config_prefix_UNSW}TSA_size"
                                        ],
                                    ))

                        # FAULTY
                        else:
                            for severity in ["M", "L"]:
                                # idx: (class, speed, load, severity, OT_method, TSA_size)
                                for _ in range(
                                    self.config[f"{self.config_prefix_ALL}num_queries"]
                                ):
                                    group.append([
                                        "UNSW",
                                        class_,
                                        speed,
                                        load,
                                        severity,
                                        self.config[
                                            f"{self.config_prefix_UNSW}OT_method"
                                        ],
                                        self.config[
                                            f"{self.config_prefix_UNSW}TSA_size"
                                        ],
                                    ])
                    batch.extend(group)

                # Keep datasets separate
                if self.config["separate_dataset_batches"]:
                    yield batch
                    batch = []

            # AGFD samples
            ##

            if self.AGFD_included:
                for option in batch_options_AGFD:
                    group = []
                    speed, load, sensor = option
                    speed = int(speed)
                    load = int(load)

                    for class_ in self.config[f"{self.config_prefix_AGFD}classes"]:
                        # HEALTHY
                        ##

                        if class_ == "healthy":
                            # Get GPs for healthy anchors and samples
                            GPs = self.rng.choice(
                                self.config[f"{self.config_prefix_AGFD}healthy_GPs"],
                                size=self.config[f"{self.config_prefix_ALL}num_anchors"]
                                + self.config[
                                    f"{self.config_prefix_ALL}num_installations"
                                ],
                                # ! High number of anchors may require replacement to be true
                                replace=True,
                            )

                            # Anchor
                            for i in range(
                                self.config[f"{self.config_prefix_ALL}num_anchors"]
                            ):
                                # idx: (class, speed, load, severity, installation, healthy_GP, OT_method, TSA_size, sensor)
                                group.append((
                                    "AGFD",
                                    "healthy",
                                    speed,
                                    load,
                                    "-",
                                    0,
                                    GPs[i],
                                    self.config[f"{self.config_prefix_AGFD}OT_method"],
                                    self.config[f"{self.config_prefix_AGFD}TSA_size"],
                                    sensor,
                                ))

                            # Query healthy
                            # NOTE: This is added to keep number of healthy samples in balance with
                            # the number of samples per faulty class.
                            for i in range(
                                self.num_query_groups(self.config_prefix_AGFD, "healthy")
                            ):
                                for _ in range(
                                    self.config[f"{self.config_prefix_ALL}num_queries"]
                                ):
                                    group.append((
                                        "AGFD",
                                        "healthy",
                                        speed,
                                        load,
                                        "-",
                                        0,
                                        GPs[
                                            self.config[
                                                f"{self.config_prefix_ALL}num_anchors"
                                            ]
                                            + i
                                        ],
                                        self.config[
                                            f"{self.config_prefix_AGFD}OT_method"
                                        ],
                                        self.config[
                                            f"{self.config_prefix_AGFD}TSA_size"
                                        ],
                                        sensor,
                                    ))
                        # FAULTY
                        ##
                        else:
                            for installation in self.rng.choice(
                                self.config[f"{self.config_prefix_AGFD}installations"],
                                size=self.num_query_groups(
                                    self.config_prefix_AGFD, class_
                                ),
                                replace=False,
                            ):
                                # NOTE: Necessary, because no installation 3 for acc2
                                fixed_installation = installation
                                if sensor == "acc2" and installation == 3:
                                    fixed_installation = 2

                                # idx: (class, speed, load, severity, installation, healthy_GP, OT_method, TSA_size, sensor)
                                for _ in range(
                                    self.config[f"{self.config_prefix_ALL}num_queries"]
                                ):
                                    group.append([
                                        "AGFD",
                                        class_,
                                        speed,
                                        load,
                                        self.dataset.dataset_AGFD.fault_to_severity[
                                            class_
                                        ](),
                                        fixed_installation,
                                        0,
                                        self.config[
                                            f"{self.config_prefix_AGFD}OT_method"
                                        ],
                                        self.config[
                                            f"{self.config_prefix_AGFD}TSA_size"
                                        ],
                                        sensor,
                                    ])
                    batch.extend(group)

                # Keep datasets separate
                if self.config["separate_dataset_batches"]:
                    yield batch
                    batch = []

            # MCC5
            ##

            if self.MCC5_included:
                for option in batch_options_MCC5:
                    group = []
                    speed, load, sensor, circulation = option
                    speed = int(speed)
                    load = int(load)

                    for class_ in self.config[f"{self.config_prefix_MCC5}classes"]:
                        # HEALTHY
                        ##

                        if class_ == "healthy":
                            # Anchor
                            # NOTE: Middle speeds and loads do not have corresponding
                            # healthy for opposite circulation
                            if speed in [1500, 2500] or load in [5, 15]:
                                anchor_circulation = circulation
                            else:
                                anchor_circulation = (
                                    "speed" if circulation == "torque" else "torque"
                                )
                            for _ in range(
                                self.config[f"{self.config_prefix_ALL}num_anchors"]
                            ):
                                # idx: (class, speed, load, severity, installation, healthy_GP, OT_method, TSA_size, sensor)
                                group.append((
                                    "MCC5",
                                    "healthy",
                                    speed,
                                    load,
                                    "-",
                                    anchor_circulation,
                                    self.config[f"{self.config_prefix_MCC5}OT_method"],
                                    self.config[f"{self.config_prefix_MCC5}TSA_size"],
                                    sensor,
                                ))

                            # Query healthy
                            # NOTE: This is added to keep number of healthy samples in balance with
                            # the number of samples per faulty class.
                            for _ in range(
                                self.num_query_groups(self.config_prefix_MCC5, "healthy")
                            ):
                                for _ in range(
                                    self.config[f"{self.config_prefix_ALL}num_queries"]
                                ):
                                    # idx: (class, speed, load, severity, installation, healthy_GP, OT_method, TSA_size, sensor)
                                    group.append((
                                        "MCC5",
                                        "healthy",
                                        speed,
                                        load,
                                        "-",
                                        circulation,
                                        self.config[
                                            f"{self.config_prefix_MCC5}OT_method"
                                        ],
                                        self.config[
                                            f"{self.config_prefix_MCC5}TSA_size"
                                        ],
                                        sensor,
                                    ))

                        # FAULTY
                        else:
                            # Missing tooth has only one severity, so it's repeated
                            severities = MCC5_class_severities(
                                self.config, self.config_prefix_MCC5, class_
                            )
                            num_groups = self.num_query_groups(
                                self.config_prefix_MCC5, class_
                            )
                            for severity in self.rng.choice(
                                severities,
                                size=num_groups,
                                replace=len(severities) < num_groups,
                            ):
                                # idx: (class, speed, load, severity, circulation, OT_method, TSA_size, sensor)
                                for _ in range(
                                    self.config[f"{self.config_prefix_ALL}num_queries"]
                                ):
                                    group.append([
                                        "MCC5",
                                        class_,
                                        speed,
                                        load,
                                        severity,
                                        circulation,
                                        self.config[
                                            f"{self.config_prefix_MCC5}OT_method"
                                        ],
                                        self.config[
                                            f"{self.config_prefix_MCC5}TSA_size"
                                        ],
                                        sensor,
                                    ])
                    batch.extend(group)

                if self.config["separate_dataset_batches"]:
                    yield batch
                    batch = []

            if not self.config["separate_dataset_batches"]:
                yield batch
                batch = []


def get_ALL_data(
    split,
    rng,
    config,
    device,
    AGFD_included=True,
    MCC5_included=True,
    UNSW_included=False,
):
    """
    Create a dataloader from the included datasets (UNSW, MCC5, AGFD), using the
    `<dataset>_<split>_*` and `ALL_<split>_*` entries of the configuration.

    Parameters:
        split: str
            "train" or "validation".
        rng: np.random.Generator
            Random number generator.
        config: dict
            The chosen configuration dict from /configs.
        device: torch.device
            Not currently used for anything. Can be used to move all data to
            GPU at start in the future.

    Returns:
        data_loader: torch.Dataloader
    """

    batchPreprocessor = BatchPreprocessor(config, "preprocessing_batch", split, rng)
    afterDifferenceBatchPreprocessor = BatchPreprocessor(
        config, "preprocessing_difference_batch", split, rng
    )

    # Load data
    ###########

    log.debug("Reading data")
    log.debug("")

    config_prefix_AGFD = f"AGFD_{split}_"
    config_prefix_MCC5 = f"MCC5_{split}_"
    config_prefix_UNSW = f"UNSW_{split}_"

    if AGFD_included:
        assert config[f"{config_prefix_AGFD}TSA_size"] == 40, (
            "AGFD TSA size must be 40, please check your config."
        )
    if MCC5_included:
        assert config[f"{config_prefix_MCC5}TSA_size"] == 30, (
            "MCC5 TSA size must be 30, please check your config."
        )
    if UNSW_included:
        assert config[f"{config_prefix_UNSW}TSA_size"] == 40, (
            "UNSW TSA size must be 40, please check your config."
        )

    # AGFD
    ##

    if AGFD_included:
        df_AGFD = pd.read_feather(DATA_DIR / config["data_files"]["AGFD"])

        # Severe "crack" (Severe_TFF_GP5) is an entirely missing tooth
        missing_tooth = (df_AGFD["class"] == "crack") & (df_AGFD["severity"] == "L")
        df_AGFD.loc[missing_tooth, "class"] = "missing_tooth"

        df_AGFD = df_AGFD[
            df_AGFD["speed"].isin(config[f"{config_prefix_AGFD}speeds"])
            & (df_AGFD["load"].isin(config[f"{config_prefix_AGFD}loads"]))
            & (
                (
                    df_AGFD["installation"].isin(
                        config[f"{config_prefix_AGFD}installations"]
                    )
                )
                | (df_AGFD["installation"] == 0)
            )
            & (
                (df_AGFD["healthy_GP"].isin(config[f"{config_prefix_AGFD}healthy_GPs"]))
                | (df_AGFD["healthy_GP"] == 0)
            )
            & (
                (df_AGFD["severity"].isin(config[f"{config_prefix_AGFD}severities"]))
                | (df_AGFD["severity"] == "-")
            )
            & (df_AGFD["class"].isin(config[f"{config_prefix_AGFD}classes"]))
            & (df_AGFD["OT_method"] == config[f"{config_prefix_AGFD}OT_method"])
        ]

    # MCC5
    ##

    if MCC5_included:
        df_MCC5 = pd.read_feather(DATA_DIR / config["data_files"]["MCC5"])

        df_MCC5 = df_MCC5[
            df_MCC5["speed"].isin(config[f"{config_prefix_MCC5}speeds"])
            & (df_MCC5["load"].isin(config[f"{config_prefix_MCC5}loads"]))
            & (df_MCC5["class"].isin(config[f"{config_prefix_MCC5}classes"]))
            & (
                (df_MCC5["severity"] == "-")
                | pd.Series(
                    [
                        severity
                        in MCC5_class_severities(config, config_prefix_MCC5, class_)
                        for class_, severity in zip(
                            df_MCC5["class"], df_MCC5["severity"]
                        )
                    ],
                    index=df_MCC5.index,
                )
            )
            & (df_MCC5["OT_method"] == config[f"{config_prefix_MCC5}OT_method"])
        ]

    # UNSW
    ##
    if UNSW_included:
        df_UNSW = pd.read_feather(DATA_DIR / config["data_files"]["UNSW"])

        df_UNSW = df_UNSW[
            df_UNSW["speed"].isin(config[f"{config_prefix_UNSW}speeds"])
            & (df_UNSW["load"].isin(config[f"{config_prefix_UNSW}loads"]))
            & (df_UNSW["severity"].isin(config[f"{config_prefix_UNSW}severities"]))
            & (df_UNSW["OT_method"] == config[f"{config_prefix_UNSW}OT_method"])
        ]

    # Data formatting
    #################

    log.debug("")
    log.debug("Formatting data")

    # Dataset creation
    ##################

    # Individual datasets

    if AGFD_included:
        dataset_AGFD = AGFD_Dataset(
            df_AGFD,
            IndividualPreprocessor(config, "AGFD", split),
            rng,
            config_prefix_AGFD,
            config,
            device,
        )
    else:
        dataset_AGFD = None

    if MCC5_included:
        dataset_MCC5 = MCC5_Dataset(
            df_MCC5,
            IndividualPreprocessor(config, "MCC5", split),
            rng,
            config_prefix_MCC5,
            config,
            device,
        )
    else:
        dataset_MCC5 = None

    if UNSW_included:
        dataset_UNSW = UNSW_Dataset(
            df_UNSW,
            IndividualPreprocessor(config, "UNSW", split),
            rng,
            config_prefix_UNSW,
            config,
            device,
        )
    else:
        dataset_UNSW = None

    # Combined dataset

    dataset = ALL_Dataset(dataset_AGFD, dataset_MCC5, dataset_UNSW, rng, config, device)

    # DATALOADER
    #############

    data_loader = DataLoader(
        dataset,
        batch_sampler=ALL_BatchSampler(dataset, split, rng, config),
        collate_fn=partial(
            collate_all,
            batchPreprocessor=batchPreprocessor,
            afterDifferencePreprocessor=afterDifferenceBatchPreprocessor,
            split=split,
            config=config,
            device=device,
        ),
    )

    log.debug("")
    log.debug("Formatted data")

    return data_loader
