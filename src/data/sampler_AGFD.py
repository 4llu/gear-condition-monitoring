import itertools
import logging
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import BatchSampler, DataLoader, Dataset

from src.data.preprocessing_batch import BatchPreprocessor, collate_w_difference
from src.data.preprocessing_individual import IndividualPreprocessor

log = logging.getLogger("gear-cm")


class AGFD_Dataset(Dataset):
    NAME = "AGFD"

    def __init__(self, df, individualPreprocessor, rng, config_prefix, config, device):
        self.data = df
        self.individualPreprocessor = individualPreprocessor
        self.config_prefix = config_prefix
        self.config = config
        self.device = device  # * Here just in case, not currently used for anything
        self.rng = rng

        self.fault_to_severity = {
            "healthy": lambda: "-",
            # "wear": lambda: "L",
            # "crack": lambda: "S",
            "wear": lambda: (
                np.random.choice(["L", "S"])
                if self.config[f"{config_prefix}include_small_wear"]
                else "L"
            ),
            "crack": lambda: (
                np.random.choice(["L", "S"])
                if self.config[f"{config_prefix}include_large_crack"]
                else "S"
            ),
            "pitting": lambda: "L",
            "missing_tooth": lambda: "L",
        }

        # NOTE: "L" severity cracks are actually missing teeth, so we rename them
        # self.data.loc[
        #     (self.data["class"] == "crack") & (self.data["severity"] == "L"), "class"
        # ] = "missing_tooth"

        # Pick the right sensors and separate them
        self.data = pd.melt(
            self.data,
            id_vars=[
                "speed",
                "load",
                "installation",
                "healthy_GP",
                "class",
                "severity",
                "OT_method",
                "TSA_size",
            ],
            value_vars=self.config[f"{config_prefix}sensors"],
            var_name="sensor",
            value_name="signal",
        )

        # ! Remove installation 3 for acc2, as the sensor is broken
        self.data = self.data[
            ~((self.data["installation"] == 3) & (self.data["sensor"] == "acc2"))
        ]

        # Perform preprocessing that is independent of the batch
        samples = self.individualPreprocessor.transform(
            np.array(self.data["signal"].tolist())
        )
        samples = [
            sample for sample in samples
        ]  # Convert 2D array to list of 1D arrays
        self.data["signal"] = samples

        # Group data by operating conditions and make each group a list in a dict
        self.data = (
            self.data
            .groupby([
                "class",
                "speed",
                "load",
                "severity",
                "installation",
                "healthy_GP",
                "OT_method",
                "TSA_size",
                "sensor",
            ])["signal"]
            .apply(list)
            .to_dict()
        )

        # Get the number of operating conditions for each class
        for fault_class in self.config[f"{config_prefix}classes"]:
            num_operating_conditions = sum(
                1 for key in self.data.keys() if key[0] == fault_class
            )
            num_samples = sum(
                len(self.data[key]) for key in self.data.keys() if key[0] == fault_class
            )

            log.debug(
                f"{self.config_prefix} Num {fault_class} operating conditions {num_operating_conditions} and samples {num_samples}"
            )

        # Skip for non-episodic batches
        if self.config["episodic_training"]:
            for k in self.data.keys():
                assert len(self.data[k]) >= self.config["n_query"], (
                    f"Not enough samples for the query set for dataset {self.config_prefix} idx {k}."
                )

    def __len__(self):
        return len(self.data.keys())  # ! FAIRLY MEANINGLESS!!!

    def __getitem__(self, idx):
        """
        Get a random sample and its label from the given operating condition.

        Parameters:
            idx : tuple(string, int, int, int, int, int, string, string)
                Tuple of class, speed, load, severity, installation, healthy_GP, OT_method, TSA_size, and sensor.

        Args:
            sample : np.array(np.float32)
                Random sample from the given operating condition.
            label : int
                Label of the sample.
        """

        # TODO: Include speed and torque average in the sample

        possible_samples = self.data[idx]
        sample = possible_samples[self.rng.integers(len(possible_samples))]
        label = self.config["class_map"][idx[0]]

        return sample, label, idx


class AGFD_Classical_BatchSampler(BatchSampler):
    def __init__(self, dataset, rng, config):
        self.dataset = dataset
        self.rng = rng
        self.config = config
        self.batch_num = 0

        self.condition_options = list(
            itertools.product(
                self.config[f"{self.dataset.config_prefix}speeds"],
                self.config[f"{self.dataset.config_prefix}loads"],
                self.config[f"{self.dataset.config_prefix}installations"],
                self.config[f"{self.dataset.config_prefix}OT_methods"],  # 1
                self.config[f"{self.dataset.config_prefix}TSA_sizes"],  # 1
                self.config[f"{self.dataset.config_prefix}sensors"],  # 1
            )
        )

        log.debug(f"AGFD condition options: {len(self.condition_options)}")

    def __iter__(self):
        while True:
            batch = []

            batch_options = self.rng.choice(
                self.condition_options,
                size=self.config["AGFD_train_num_options"],
                replace=False,
            )

            for option in batch_options:
                for class_ in self.config["AGFD_train_classes"]:
                    # HEALTHY
                    ##

                    if class_ == "healthy":
                        GP = self.rng.choice(
                            self.config["AGFD_train_healthy_GPs"],
                            size=1,
                            replace=False,
                        )[0]

                        for _ in range(self.config["AGFD_train_num_queries"]):
                            batch.append((
                                class_,
                                int(option[0]),  # speed
                                int(option[1]),  # load
                                "-",  # severity
                                0,  # installation
                                GP,  # healthy_GP
                                option[3],  # OT_method
                                int(option[4]),  # TSA_size
                                option[5],  # sensor
                            ))
                    # FAULTY
                    ##
                    else:
                        for _ in range(self.config["AGFD_train_num_queries"]):
                            batch.append((
                                class_,
                                int(option[0]),  # speed
                                int(option[1]),  # load
                                self.dataset.fault_to_severity[class_](),  # severity
                                int(option[2]),  # installation
                                0,  # healthy_GP
                                option[3],  # OT_method
                                int(option[4]),  # TSA_size
                                option[5],  # sensor
                            ))

            # for b in batch:
            #     print(b)
            # quit()
            yield batch
            batch = []


class AGFD_FS_Difference_BatchSampler(BatchSampler):
    def __init__(self, dataset, rng, config):
        self.dataset = dataset
        self.rng = rng
        self.config = config
        self.fault_classes = self.config[f"{self.dataset.config_prefix}classes"][
            1:
        ]  #! "healthy" must be first
        self.batch_num = 0  # Batch num
        self.prev_batch = None

        # SETUP OPERATING CONDITION SAMPLING
        ####################################

        # Get possible operating conditions
        operating_conditions = list(self.dataset.data.keys())
        # Drop healthy (reconstructed later) and keep only:
        # Key - (class, speed, load, severity, installation, healthy_GP, OT_method, TSA_size, sensor)
        # Keep - (speed, load, installation, OT_method, TSA_size, sensor)
        operating_conditions = [
            (k[1], k[2], k[4], k[6], k[7], k[8])
            for k in self.dataset.data.keys()
            if k[0] != "healthy"
        ]
        # Keep only unique
        self.operating_conditions = list(set(operating_conditions))
        # Get permutated order
        self.operating_conditions_order = self.rng.permutation(
            len(self.operating_conditions)
        )

        # CHECKS
        ########

        if "installation" in self.config[f"{self.dataset.config_prefix}shift_methods"]:
            assert len(self.config[f"{self.dataset.config_prefix}installations"]) > 1, (
                f"AGFD_FS_Difference_BatchSampler requires at least 2 installation to do installation shift ({self.dataset.config_prefix})"
            )

        if "load" in self.config[f"{self.dataset.config_prefix}shift_methods"]:
            assert len(self.config[f"{self.dataset.config_prefix}loads"]) > 2, (
                f"AGFD_FS_Difference_BatchSampler requires at least 3 loads to do load shift ({self.dataset.config_prefix})"
            )

    def __iter__(self):
        while True:
            batch = []

            # Sample some base operating condition (operating condition order permuted in __init__)
            # base: (speed, load, OT_method, TSA_size, sensor)
            base = self.operating_conditions[
                self.operating_conditions_order[
                    self.batch_num % len(self.operating_conditions)
                ]
            ]
            self.batch_num += 1

            # SHIFTS
            ########

            # Sample shift method(s)
            # TODO: Multiple simultaneous shifts
            # TODO: Add speed shifts
            shift_method = self.rng.choice(
                self.config[f"{self.dataset.config_prefix}shift_methods"], size=1
            )[0]

            # Base loads
            support_load = base[1]
            query_load = base[1]

            # Base installation healthy severity
            # always 0, so no need for explicit anchor installation
            support_installation = base[2]
            query_installation = base[2]

            # Base healthy GP
            # NOTE: Enough sampled for installation/GP shift, even if not used
            GPs = self.rng.choice(
                self.config[f"{self.dataset.config_prefix}healthy_GPs"],
                size=1 if "installation" not in shift_method else 3,
                replace=False,
            )

            anchor_GP = GPs[0]
            support_GP = GPs[0]
            query_GP = GPs[0]

            # Load shifting
            ##

            if shift_method == "load" or shift_method == "load_and_installation":
                # Sample random load for query

                query_load = self.rng.choice(
                    # Remove current load from the list
                    list(
                        set(self.config[f"{self.dataset.config_prefix}loads"])
                        - set([base[1]])
                    ),
                    size=1,
                    replace=False,
                )[0]

            # Installation/healthy_GP shifting

            if (
                shift_method == "installation"
                or shift_method == "load_and_installation"
            ):
                # Remove base installation from the list
                available_installations = set(
                    self.config[f"{self.dataset.config_prefix}installations"]
                ) - set([base[2]])

                # Remove installation 3 if using acc2
                # NOTE: This is because acc3 for installation 3 is broken
                if base[5] == "acc2":
                    available_installations = available_installations - set([3])
                available_installations = list(available_installations)

                # Sample random installation for faulty query
                # Support installation is always the same as base
                query_installation = self.rng.choice(
                    available_installations,
                    size=1,
                    replace=False,
                )[0]

                # Sample random healthy GPs (support GP doesn't change)
                anchor_GP = GPs[1]
                query_GP = GPs[2]

            if shift_method not in [
                "identity",
                "installation",
                "load",
                "load_and_installation",
            ]:
                raise ValueError(f"Invalid shift method: {shift_method}")

            # BUILD BATCH
            #############

            # HEALTHY
            ##

            # idx: (class, speed, load, severity, installation, healthy_GP, OT_method, TSA_size, sensor)
            # base: (speed, load, installation, OT_method, TSA_size, sensor)

            # Healthy anchor
            # NOTE: Only use if difference is used
            if self.config["use_difference"]:
                batch.append((
                    "healthy",
                    base[0],
                    support_load,
                    "-",  # Severity is always "-" for healthy
                    0,  # Installation is always 0 for healthy
                    anchor_GP,
                    base[3],
                    base[4],
                    base[5],
                ))
                # If load doesn't change, only one anchor is needed
                if shift_method == "load" or shift_method == "load_and_installation":
                    batch.append((
                        "healthy",
                        base[0],
                        query_load,
                        "-",  # Severity is always "-" for healthy
                        0,  # Installation is always 0 for healthy
                        anchor_GP,
                        base[3],
                        base[4],
                        base[5],
                    ))

            # Healthy support
            batch.extend([
                (
                    "healthy",
                    base[0],
                    support_load,  # Base load
                    "-",  # Severity is always "-" for healthy
                    0,  # Installation is always 0 for healthy
                    support_GP,
                    base[3],
                    base[4],
                    base[5],
                )
                for _ in range(self.config["k_shot"])
            ])

            # Healthy query
            batch.extend([
                (
                    "healthy",
                    base[0],
                    query_load,
                    "-",  # Severity is always "-" for healthy
                    0,  # Installation is always 0 for healthy
                    query_GP,
                    base[3],
                    base[4],
                    base[5],
                )
                for _ in range(self.config["n_query"])
            ])

            # OTHER CLASSES
            ##

            # idx: (class, speed, load, severity, installation, healthy_GP, OT_method, TSA_size, sensor)
            # base: (speed, load, installation, OT_method, TSA_size, sensor)

            for fault_class in self.fault_classes:
                # Faulty support
                batch.extend([
                    (
                        fault_class,
                        base[0],
                        base[1],
                        self.dataset.fault_to_severity[
                            fault_class
                        ](),  # FIXME: Only using severe wear and mild TFF!!!
                        support_installation,
                        0,  # No healthy GP for faulty classes
                        base[3],
                        base[4],
                        base[5],
                    )
                    for _ in range(self.config["k_shot"])
                ])

                # Faulty query
                batch.extend([
                    (
                        fault_class,
                        base[0],
                        query_load,
                        self.dataset.fault_to_severity[
                            fault_class
                        ](),  # FIXME: Only using severe wear and mild TFF!!!,
                        query_installation,
                        0,  # No healthy GP for faulty classes
                        base[3],
                        base[4],
                        base[5],
                    )
                    for _ in range(self.config["n_query"])
                ])

            # for b in batch:
            #     print(b)
            # # quit()
            # print()

            self.prev_batch = batch
            yield batch

    def __len__(self):
        # ! FAIRLY MEANINGLESS!!!
        return len(self.operating_condition_cycle)


def get_AGFD_data(split, rng, config, device):
    """
    Create a dataloader from AGFD according to the given configuration.

    Parameters:
        split: str
            "train", "validation", or "test".
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

    config_prefix = f"AGFD_{split}_"

    individualPreprocessor = IndividualPreprocessor(config, "AGFD", split)
    batchPreprocessor = BatchPreprocessor(config, "preprocessing_batch", split, rng)
    afterDifferenceBatchPreprocessor = BatchPreprocessor(
        config, "preprocessing_difference_batch", split, rng
    )

    # Load data
    ###########

    log.debug("Reading {config_prefix} data")
    log.debug("")

    # dfs = []
    data = None
    # Hardcoded because of file names
    if config[f"{config_prefix}TSA_size"] == 40:
        data_path = (
            Path(__file__).resolve().parent.parent.parent / "data" / "AGFD_OT.feather"
        )
        # dfs.append(pd.read_feather(data_path))
        data = pd.read_feather(
            Path(__file__).resolve().parent.parent.parent
            / "data"
            / "AGFD_OT_V2.feather"
        )
    if config[f"{config_prefix}TSA_size"] == 15:
        raise Exception("Probably shouldn't use this right now!")
        data_path = (
            Path(__file__).resolve().parent.parent.parent
            / "data"
            / "AGFD_OT_TSA_15.feather"
            # / "AGFD_OT_V2_TSA_15.feather"
        )
        dfs.append(pd.read_feather(data_path))

    # data = pd.concat(dfs, ignore_index=True)

    data = data[
        data["speed"].isin(config[f"{config_prefix}speeds"])
        & (data["load"].isin(config[f"{config_prefix}loads"]))
        & (
            (data["installation"].isin(config[f"{config_prefix}installations"]))
            | (data["installation"] == 0)
        )
        & (
            (data["healthy_GP"].isin(config[f"{config_prefix}healthy_GPs"]))
            | (data["healthy_GP"] == 0)
        )
        & (
            (data["severity"].isin(config[f"{config_prefix}severities"]))
            | (data["severity"] == "-")
        )
        & (data["class"].isin(config[f"{config_prefix}classes"]))
        & (data["OT_method"] == config[f"{config_prefix}OT_method"])
    ]

    # Data formatting
    #################

    log.debug("")
    log.debug(f"Formatting {config_prefix} data")

    # Dataset creation
    ##################

    dataset = AGFD_Dataset(
        data, individualPreprocessor, rng, config_prefix, config, device
    )

    if config["model"] == "classical" and split == "train":
        batch_sampler = AGFD_Classical_BatchSampler(dataset, rng, config)
    else:
        batch_sampler = AGFD_FS_Difference_BatchSampler(dataset, rng, config)

    # DATALOADER
    #############

    data_loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=partial(
            collate_w_difference,
            batchPreprocessor=batchPreprocessor,
            afterDifferencePreprocessor=afterDifferenceBatchPreprocessor,
            config_prefix=config_prefix,
            config=config,
            device=device,
        ),
        # pin_memory=True,
        # num_workers=0,
        # prefetch_factor=4,
    )

    log.debug("")
    log.debug(f"Formatted {config_prefix} data")

    return data_loader
