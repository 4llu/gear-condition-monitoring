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


class UNSW_Dataset(Dataset):
    NAME = "UNSW"

    def __init__(self, df, individualPreprocessor, rng, config_prefix, config, device):
        self.data = df
        self.individualPreprocessor = individualPreprocessor
        self.config_prefix = config_prefix
        self.config = config
        self.device = device  # * Here just in case, not currently used for anything
        self.rng = rng

        self.severity_replacement_map = {
            "H": "H1",
            "S": "H2",
            "M": "M",
            "L": "L",
        }

        self.severity_map = {
            "H1": "healthy",
            "H2": "healthy",
            # "S": "crack",
            "M": "crack",
            "L": "crack",
        }

        # Remap severity labels
        self.data["severity"] = self.data["severity"].map(self.severity_replacement_map)
        # Add class column
        self.data["class"] = self.data["severity"].map(self.severity_map)

        ##
        # Signal manipulation
        ##

        # Perform preprocessing that is independent of the batch
        samples = self.individualPreprocessor.transform(
            np.array(self.data["signal"].to_list())
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
                "OT_method",
                "TSA_size",
            ])["signal"]
            .apply(list)
            .to_dict()
        )

        # Get the number of healthy and faulty samples
        # num_healthy_samples = sum(1 for key in self.data.keys() if key[0] == "healthy")
        # num_faulty_samples = len(self.data) - num_healthy_samples
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

        # log.debug(f"{self.config_prefix} Num healthy keys {num_healthy_samples}")
        # log.debug(f"{self.config_prefix} Num faulty keys {num_faulty_samples}")
        for k in self.data.keys():
            assert len(self.data[k]) >= self.config[f"n_query"], (
                f"Not enough samples for the query set for dataset {self.config_prefix} idx {k}."
            )

    def __len__(self):
        return len(self.data.keys())  # ! FAIRLY MEANINGLESS!!!

    def __getitem__(self, idx):
        """
        Get a random sample and its label from the given operating condition.

        Args:
            idx : tuple(string, int, int, string, string, int)
                Tuple of class, speed, load, severity, OT_method, and TSA_size.

        Returns:
            sample : np.array(np.float32)
                Random sample from the given operating condition.
            label : int
                Label of the sample.
        """

        # TODO Include speed and torque average in the sample

        # print(self.data.keys())

        possible_samples = self.data[idx]
        sample = possible_samples[self.rng.integers(len(possible_samples))]
        label = self.config["class_map"][idx[0]]

        return sample, label, idx


class UNSW_Classical_BatchSampler(BatchSampler):
    def __init__(self, dataset, rng, config):
        self.dataset = dataset
        self.rng = rng
        self.config = config
        self.batch_num = 0

        assert len(
            self.config[f"{self.dataset.config_prefix}severities"]
        ) == 2 and self.config[f"{self.dataset.config_prefix}severities"][0] in [
            "H",
            "S",
        ], (
            "UNSW_Classical_BatchSampler requires exactly 2 severities, the first of which must be 'H' or 'S'."
        )

        self.condition_options = list(
            itertools.product(
                self.config[f"{self.dataset.config_prefix}speeds"],
                self.config[f"{self.dataset.config_prefix}loads"],
                self.config[f"{self.dataset.config_prefix}OT_methods"],  # 1
                self.config[f"{self.dataset.config_prefix}TSA_sizes"],  # 1
            )
        )

        log.debug(f"UNSW condition options: {len(self.condition_options)}")

    def __iter__(self):
        while True:
            batch = []

            batch_options = self.rng.choice(
                self.condition_options,
                size=self.config["UNSW_train_num_options"],
                replace=False,
            )

            for option in batch_options:
                for class_ in self.config["UNSW_train_classes"]:
                    # HEALTHY
                    ##

                    if class_ == "healthy":
                        for _ in range(self.config[f"UNSW_train_num_queries"]):
                            batch.append((
                                class_,
                                int(option[0]),  # speed
                                int(option[1]),  # load
                                self.dataset.severity_replacement_map[
                                    self.config[
                                        f"{self.dataset.config_prefix}severities"
                                    ][0]
                                ],  # severity # ! Hack
                                option[2],  # OT_method
                                int(option[3]),  # TSA_size
                            ))
                    # FAULTY
                    ##
                    else:
                        for _ in range(self.config[f"UNSW_train_num_queries"]):
                            batch.append((
                                class_,
                                int(option[0]),  # speed
                                int(option[1]),  # load
                                self.dataset.severity_replacement_map[
                                    self.config[
                                        f"{self.dataset.config_prefix}severities"
                                    ][1]
                                ],  # severity # ! Hack
                                option[2],  # OT_method
                                int(option[3]),  # TSA_size
                            ))

            # for b in batch:
            #     print(b)
            # quit()
            yield batch
            batch = []


class UNSW_FS_Difference_BatchSampler(BatchSampler):
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
        # Key - (class, speed, load, severity, OT_method, TSA_size)
        # Drop healthy (reconstructed later) and keep only (drop class):
        # Keep - (speed, load, severity, OT_method, TSA_size)
        operating_conditions = [
            (k[1], k[2], k[3], k[4], k[5])
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

        if "severity" in self.config[f"{self.dataset.config_prefix}shift_methods"]:
            assert len(self.config[f"{self.dataset.config_prefix}severities"]) > 1, (
                f"UNSW_FS_Difference_BatchSampler requires at least 2 severities to do severity shift ({self.dataset.config_prefix})"
            )

        if "load" in self.config[f"{self.dataset.config_prefix}shift_methods"]:
            assert len(self.config[f"{self.dataset.config_prefix}loads"]) > 2, (
                f"UNSW_FS_Difference_BatchSampler requires at least 3 loads to do load shift ({self.dataset.config_prefix})"
            )

    def __iter__(self):
        while True:
            batch = []

            # Sample some base operating condition
            # (operating condition order permuted in __init__)
            # base: (speed, load, severity, OT_method, TSA_size)
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

            # Base loads (support never changes)
            support_load = base[1]
            query_load = base[1]

            # Base severity (support never changes and healthy severity
            # always "-", so no need for explicit anchor severity)
            support_severity = base[2]
            query_severity = base[2]

            healthy_severities = []
            if "H" in self.config[f"{self.dataset.config_prefix}severities"]:
                healthy_severities.append("H1")
            if "S" in self.config[f"{self.dataset.config_prefix}severities"]:
                healthy_severities.append("H2")
            healthy_sample_severity = self.rng.choice(healthy_severities, size=1)[0]
            anchor_severity = healthy_sample_severity

            # Load shifting
            ##

            if shift_method == "load" or shift_method == "load_and_severity":
                # Sample random loads for shift
                query_load = self.rng.choice(
                    # Remove current load from the list
                    list(
                        set(self.config[f"{self.dataset.config_prefix}loads"])
                        - set([base[1]])
                    ),
                    size=1,
                    replace=False,
                )[0]

            # Severity shifting
            ##

            if shift_method == "severity" or shift_method == "load_and_severity":
                # Sample random severity for query
                # query_severity = self.rng.choice(
                #     list(
                #         set(self.config[f"{self.dataset.config_prefix}severities"])
                #         - set([base[2]])
                #     ),
                #     size=1,
                #     replace=False,
                # )[0]
                query_severity = "M" if support_severity == "L" else "L"

                # Different severity for anchor and healthy sample
                anchor_severity = "H1" if healthy_sample_severity == "H2" else "H2"

            if shift_method not in [
                "identity",
                "severity",
                "load",
                "load_and_severity",
            ]:
                raise ValueError(f"Invalid shift method: {shift_method}")

            # BUILD BATCH
            #############

            # HEALTHY
            ##

            # idx: (class, speed, load, severity, OT_method, TSA_size)
            # base: (speed, load, severity, OT_method, TSA_size)

            # Healthy anchor
            # NOTE: Only use if difference is used
            if self.config["use_difference"]:
                batch.append((
                    "healthy",
                    base[0],
                    support_load,
                    anchor_severity,
                    base[3],
                    base[4],
                ))

                # If load doesn't change, only one anchor is needed
                if shift_method == "load" or shift_method == "load_and_severity":
                    # Healthy anchor with different severity
                    batch.append((
                        "healthy",
                        base[0],
                        query_load,
                        healthy_sample_severity,
                        base[3],
                        base[4],
                    ))

            # Healthy support
            batch.extend([
                (
                    "healthy",
                    base[0],
                    support_load,
                    healthy_sample_severity,
                    base[3],
                    base[4],
                )
                for _ in range(self.config["k_shot"])
            ])

            # Healthy query
            batch.extend([
                (
                    "healthy",
                    base[0],
                    query_load,
                    healthy_sample_severity,
                    base[3],
                    base[4],
                )
                for _ in range(self.config["n_query"])
            ])

            # OTHER CLASSES
            ##
            # NOTE: Just crack for this dataset (UNSW)

            # idx: (class, speed, load, severity, OT_method, TSA_size)
            # base: (speed, load, severity, OT_method, TSA_size)

            for fault_class in self.fault_classes:
                # Faulty support
                batch.extend([
                    (
                        fault_class,
                        base[0],
                        support_load,
                        support_severity,
                        base[3],
                        base[4],
                    )
                    for _ in range(self.config["k_shot"])
                ])

                # Faulty query
                batch.extend([
                    (
                        fault_class,
                        base[0],
                        query_load,
                        query_severity,
                        base[3],
                        base[4],
                    )
                    for _ in range(self.config["n_query"])
                ])

            # for b in batch:
            #     print(b)
            # quit()
            # print()

            self.prev_batch = batch
            yield batch

    def __len__(self):
        # ! FAIRLY MEANINGLESS!!!
        return len(self.operating_condition_cycle)


def get_UNSW_data(split, rng, config, device):
    """
    Create a dataloader from UNSW according to the given configuration.

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

    config_prefix = f"UNSW_{split}_"

    individualPreprocessor = IndividualPreprocessor(config, "UNSW", split)
    batchPreprocessor = BatchPreprocessor(config, "preprocessing_batch", split, rng)
    afterDifferenceBatchPreprocessor = BatchPreprocessor(
        config, "preprocessing_difference_batch", split, rng
    )

    # Load data
    ###########

    log.debug(f"Reading {config_prefix} data")
    log.debug(f"")

    dfs = []
    if 40 in config[f"{config_prefix}TSA_sizes"]:
        data_path = (
            Path(__file__).resolve().parent.parent.parent
            / "data"
            / "UNSW_gear_crack_OT.feather"
            # / "UNSW_gear_crack_OT_TSA_FFT_V2.feather"
        )
        dfs.append(pd.read_feather(data_path))
    if 15 in config[f"{config_prefix}TSA_sizes"]:
        data_path = (
            Path(__file__).resolve().parent.parent.parent
            / "data"
            / "UNSW_gear_crack_OT_TSA_15.feather"
            # / "UNSW_gear_crack_OT_TSA_FFT_V2_TSA_15.feather"
        )
        dfs.append(pd.read_feather(data_path))

    data = pd.concat(dfs, ignore_index=True)

    data = data[
        data["speed"].isin(config[f"{config_prefix}speeds"])
        & (data["load"].isin(config[f"{config_prefix}loads"]))
        & (
            (data["severity"].isin(config[f"{config_prefix}severities"]))
            | (data["severity"] == "H")
        )
        & (data["OT_method"].isin(config[f"{config_prefix}OT_methods"]))
    ]

    # Data formatting
    #################

    log.debug("")
    log.debug(f"Formatting {config_prefix} data")

    dataset = UNSW_Dataset(
        data, individualPreprocessor, rng, config_prefix, config, device
    )

    if config["model"] == "classical" and split == "train":
        batch_sampler = UNSW_Classical_BatchSampler(dataset, rng, config)
    else:
        batch_sampler = UNSW_FS_Difference_BatchSampler(dataset, rng, config)
    # DATALOADER
    ############

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
