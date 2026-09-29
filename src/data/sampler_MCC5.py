import logging
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import BatchSampler, DataLoader, Dataset

from src.data.preprocessing_batch import BatchPreprocessor, collate_w_difference
from src.data.preprocessing_individual import IndividualPreprocessor

log = logging.getLogger("gear-cm")


class MCC5_Dataset(Dataset):
    NAME = "MCC5"

    def __init__(self, df, individualPreprocessor, rng, config_prefix, config, device):
        self.data = df
        self.individualPreprocessor = individualPreprocessor
        self.config_prefix = config_prefix
        self.config = config
        self.device = device  # * Here just in case, not currently used for anything
        self.rng = rng

        self.data = pd.melt(
            self.data,
            id_vars=[
                "speed",
                "load",
                "class",
                "severity",
                "circulation",
                "OT_method",
                "TSA_size",
            ],
            value_vars=self.config[f"{config_prefix}sensors"],
            var_name="sensor",
            value_name="signal",
        )

        # Perform preprocessing that is independent of the batch
        samples = self.individualPreprocessor.transform(
            np.array(self.data["signal"].tolist())
        )
        samples = [
            sample for sample in samples
        ]  # Convert 2D array to list of 1D arrays
        self.data["signal"] = samples

        # Group data by operating conditions and make into a dictionary
        self.data = (
            self.data.groupby(
                [
                    "class",
                    "speed",
                    "load",
                    "severity",
                    "circulation",
                    "OT_method",
                    "TSA_size",
                    "sensor",
                ]
            )["signal"]
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
                assert (
                    len(self.data[k]) >= self.config["n_query"]
                ), f"Not enough samples ({len(self.data[k])}) for the query set for dataset {self.config_prefix} idx {k}."

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


class MCC5_FS_Difference_BatchSampler(BatchSampler):
    def __init__(self, dataset, rng, config):
        self.dataset = dataset
        self.rng = rng
        self.config = config
        self.fault_classes = self.config[f"{self.dataset.config_prefix}classes"][
            1:
        ]  #! "healthy" must be first
        self.batch_num = 0  # Batch num
        self.prev_batch = None

        # 5 Nm and 15 Nm are special cases, so they can't be trusted to exist
        self.possible_loads_for_shift = [
            x
            for x in self.config[f"{self.dataset.config_prefix}loads"]
            if x == 10 or x == 20
        ]

        # SETUP OPERATING CONDITION SAMPLING
        ####################################

        # Get possible operating conditions
        operating_conditions = list(self.dataset.data.keys())
        # Key - (class, speed, load, severity, circulation, OT_method, TSA_size, sensor)
        # Drop healthy (reconstructed later) and keep only (i.e. drop class):
        # Keep - (speed, load, severity, circulation, OT_method, TSA_size, sensor)
        operating_conditions = [
            (k[1], k[2], k[3], k[4], k[5], k[6], k[7])
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
            assert (
                len(self.config[f"{self.dataset.config_prefix}severities"]) > 1
            ), f"MCC5_FS_Difference_BatchSampler requires at least 2 severities to do severity shift ({self.dataset.config_prefix})"

        if "load" in self.config[f"{self.dataset.config_prefix}shift_methods"]:
            assert (
                len(self.possible_loads_for_shift) >= 2
            ), f"MCC5_FS_Difference_BatchSampler requires both 10 Nm and 20 Nm loads to do load shift ({self.dataset.config_prefix})"

    def __iter__(self):
        while True:
            batch = []

            # Sample some base operating condition (operating condition order permuted in __init__)
            # base: (speed, load, severity, circulation, OT_method, TSA_size, sensor)
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

            # Base severity
            # (healthy severity always "-", so no need for explicit anchor severity)
            support_severity = base[2]
            query_severity = base[2]

            anchor_circulation = base[3]
            healthy_sample_circulation = base[3]

            # Load shifting
            ##

            if shift_method == "load" or shift_method == "load_and_severity":
                # Sample random loads for shift
                query_load = self.rng.choice(
                    # Remove current load from the list
                    list(set(self.possible_loads_for_shift) - set([base[1]])),
                    size=1,
                    # Needs to be true, because if support is 10 or 20, only one choice left,
                    # because 5 and 15 cannot be trusted to exist for all speeds
                    replace=True,
                )[0]

            # Severity shifting
            ##

            if shift_method == "severity" or shift_method == "load_and_severity":
                # Sample random severity for query
                # Support severity is always the same as base and anchor not needed
                query_severity = self.rng.choice(
                    list(
                        set(self.config[f"{self.dataset.config_prefix}severities"])
                        - set([base[2]])
                    ),
                    size=1,
                    replace=False,
                )[0]

                # Flip circulation for healthy query to get at least some difference
                # between healthy support and query.
                # NOTE: Anchor and healthy support are the same, because it kind of makes
                # sense that the baseline samples would be close to each other than
                # query samples.
                anchor_circulation = (
                    "speed" if healthy_sample_circulation == "torque" else "torque"
                )

            if shift_method not in [
                "identity",
                "severity",
                "load",
                "load_and_severity",
            ]:
                raise ValueError(f"Invalid shift method: {shift_method}")

            # BUILD BATCH
            #############

            # Healthy
            ##

            # Healthy anchor
            # NOTE: Only use if difference is used
            if self.config["use_difference"]:
                batch.append(
                    (
                        "healthy",
                        base[0],
                        support_load,
                        "-",
                        anchor_circulation,
                        base[4],
                        base[5],
                        base[6],
                    )
                )

                # If load doesn't change, only one anchor is needed
                if shift_method == "load" or shift_method == "load_and_severity":
                    batch.append(
                        (
                            "healthy",
                            base[0],
                            query_load,
                            "-",
                            anchor_circulation,
                            base[4],
                            base[5],
                            base[6],
                        )
                    )

            # Healthy support
            batch.extend(
                [
                    (
                        "healthy",
                        base[0],
                        support_load,
                        "-",
                        healthy_sample_circulation,
                        base[4],
                        base[5],
                        base[6],
                    )
                    for _ in range(self.config["k_shot"])
                ]
            )

            # Healthy query
            batch.extend(
                [
                    (
                        "healthy",
                        base[0],
                        query_load,
                        "-",
                        healthy_sample_circulation,
                        base[4],
                        base[5],
                        base[6],
                    )
                    for _ in range(self.config["n_query"])
                ]
            )

            # OTHER CLASSES
            ##

            # idx: (class, speed, load, severity, OT_method, TSA_size, sensor)
            # base: (speed, load, severity, OT_method, TSA_sizem, sensor)

            for fault_class in self.fault_classes:
                # Faulty support
                ##

                support_key = (
                    fault_class,
                    base[0],
                    support_load,
                    support_severity,
                    base[3],
                    base[4],
                    base[5],
                    base[6],
                )
                # Pitting is missing a few measurements for M severity, so use L
                if support_key not in self.dataset.data.keys():
                    support_key = (
                        fault_class,
                        base[0],
                        support_load,
                        "L",
                        base[3],
                        base[4],
                        base[5],
                        base[6],
                    )

                batch.extend([support_key for _ in range(self.config["k_shot"])])

                # Faulty query
                ##

                query_key = (
                    fault_class,
                    base[0],
                    query_load,
                    query_severity,
                    base[3],
                    base[4],
                    base[5],
                    base[6],
                )
                # Pitting is missing a few measurements for M severity, so use L
                if query_key not in self.dataset.data.keys():
                    query_key = (
                        fault_class,
                        base[0],
                        query_load,
                        "L",
                        base[3],
                        base[4],
                        base[5],
                        base[6],
                    )

                batch.extend([query_key for _ in range(self.config["n_query"])])

            # for b in batch:
            #     print(b)
            # # quit()
            # print()

            self.prev_batch = batch
            yield batch


def get_MCC5_data(split, rng, config, device):
    """
    Create a dataloader from MCC5 according to the given configuration.

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

    config_prefix = f"MCC5_{split}_"

    individualPreprocessor = IndividualPreprocessor(config, "MCC5", split)
    batchPreprocessor = BatchPreprocessor(config, "preprocessing_batch", split, rng)
    afterDifferenceBatchPreprocessor = BatchPreprocessor(
        config, "preprocessing_difference_batch", split, rng
    )

    # Load data
    ###########

    log.debug(f"Reading {config_prefix} data")
    log.debug(f"")

    # data_path = (
    #     Path(__file__).resolve().parent.parent.parent / "data" / "MCC5-THU_OT.feather"
    # )
    # data = pd.read_feather(data_path)

    dfs = []
    # Hardcoded because of file names
    if 30 in config[f"{config_prefix}TSA_sizes"]:
        data_path = (
            Path(__file__).resolve().parent.parent.parent
            / "data"
            / "MCC5-THU_OT.feather"
        )
        dfs.append(pd.read_feather(data_path))
    if 15 in config[f"{config_prefix}TSA_sizes"]:
        data_path = (
            Path(__file__).resolve().parent.parent.parent
            / "data"
            / "MCC5-THU_OT_TSA_15.feather"
        )
        dfs.append(pd.read_feather(data_path))
    data = pd.concat(dfs, ignore_index=True)

    data = data[
        data["speed"].isin(config[f"{config_prefix}speeds"])
        & (data["load"].isin(config[f"{config_prefix}loads"]))
        & (data["class"].isin(config[f"{config_prefix}classes"]))
        & (
            (data["severity"].isin(config[f"{config_prefix}severities"]))
            | (data["severity"] == "-")
        )
        & (data["OT_method"].isin(config[f"{config_prefix}OT_methods"]))
    ]

    # Data formatting
    #################

    log.debug("")
    log.debug(f"Formatting {config_prefix} data")

    # Dataset creation
    ##################

    dataset = MCC5_Dataset(
        data, individualPreprocessor, rng, config_prefix, config, device
    )

    # DATALOADER
    #############

    data_loader = DataLoader(
        dataset,
        batch_sampler=MCC5_FS_Difference_BatchSampler(dataset, rng, config),
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
