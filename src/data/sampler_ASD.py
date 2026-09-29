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


class ASD_Dataset(Dataset):
    NAME = "ASD"
    class_map = {
        "healthy": 0,
        "1-shim": 1,
        "2-shim": 2,
        "3-shim": 3,
    }

    def __init__(self, df, individualPreprocessor, rng, config_prefix, config, device):
        self.data = df
        self.individualPreprocessor = individualPreprocessor
        self.config_prefix = config_prefix
        self.config = config
        self.device = device  # * Here just in case, not currently used for anything
        self.rng = rng

        # Pick the right sensors and separate them
        self.data = pd.melt(
            self.data,
            id_vars=[
                "speed",
                "load",
                "healthy_GP",
                "class",
                "severity",
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

        # Group data by operating conditions and make each group a list in a dict
        self.data = (
            self.data.groupby(
                [
                    "class",
                    "speed",
                    "load",
                    "severity",
                    "healthy_GP",
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
                ), f"Not enough samples for the query set for dataset {self.config_prefix} idx {k}."

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
        label = self.class_map[idx[0]]

        return sample, label, idx
