"""MCC5 dataset: samples grouped by operating condition, used by sampler_ALL.py."""

import logging

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

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
            self.data
            .groupby([
                "class",
                "speed",
                "load",
                "severity",
                "circulation",
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
                1 for key in self.data if key[0] == fault_class
            )
            num_samples = sum(
                len(self.data[key]) for key in self.data if key[0] == fault_class
            )

            log.debug(
                f"{self.config_prefix} Num {fault_class} operating conditions {num_operating_conditions} and samples {num_samples}"
            )


    def __len__(self):
        # Number of operating conditions (units), not samples
        return len(self.data.keys())

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
