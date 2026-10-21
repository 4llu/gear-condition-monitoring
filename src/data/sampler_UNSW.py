"""UNSW dataset: samples grouped by operating condition, used by sampler_ALL.py."""

import logging

import numpy as np
from torch.utils.data import Dataset

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

        possible_samples = self.data[idx]
        sample = possible_samples[self.rng.integers(len(possible_samples))]
        label = self.config["class_map"][idx[0]]

        return sample, label, idx
