import logging

import numpy as np
import torch
from einops import rearrange, reduce, repeat

log = logging.getLogger("gear-cm")

# COLLATION
###########


def collate_all(
    batch, batchPreprocessor, afterDifferencePreprocessor, split, config, device
):
    samples, labels, full_labels = list(zip(*batch))
    labels = np.array(labels)
    samples = np.stack(list(samples))

    samples = batchPreprocessor.transform(samples)

    # ALL
    config_prefix_ALL = f"ALL_{split}_"

    assert config["separate_dataset_batches"], (
        "The number of classes is derived per batch, so each batch must come from one "
        "dataset (datasets can have different numbers of classes)"
    )

    # Figure out the number of classes
    #  Differs between datasets, e.g. UNSW has 2 and MCC5 has 3
    num_total_anchors = (
        config[f"{config_prefix_ALL}num_options"]
        * config[f"{config_prefix_ALL}num_anchors"]
    )

    num_total_non_anchors = (
        config[f"{config_prefix_ALL}num_options"]
        * config[f"{config_prefix_ALL}num_queries"]
        * config[f"{config_prefix_ALL}num_installations"]
    )

    num_classes = (len(samples) - num_total_anchors) / num_total_non_anchors
    if num_classes.is_integer():
        num_classes = int(num_classes)
    else:
        raise ValueError(f"Number of classes must be an integer ({num_classes})!")

    assert num_classes >= 2, "Need healthy and at least one fault class"

    # Figure out query group size

    group_query_size = (
        num_classes
        * config[f"{config_prefix_ALL}num_installations"]
        * config[f"{config_prefix_ALL}num_queries"]
    )
    group_size = group_query_size + config[f"{config_prefix_ALL}num_anchors"]
    num_groups = config[f"{config_prefix_ALL}num_options"]

    new_samples = []
    new_labels = []

    for i in range(num_groups):
        # Drop anchor labels
        group_labels = labels[
            i * group_size + config[f"{config_prefix_ALL}num_anchors"] : (i + 1)
            * group_size
        ]

        # Separate anchors and the rest of the samples
        group_anchors = samples[
            i * group_size : i * group_size + config[f"{config_prefix_ALL}num_anchors"]
        ]
        group_samples = samples[
            i * group_size + config[f"{config_prefix_ALL}num_anchors"] : (i + 1)
            * group_size
        ]

        if config["use_difference"]:
            # Compute difference to anchors
            s_diff = group_samples[:, np.newaxis, :] - group_anchors[np.newaxis, :, :]

            # Flatten or average over anchors
            if config["average_over_anchors"]:
                s = reduce(s_diff, "samples anchors emb -> samples emb", "mean")
            else:
                # Flatten
                s = rearrange(s_diff, "samples anchors emb -> (samples anchors) emb")
                # Repeat labels for each anchor
                group_labels = repeat(
                    group_labels,
                    "samples -> (samples anchors)",
                    anchors=config[f"{config_prefix_ALL}num_anchors"],
                )
        else:
            # NOTE: This option works, this is just here to make sure it isn't used by accident
            # raise Exception("NO ANCHOR DIFFERENCE! (Remove this exception if you are sure this is what you want)")
            s = group_samples

        new_samples.append(s)
        new_labels.extend(group_labels)

    # Reassemble groups
    labels = np.array(new_labels)
    samples = np.concatenate(
        new_samples,
        axis=0,
    )

    # Last preprocessing step
    samples = afterDifferencePreprocessor.transform(samples)

    # Convert to torch tensors
    samples = torch.tensor(samples, dtype=torch.float32, device=device)
    labels = torch.tensor(labels, dtype=torch.long, device=device)

    return samples, labels, full_labels


# PREPROCESSING FUNCTIONS
#########################


def amplitude_log10(batch, config, split, rng):
    return np.log10(batch)


def mask_blocks_non_episodic(batch, config, split, rng):
    """
    Mask blocks of the batch to zero for non-episodic training.
    """

    # Skip masking during validation and testing or with 80% chance
    if (
        split != "train"
        or config["num_blocks_to_mask"] == 0
        or rng.choice([True, False], p=[0.6, 0.4])
    ):
        return batch

    # Block size is 21 (gear_sideband_block_size * 4 + 1)
    block_size = config["gear_sideband_block_size"] * 4 + 1
    num_blocks = config["gear_harmonics_to_include"]

    # Randomly select which blocks to mask
    blocks_to_mask = rng.choice(
        num_blocks, size=config["num_blocks_to_mask"], replace=False
    )

    # Create mask for all samples
    mask = np.ones(batch.shape[1], dtype=bool)
    for block_idx in blocks_to_mask:
        start_idx = block_idx * block_size
        end_idx = start_idx + block_size
        mask[start_idx:end_idx] = False

    # Apply mask to all samples in batch
    batch = batch.copy()
    batch[:, ~mask] = 0
    return batch


def add_unit_variation(batch, config, split, rng):
    """
    Add a random deviation to every sample (after the anchor difference) to imitate
    the variation between healthy units, which training data with few healthy units
    (UNSW: two healthy-labelled gears, MCC5: one) doesn't contain. Labels are
    unchanged.

    `unit_variation_type`:
        "structured": a random offset per harmonic block plus a smooth tilt over the
            blocks, i.e. gain differences of the path to the sensor
        "white": independent noise for every bin (control)
    `unit_variation_rms`: mean RMS of the deviation (log10 units), the RMS of each
        sample's deviation is drawn uniformly from [0, 2 * unit_variation_rms]
    """

    if split != "train":
        return batch

    num_blocks = config["gear_harmonics_to_include"]
    block_size = config["gear_sideband_block_size"] * 4 + 1

    if config["unit_variation_type"] == "structured":
        offsets = rng.normal(size=(batch.shape[0], num_blocks))
        tilt = rng.normal(size=(batch.shape[0], 1)) * np.linspace(-1, 1, num_blocks)
        deviation = np.repeat(offsets + tilt, block_size, axis=1)
    elif config["unit_variation_type"] == "white":
        deviation = rng.normal(size=batch.shape)
    else:
        raise ValueError(f"Unknown unit_variation_type {config['unit_variation_type']}")

    # Units differ by varying amounts, so every sample gets its own RMS
    rms = np.sqrt(np.mean(deviation**2, axis=1, keepdims=True))
    target_rms = rng.uniform(0, 2 * config["unit_variation_rms"], size=(batch.shape[0], 1))
    deviation = deviation / rms * target_rms

    return batch + deviation


def augment_amplitude(batch, config, split, rng):
    """
    Augment the amplitude of the batch by a random factor
    """

    # Skip during validation and testing
    if split != "train" or rng.choice([True, False], p=[0.5, 0.5]):
        return batch

    factor = rng.uniform(
        config["amplitude_augment_low"],
        config["amplitude_augment_high"],
        size=(batch.shape[0], 1),
    )

    batch = batch * factor
    return batch


class BatchPreprocessor:
    def __init__(self, config, stage, split, rng):
        self.config = config
        self.stage = stage
        self.split = split
        self.rng = rng

    def transform(self, batch):
        for func_name in self.config[self.stage]:
            func = globals().get(func_name)
            if callable(func):
                batch = func(batch, self.config, self.split, self.rng)
            else:
                raise ValueError(  # noqa: TRY004
                    f"Unknown batch preprocessing step '{func_name}' in "
                    f"config['{self.stage}']!"
                )

        return batch
