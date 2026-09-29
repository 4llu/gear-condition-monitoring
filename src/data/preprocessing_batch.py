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
    # NOTE: Use AGFD options for both. Installations is severity for MCC5
    config_prefix_AGFD = f"AGFD_{split}_"
    config_prefix_MCC5 = f"MCC5_{split}_"
    # config_prefix_UNSW = f"UNSW_{split}_"
    config_prefix_ALL = f"ALL_{split}_"

    assert len(config[f"{config_prefix_AGFD}classes"]) == len(
        config[f"{config_prefix_MCC5}classes"]
    ), "Number of classes in AGFD and MCC5 datasets must be the same!"

    # Figure out the number of classes
    #  UNSW has 2, the rest have 3
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

    assert num_classes in [2, 3], "Number of classes must be 2 or 3"

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

    for i in range(num_groups * 1 if config["separate_dataset_batches"] else 2):
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
            # s_diff = np.maximum(s_diff, 0) # XXX
            # print("->", s_diff.shape)
            # quit()
            # HERE: Implement 0-floor
            # Also, remove ref-based log from gencm

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
    # else:

    # Convert to torch tensors
    samples = torch.tensor(samples, dtype=torch.float32, device=device)
    labels = torch.tensor(labels, dtype=torch.long, device=device)

    return samples, labels, full_labels


def collate_w_difference(
    batch, batchPreprocessor, afterDifferencePreprocessor, config_prefix, config, device
):
    """
    Replacement for default collate_fn to do preprocessing for batches

    Parameters:
        See PyTorch default collate_fn

    Returns:
        samples: torch.tensor(TODO)
        labels: torch.tensor(TODO)
    """

    # Realign dataset `__getitem__` outputs
    samples, labels, full_labels = list(zip(*batch))
    samples = np.stack(list(samples))
    samples = batchPreprocessor.transform(samples)

    # Anchor specific stuff done only when using anchors/difference
    if config["use_difference"]:
        # Separate anchors and the rest of the samples
        if (
            len(samples)
            == len(config[f"{config_prefix}classes"])
            * (config["k_shot"] + config["n_query"])
            + 2
        ):
            support_anchor = samples[0]
            query_anchor = samples[1]
            samples = samples[2:]
            labels = labels[2:]  # Remove anchor labels
            full_labels = full_labels[2:]
        elif (
            len(samples)
            == len(config[f"{config_prefix}classes"])
            * (config["k_shot"] + config["n_query"])
            + 1
        ):
            support_anchor = samples[0]
            query_anchor = samples[0]
            samples = samples[1:]
            labels = labels[1:]  # Remove anchor label
            full_labels = full_labels[1:]
        else:
            raise ValueError(
                f"Something is wrong with the batch size: {samples.shape}!"
            )

        samples = rearrange(
            samples,
            "(n kq) i -> n kq i",
            n=len(config[f"{config_prefix}classes"]),
            kq=config["k_shot"] + config["n_query"],
        )

        supports = samples[:, : config["k_shot"], :]
        queries = samples[:, config["k_shot"] :, :]

        # Calculate difference to healthy anchor
        # (n_way, k_shot OR n_query, input_size)
        # (1    , 1                , input_size)
        supports = supports - support_anchor[np.newaxis, np.newaxis, :]
        queries = queries - query_anchor[np.newaxis, np.newaxis, :]

        samples = np.concatenate(
            [supports, queries],
            axis=1,
        )

        # Last preprocessing step
        samples = afterDifferencePreprocessor.transform(samples)
    elif config_prefix == "AGFD_train_" or config_prefix == "UNSW_train_":
        samples = rearrange(
            samples,
            "(n kq) i -> n kq i",
            n=len(config[f"{config_prefix}classes"]),
            kq=config[f"{config_prefix}num_options"]
            * config[f"{config_prefix}num_queries"],
        )
    else:
        samples = rearrange(
            samples,
            "(n kq) i -> n kq i",
            n=len(config[f"{config_prefix}classes"]),
            kq=config["k_shot"] + config["n_query"],
        )

    # Convert to torch tensors
    samples = torch.tensor(samples, dtype=torch.float32, device=device)
    labels = torch.tensor(labels, dtype=torch.long, device=device)

    return samples, labels, full_labels


def collate_w_batch_processing(batch, batchPreprocessor, config, device):
    """
    Replacement for default collate_fn to do preprocessing for batches

    Parameters:
        See PyTorch default collate_fn

    Returns:
        samples: torch.tensor[TODO]
        labels: torch.tensor(TODO)
    """

    # Realign dataset `__getitem__` outputs
    samples, labels = list(zip(*batch))
    samples = np.stack(list(samples))

    samples = samples.reshape(
        samples.shape[0] // (config["k_shot"] + config["n_query"]),
        config["k_shot"] + config["n_query"],
        -1,
    )

    # Do preprocessing still as numpy arrays
    samples = np.stack(samples)  # FIXME: What does this do?
    samples = batchPreprocessor.transform(samples)

    # Convert to torch tensors
    samples = torch.tensor(samples, dtype=torch.float32, device=device)
    labels = torch.tensor(labels, dtype=torch.long, device=device)

    # log.debug(labels.shape)
    # log.debug(samples.shape)

    return samples, labels


# PREPROCESSING FUNCTIONS
#########################


def add_white_noise(batch, config, split, rng):
    """
    Add white noise to the batch
    """

    # Skip during validation and testing
    if split != "train" or rng.choice([True, False], p=[0.5, 0.5]):
        return batch

    noise = rng.normal(0, config["white_noise_std"], batch.shape)
    return batch + noise


def amplitude_log(batch, config, split, rng):
    return np.log(batch)


def amplitude_log10(batch, config, split, rng):
    return np.log10(batch)


def amplitude_log1p(batch, config, split, rng):
    return np.log1p(batch)


def normalize_to_GMF(batch, config, split, rng):
    # NOTE: Technically the first GMF is useless after this
    return batch / batch[:, config["gear_sideband_block_size"] * 2, np.newaxis]


def zero_floor(batch, config, split, rng):
    return np.maximum(batch, 0)


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


def augment_amplitude(batch, config, split, rng):
    """
    Augment the amplitude of the batch by a random factor
    """

    # Skip during validation and testing
    if split != "train" or rng.choice([True, False], p=[0.5, 0.5]):
        return batch

    # if rng.choice([True, False], p=[0.5, 0.5]):
    factor = rng.uniform(
        config["amplitude_augment_low"],
        config["amplitude_augment_high"],
        size=(batch.shape[0], 1),
    )
    # else:
    #     factor = rng.uniform(
    #         config["amplitude_augment_low"],
    #         config["amplitude_augment_high"],
    #         size=(1, 1),
    #     )

    batch = batch * factor
    return batch


def permutate_blocks(batch, config, split, rng):
    """
    Permutate blocks of the batch
    """

    # Skip during validation and testing
    if split != "train" or rng.choice([True, False], p=[0.6, 0.4]):
        return batch

    # Get the block size
    harmonic_block_size = config["gear_sideband_block_size"] * 4 + 1
    num_harmonics = config["gear_harmonics_to_include"]

    # Create ONE permutation for all samples
    perm = rng.permutation(num_harmonics)

    # Reshape to [N, num_blocks, block_size], permute blocks, reshape back
    batch_reshaped = batch.reshape(batch.shape[0], num_harmonics, harmonic_block_size)
    permuted_batch = batch_reshaped[:, perm, :]
    permuted_batch = permuted_batch.reshape(batch.shape[0], -1)

    return permuted_batch


def permute_within_blocks(batch, config, split, rng):
    """
    Permute positions within each block.
    Same permutation applied to every block of every sample in the batch.
    """

    # Skip during validation and testing
    if split != "train" or rng.choice([True, False], p=[0.8, 0.2]):
        return batch

    # Get the block size
    harmonic_block_size = config["gear_sideband_block_size"] * 4 + 1
    num_harmonics = config["gear_harmonics_to_include"]

    # Create ONE permutation for positions within a block
    perm = rng.permutation(harmonic_block_size)

    # Reshape to [N, num_blocks, block_size], permute within blocks, reshape back
    batch_reshaped = batch.reshape(batch.shape[0], num_harmonics, harmonic_block_size)
    permuted_batch = batch_reshaped[:, :, perm]
    permuted_batch = permuted_batch.reshape(batch.shape[0], -1)

    return permuted_batch


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
