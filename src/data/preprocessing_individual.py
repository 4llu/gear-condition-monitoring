import numpy as np

# PREPROCESSING FUNCTIONS
#########################


def get_important_frequencies(teeth, harmonics, sideband_block_size):
    """
    Calculate important frequencies for gear analysis. `sidebank_block_size`
    furthest and closest sidebands (on both sides) to each GMF harmonic will
    be included, in addition to the harmonic itself.

    Args:
        teeth (int): Number of teeth on the gear
        harmonics (int): Number of harmonics to consider
        furthest_sideband (int): Maximum sideband distance from harmonic
        sideband_block_size (int): Number of sidebands to include in each block

    Returns:
        list: Important frequencies in ascending order
    """

    important_freq = []
    furthest_sideband = np.ceil((teeth / 2) - 1).astype(int)
    base_freqs = [i * teeth for i in range(1, harmonics + 1)]

    for base in base_freqs:
        # Left sidebands
        important_freq.extend(
            range(
                base - furthest_sideband,
                base - furthest_sideband + sideband_block_size,
            )
        )
        important_freq.extend(range(base - sideband_block_size, base))

        # Center frequency
        important_freq.append(base)

        # Right sidebands
        important_freq.extend(range(base + 1, base + sideband_block_size + 1))
        important_freq.extend(
            range(
                base + furthest_sideband - sideband_block_size + 1,
                base + furthest_sideband + 1,
            )
        )

    return important_freq


def cherry_pick_indices(samples, config_prefix, config):
    return samples[
        :,
        get_important_frequencies(
            teeth=config[f"{config_prefix}teeth"],
            harmonics=config["gear_harmonics_to_include"],
            sideband_block_size=config["gear_sideband_block_size"],
        ),
    ]


def harmonic_band(samples, config_prefix, config):
    assert config[f"{config_prefix}teeth"] % 2 != 0, (
        "Check if the math checks out of odd number of teeth (also for masking)!"
    )

    # print(samples.shape)
    samples = samples[
        :,
        config[f"{config_prefix}teeth"]
        - (config[f"{config_prefix}teeth"] // 2) : config["gear_harmonics_to_include"]
        * config[f"{config_prefix}teeth"]
        + config[f"{config_prefix}teeth"] // 2  # Last right side sideband
        + 1,  # Because DC is skipped
    ]
    # print(samples.shape)

    return samples


class IndividualPreprocessor:
    def __init__(self, config, dataset_name, split):
        self.config = config
        self.config_prefix = f"{dataset_name}_{split}_"

    def transform(self, samples):
        for func_name in self.config["preprocessing_individual"]:
            func = globals().get(func_name)
            if callable(func):
                samples = func(samples, self.config_prefix, self.config)

        return samples
