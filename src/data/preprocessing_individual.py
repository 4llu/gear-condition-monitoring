import numpy as np

# PREPROCESSING FUNCTIONS
#########################


def get_important_frequencies(teeth, harmonics, sideband_block_size):
    """
    Calculate important frequencies for gear analysis. For each GMF harmonic,
    the `sideband_block_size` furthest and closest sidebands on both sides are
    included, in addition to the harmonic itself. The furthest sideband is
    `ceil(teeth / 2 - 1)` orders from the harmonic.

    The output always has `harmonics * (4 * sideband_block_size + 1)` entries,
    independent of `teeth`, so datasets with different gears give inputs of
    the same size.

    NOTE: If `2 * sideband_block_size > furthest_sideband`, the furthest and
    closest blocks on the same side overlap, so some frequencies are included
    twice and the output is no longer in ascending order. With
    `sideband_block_size > furthest_sideband`, blocks also reach into the
    neighbouring harmonic's sidebands. The largest non-overlapping block size
    is 3 for 15 teeth (AGFD), 6 for 27 teeth (UNSW) and 8 for 36 teeth (MCC5).

    Args:
        teeth (int): Number of teeth on the gear
        harmonics (int): Number of GMF harmonics to include
        sideband_block_size (int): Number of sidebands in each block

    Returns:
        list: Frequency indices (in orders), grouped per harmonic as
            [furthest left, closest left, harmonic, closest right, furthest right].
            Ascending only when the blocks don't overlap.
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


class IndividualPreprocessor:
    def __init__(self, config, dataset_name, split):
        self.config = config
        self.config_prefix = f"{dataset_name}_{split}_"

    def transform(self, samples):
        for func_name in self.config["preprocessing_individual"]:
            func = globals().get(func_name)
            if callable(func):
                samples = func(samples, self.config_prefix, self.config)
            else:
                raise ValueError(  # noqa: TRY004
                    f"Unknown batch preprocessing step '{func_name}' in "
                    f"config['{self.stage}']!"
                )

        return samples
