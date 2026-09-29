import numpy as np


def freq_to_index(target_freq, freqs):
    return int(np.argmin(np.abs(freqs - target_freq)))
