import numpy as np
from scipy.interpolate import CubicSpline


def order_track_signal_3(signal, timestamps, key_crossings, points_per_rev, fs, rpm):
    """
    Resample a vibration signal from constant time intervals to constant shaft angle intervals.

    Parameters:
    -----------
    signal : numpy.ndarray
        The vibration signal sampled at constant time intervals.
    timestamps : numpy.ndarray
        Timestamps corresponding to each sample in the signal.
    key_crossings : numpy.ndarray
        Array containing the exact shaft key crossing time of each revolution.
    points_per_rev : int
        Number of sample points per revolution to include after resampling.

    Returns:
    --------
    numpy.ndarray
        The resampled signal at constant shaft angle intervals.
    """

    # Validate inputs
    if len(signal) != len(timestamps):
        raise ValueError("Signal and timestamps must have the same length")

    # sos = butter(2, highest_required_freq, btype="lowpass", output="sos", fs=3012)
    # signal = sosfiltfilt(sos, signal)

    first_valid_index = np.searchsorted(timestamps, key_crossings[0], side="right")
    last_valid_index = np.searchsorted(timestamps, key_crossings[-1], side="right")
    signal = signal[first_valid_index:last_valid_index]

    # Determine shaft angle for each timestamp
    shaft_angles = []

    # Calculate shaft angles for each timestamp
    for i in range(len(key_crossings) - 1):
        # Find all timestamps between consecutive key crossings
        mask = (timestamps >= key_crossings[i]) & (timestamps < key_crossings[i + 1])

        if np.any(mask):
            # Normalize time to angle (0 to 2π) for each revolution
            t_norm = (timestamps[mask] - key_crossings[i]) / (key_crossings[i + 1] - key_crossings[i])
            shaft_angles_segment = 2 * np.pi * i + 2 * np.pi * t_norm

            shaft_angles.extend(shaft_angles_segment)

    shaft_angles = np.array(shaft_angles)

    # Create uniform shaft angle grid for resampling
    # Total angle span
    total_revs = len(key_crossings) - 1
    angle_start = 0
    angle_end = 2 * np.pi * total_revs

    # Create uniformly spaced angles
    uniform_angles = np.linspace(angle_start, angle_end, int(points_per_rev * total_revs), endpoint=False)

    # Perform cubic spline interpolation
    cs = CubicSpline(shaft_angles, signal)

    # Resample signal at uniform shaft angles
    order_tracked_signal = cs(uniform_angles)

    return order_tracked_signal
