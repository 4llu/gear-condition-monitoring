import numpy as np
from scipy.interpolate import CubicSpline


def order_track_FFT(
    signal,
    time,
    angle,
    num_orders,
    fs=None,
):
    """
    Resample a vibration signal from constant time intervals to constant shaft angle intervals.

    Parameters:
    -----------
    signal : numpy.ndarray
        The vibration signal sampled at constant time intervals.
    time : numpy.ndarray | None
        Timestamps corresponding to each sample in the signal. Used for cleaning out revolutions where sampling skipping happens. Skipped if None.
        (Necessary because of problems with the AGFD data)
    angle : numpy.ndarray
        Shaft angle (unwrapped) at each sample point.
    num_orders : int
        Number of sample points per revolution to include after resampling.

    Returns:
    --------
    numpy.ndarray
        The order FFTs of all the revolutions in the signal up to the requested number of orders.
    """

    # Constants
    encoder_target = 5  # * Arbitrarily chosen key revolution starting point

    # Wrap the encoder angle (degrees)
    encoder_wrapped = angle % 360

    # Validate inputs
    if len(signal) != len(angle):
        raise ValueError(
            f"Signal and angle must have the same length ({len(signal)} vs {len(angle)})"
        )

    # ORDER TRACKING
    ##

    # Get times of crossings
    crossings = np.where(
        np.logical_and(
            encoder_wrapped[:-1] <= encoder_target, encoder_wrapped[1:] > encoder_target
        )
    )[0]

    FFTs = []
    for i in range(1, len(crossings) - 1):
        if time is not None:
            # Skip any revolutions where sampling has failed
            time_diff = np.diff(time[crossings[i - 1] : crossings[i]])
            if fs is not None and (
                any(time_diff > 3 * 1 / fs) or any(time_diff < 0.1 * 1 / fs)
            ):
                print(f"Skipping revolution {i} due to sampling failure")
                continue

        revolution_s = signal[crossings[i - 1] : crossings[i]]
        fft = np.fft.rfft(revolution_s, norm="forward")

        if len(fft) < num_orders + 1:
            raise ValueError(
                f"Not long enough signal to keep that many orders ({num_orders}) vs {len(fft)})"
            )

        FFTs.append(fft[:num_orders])

    return np.array(FFTs)


def FFT_TSA(revolution_FFTs, window_len, step_size):
    """
    Average windows of FFTs from separate revolutions to create an average of the FFTs. Takes in the complex spectrums.

    Parameters:
    -----------
    revolution_FFTs : numpy.ndarray
        The FFTs of singular revolutions. COMPLEX NUMBERS! Shape (# revolutions, # bins)
    window_len : int
        Number of revolutions to include per TSA
    step_size : int
        Number of revolutions to skip between windows

    Returns:
    --------
    numpy.ndarray
        Averaged FFTs. Shape (# windows, # orders)
    """

    i = 0
    samples = []
    while i + window_len < len(revolution_FFTs):
        # ! Choose averaging method
        samples.append(np.abs(np.mean(revolution_FFTs[i : i + window_len], axis=0)))
        # samples.append(np.mean(np.abs(revolution_FFTs[i : i + window_len]), axis=0))
        # samples.append(np.mean(np.square(np.abs(revolution_FFTs[i : i + window_len])), axis=0))
        i += step_size

    # Convert to float 32 as required by pytorch
    return np.array(samples)


def order_track_signal(
    signal,
    time,
    angle,
    num_samples_per_revolution,
):
    """
    Resample a vibration signal from constant time intervals to constant shaft angle intervals.

    Parameters:
    -----------
    signal : numpy.ndarray
        The vibration signal sampled at constant time intervals.
    t : numpy.ndarray
        Timestamps corresponding to each sample in the signal.
    angle : numpy.ndarray
        Shaft angle (unwrapped) at each sample point.
    num_samples_per_revolution : int
        Number of sample points per revolution to include after resampling.
    fs : int
        Sampling frequency of the original signal.
    highest_rpm : int
        Highest RPM in the signal (required for anti-aliasing filter).
    num_teeth : int
        Number of teeth in gear being tracked.
    skip_anti_aliasing_warning : bool
        If True, skip raising an error if the sampling frequency isn't enough for 8 orders of GMF. Default is False.

    Returns:
    --------
    numpy.ndarray
        The resampled signal at constant shaft angle intervals.
    """

    # Constants
    encoder_target = 5  # * Arbitrarily chosen key revolution starting point
    GMF_orders = 10  # Number of GMF orders to include in the anti-aliasing filter

    # Wrap the encoder angle (degrees)
    encoder_wrapped = angle % 360

    # Validate inputs
    assert len(signal) == len(
        time
    ), f"Signal and timestamps must have the same length ({len(signal)} vs {len(time)})"
    assert len(signal) == len(
        angle
    ), f"Signal and angle must have the same length ({len(signal)} vs {len(angle)})"
    assert len(signal) > 0, "Signal must not be empty"

    # ORDER TRACKING
    ##

    # Get times of crossings
    crossings = np.where(
        np.logical_and(
            encoder_wrapped[:-1] <= encoder_target, encoder_wrapped[1:] > encoder_target
        )
    )[0]

    # Get accurate times for crossings
    t_crossings = time[crossings] + (encoder_target - encoder_wrapped[crossings]) / (
        encoder_wrapped[crossings + 1] - encoder_wrapped[crossings]
    ) * (time[crossings + 1] - time[crossings])

    # Create angles for each crossing (different anchor to original encoder angles)
    angle_crossings = np.arange(len(t_crossings)) * 360

    # Add intermediate sampling point angles
    interpolation_angles = (
        np.arange(num_samples_per_revolution * (len(t_crossings) - 1))
        / num_samples_per_revolution
        * 360
    )
    # Get corresponding times for each new sample point
    interpolation_times = np.interp(interpolation_angles, angle_crossings, t_crossings)

    cs = CubicSpline(time, signal)

    return cs(interpolation_times)


def signal_TSA(OT_signal, num_samples_per_revolution, window_len, step_size):
    """
    Average windows of FFTs from separate revolutions to create an average of the FFTs. Takes in the complex spectrums.

    Parameters:
    -----------
    OT_signal : numpy.ndarray
        Order tracked signal
    num_samples_per_revolution : int
        Number of samples per revolution
    window_len : int
        Number of revolutions to include per TSA
    step_size : int
        Number of revolutions to skip between windows

    Returns:
    --------
    numpy.ndarray
        TSA:d revolutions Shape (# windows, # numer_samples_per_revolution)
    """

    OT_signals = OT_signal.reshape(-1, num_samples_per_revolution)

    i = 0
    samples = []
    while i + window_len < len(OT_signals):
        samples.append(np.mean(OT_signals[i : i + window_len], axis=0))
        i += step_size

    return np.array(samples)


# OTHER HELPERS


def fold(x, w):
    org_shape = x.shape
    extra = len(x) % w
    if extra != 0:
        x = x[:-extra]
    return x.reshape(-1, w, *org_shape[1:])
