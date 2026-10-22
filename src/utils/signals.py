import numpy as np
from scipy.interpolate import CubicSpline


def freq_to_index(target_freq, freqs):
    return int(np.argmin(np.abs(freqs - target_freq)))


def calc_antialiasing_cutoff(
    nominal_speed_hz: float,
    pulses_per_rev: int,
    order_of_interest: float,
    raw_sample_rate_hz: float,
    order_margin_factor: float = 1.3,
    nyquist_safety_factor: float = 0.9,
) -> int:
    """
    Calculate the recommended anti-aliasing low-pass filter cutoff frequency
    for a single order-tracked measurement at a fixed nominal shaft speed.

    Returns:
    --------
    int
        The recommended anti-aliasing low-pass filter cutoff frequency in Hz.
    """

    order_nyquist = pulses_per_rev / 2.0
    nyquist_freq_hz = order_nyquist * nominal_speed_hz

    cutoff_hz = order_of_interest * order_margin_factor * nominal_speed_hz

    nyquist_limit_hz = nyquist_freq_hz * nyquist_safety_factor
    if cutoff_hz > nyquist_limit_hz:
        print(
            f"Cutoff ({cutoff_hz:.0f} Hz) exceeds {nyquist_safety_factor * 100:.0f}% "
            f"of order-tracking Nyquist ({nyquist_freq_hz:.0f} Hz) at this speed. "
            f"Increase pulses_per_rev, reduce order_of_interest, or reduce "
            f"order_margin_factor."
        )
        cutoff_hz = nyquist_limit_hz

    raw_nyquist_hz = raw_sample_rate_hz / 2.0
    if cutoff_hz > 0.95 * raw_nyquist_hz:
        print(
            f"Cutoff ({cutoff_hz:.0f} Hz) is more than half the raw ADC "
            f"Nyquist ({raw_nyquist_hz:.0f} Hz) -- limited headroom for "
            f"filter roll-off before raw aliasing. Consider a higher raw "
            f"sample rate or sharper filter."
        )

    return round(cutoff_hz, 1)


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
    ENCODER_TARGET = 5  # * Arbitrarily chosen key revolution starting point

    # Wrap the encoder angle (degrees)
    encoder_wrapped = angle % 360

    # Validate inputs
    assert len(signal) == len(time), (
        f"Signal and timestamps must have the same length ({len(signal)} vs {len(time)})"
    )
    assert len(signal) == len(angle), (
        f"Signal and angle must have the same length ({len(signal)} vs {len(angle)})"
    )
    assert len(signal) > 0, "Signal must not be empty"

    # ORDER TRACKING
    ##

    # Get times of crossings
    crossings = np.where(
        np.logical_and(
            encoder_wrapped[:-1] <= ENCODER_TARGET, encoder_wrapped[1:] > ENCODER_TARGET
        )
    )[0]

    # Get accurate times for crossings
    t_crossings = time[crossings] + (ENCODER_TARGET - encoder_wrapped[crossings]) / (
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


def order_track_signal_from_encoder_pulses(
    signal,
    encoder_signal,
    enc_pulses_per_rev,
    time,
    num_samples_per_revolution,
):
    """
    Resample a vibration signal from constant time intervals to constant
    shaft angle intervals, using every pulse of a raw incremental encoder
    signal.

    Unlike `order_track_signal` -- which locates only one crossing per
    revolution and then assumes constant angular velocity across the whole
    360 degrees in between -- this uses every encoder pulse crossing
    (`enc_pulses_per_rev` of them per revolution) to build a fine-grained
    time-vs-angle mapping. Angular velocity is then only assumed constant
    *between consecutive pulses* (e.g. 1/3600 of a revolution apart), not
    across an entire revolution, so genuine within-revolution speed
    variation (torsional vibration, gear-mesh-induced speed ripple, etc.) is
    preserved instead of smeared.

    Parameters:
    -----------
    signal : numpy.ndarray
        The vibration signal sampled at constant time intervals.
    encoder_signal : numpy.ndarray
        Raw incremental encoder signal, sampled at the same time base as
        `signal`. Assumed to swing through zero once per encoder pulse (e.g.
        a sinusoidal or AC-coupled tachometer output); pulses are detected
        as upward zero-crossings.
    enc_pulses_per_rev : int
        Number of encoder pulses per shaft revolution.
    time : numpy.ndarray
        Timestamps corresponding to each sample in `signal`/`encoder_signal`.
    num_samples_per_revolution : int
        Number of sample points per revolution to include after resampling.

    Returns:
    --------
    numpy.ndarray
        The resampled signal at constant shaft angle intervals, covering as
        many complete revolutions as the encoder data allows.
    """

    assert len(signal) == len(time), (
        f"Signal and timestamps must have the same length ({len(signal)} vs {len(time)})"
    )
    assert len(signal) == len(encoder_signal), (
        f"Signal and encoder signal must have the same length ({len(signal)} vs {len(encoder_signal)})"
    )
    assert len(signal) > 0, "Signal must not be empty"
    assert enc_pulses_per_rev > 0, "enc_pulses_per_rev must be positive"

    # Find every pulse crossing (upward zero-crossing) -- not just one per revolution.
    pulses = np.where(np.logical_and(encoder_signal[:-1] <= 0, encoder_signal[1:] > 0))[
        0
    ]
    assert len(pulses) >= 2, "Need at least two encoder pulses to order track"

    # Accurate (sub-sample) time of each pulse, via linear interpolation.
    t_pulses = time[pulses] + (0 - encoder_signal[pulses]) / (
        encoder_signal[pulses + 1] - encoder_signal[pulses]
    ) * (time[pulses + 1] - time[pulses])

    # Unwrapped angle at each pulse: exactly enc_pulses_per_rev pulses per
    # revolution, by definition of the encoder.
    degrees_per_pulse = 360.0 / enc_pulses_per_rev
    angle_pulses = np.arange(len(t_pulses)) * degrees_per_pulse

    # Only resample whole revolutions, to avoid extrapolating past the last pulse.
    num_complete_revs = int(angle_pulses[-1] // 360)
    interpolation_angles = (
        np.arange(num_samples_per_revolution * num_complete_revs)
        / num_samples_per_revolution
        * 360
    )
    # Time of each desired output angle, interpolated from the full
    # pulse-resolution angle-vs-time mapping -- not just once per revolution.
    interpolation_times = np.interp(interpolation_angles, angle_pulses, t_pulses)

    cs = CubicSpline(time, signal)

    return cs(interpolation_times)


def order_track_signal_from_encoder_angle(
    signal,
    angle,
    time,
    num_samples_per_revolution,
):
    """
    Resample a vibration signal from constant time intervals to constant
    shaft angle intervals, using a continuous (already unwrapped) angle
    signal given at every time sample -- e.g. an encoder's own reported
    angle channel, rather than a raw pulse train.

    Unlike `order_track_signal` -- which locates only one crossing per
    revolution and then assumes constant angular velocity across the whole
    360 degrees in between -- this uses every sample of `angle` directly to
    build the time-vs-angle mapping. Angular velocity is then only assumed
    constant *between consecutive samples* (the native sample resolution),
    which is the finest resolution available when angle is already given as
    a per-sample channel rather than sparse pulses -- no pulse detection is
    needed at all, since we already have angle at every sample.

    Parameters:
    -----------
    signal : numpy.ndarray
        The vibration signal sampled at constant time intervals.
    angle : numpy.ndarray
        Continuous, already-unwrapped shaft angle in degrees, sampled at the
        same time base as `signal`. Must be monotonically increasing (does
        not need to start at 0).
    time : numpy.ndarray
        Timestamps corresponding to each sample in `signal`/`angle`.
    num_samples_per_revolution : int
        Number of sample points per revolution to include after resampling.

    Returns:
    --------
    numpy.ndarray
        The resampled signal at constant shaft angle intervals, covering as
        many complete revolutions as the angle data allows.
    """

    assert len(signal) == len(time), (
        f"Signal and timestamps must have the same length ({len(signal)} vs {len(time)})"
    )
    assert len(signal) == len(angle), (
        f"Signal and angle must have the same length ({len(signal)} vs {len(angle)})"
    )
    assert len(signal) > 0, "Signal must not be empty"
    assert np.all(np.diff(angle) > 0), "angle must be strictly increasing"

    # Only resample whole revolutions, to avoid extrapolating past the last sample.
    num_complete_revs = int((angle[-1] - angle[0]) // 360)
    interpolation_angles = angle[0] + (
        np.arange(num_samples_per_revolution * num_complete_revs)
        / num_samples_per_revolution
        * 360
    )
    # Time of each desired output angle, interpolated from the full
    # sample-resolution angle-vs-time mapping.
    interpolation_times = np.interp(interpolation_angles, angle, time)

    cs = CubicSpline(time, signal)

    return cs(interpolation_times)


def signal_TSA2(
    OT_signal, samples_per_rev, revs_per_TSA_window, trials_per_window, step_size
):
    """
    Time-synchronous average, but keeping each output window `revs_per_TSA_window`
    revolutions long instead of collapsing it to a single revolution.

    Collapsing a TSA window down to one revolution (as `signal_TSA`/`signal_TSA_`
    do) gives an FFT with exactly 1-order resolution -- too coarse to resolve
    sidebands spaced closer than a whole order apart from the carrier or from
    each other. Keeping the window `revs_per_TSA_window` revolutions long
    instead gives 1/revs_per_TSA_window-order resolution (see the `freqs`
    computation at the call site, which divides bin index by
    `revs_per_TSA_window` accordingly), at the cost of needing to average
    several independent `revs_per_TSA_window`-revolution *trials* together
    (instead of single revolutions) to still cancel non-order-locked noise:
    each output window is the average of `trials_per_window` consecutive,
    non-overlapping `revs_per_TSA_window`-revolution segments.

    Successive output windows start `step_size` *revolutions* apart (not
    `step_size` whole trials/blocks apart), so they can overlap heavily --
    this uses far more of a finite recording than stepping a full
    `trials_per_window * revs_per_TSA_window` revolutions between windows
    would.

    Parameters:
    -----------
    OT_signal : numpy.ndarray
        Order tracked signal
    samples_per_rev : int
        Number of samples per revolution
    revs_per_TSA_window : int
        Number of revolutions each output window (and each trial within it)
        spans
    trials_per_window : int
        Number of independent revs_per_TSA_window-revolution trials to
        average together per output window
    step_size : int
        Number of revolutions to advance between the start of successive
        output windows

    Returns:
    --------
    numpy.ndarray
        TSA:d windows, shape (num_windows, revs_per_TSA_window * samples_per_rev)
    """

    # Ensure only full revolutions
    if len(OT_signal) % samples_per_rev != 0:
        OT_signal = OT_signal[: len(OT_signal) - (len(OT_signal) % samples_per_rev)]

    revs = OT_signal.reshape(-1, samples_per_rev)
    n_revs = len(revs)
    revs_per_window = trials_per_window * revs_per_TSA_window

    windows = []
    start = 0
    while start + revs_per_window <= n_revs:
        trials = revs[start : start + revs_per_window].reshape(
            trials_per_window, revs_per_TSA_window * samples_per_rev
        )
        windows.append(trials.mean(axis=0))
        start += step_size

    return np.array(windows)
