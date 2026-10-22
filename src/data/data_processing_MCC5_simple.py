"""
MCC5-THU processing that only writes the original format: the magnitude
spectrum of the one-revolution TSA of the order tracked signal (the
"spectrum" representation of data_processing_MCC5.py, which also writes the
envelope representations).

- Journal paper: https://www.sciencedirect.com/science/article/pii/S2352340924004220
- Data in Brief: https://arxiv.org/pdf/2403.12521
- Mendeley Data: https://data.mendeley.com/datasets/p92gj2732w/2
- GitHub\_ https://github.com/liuzy0708/MCC5-THU-Gearbox-Benchmark-Datasets

**Notes**

Speeds seem to be +/- 50 RPM from the reported ones, but highest discrepancy is almost 150 RPM.

**ERRORS**

_Suffix `ERROR_` added to these files to make skipping easier\_

- gear_pitting_M_speed_circulation_20Nm-1000rpm.csv - Distortion, gain error, or something
- gear_pitting_M_torque_circulation_2000rpm_10Nm.csv - Distortion, gain error, or something
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

# This script lives in gear_vibe/data/, but `utils` is a package at the
# gear_vibe/ project root -- one level up. Python only puts the running
# script's own directory on sys.path, not its parent, so `from utils...`
# only works when this file happens to be run with the project root as the
# working directory. Adding the project root explicitly makes the import
# work regardless of cwd (terminal, editor "run current file", notebook, etc).
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.signals import (
    calc_antialiasing_cutoff,
    order_track_signal,
    signal_TSA2,
)

SAMPLING_RATE = 12800  # Hz

# The only speed measurement is a once-per-revolution key phase pulse on the
# MOTOR output shaft -- there is no encoder on the intermediate shaft that
# actually carries the (faulty) stage-2 pinion, and no per-pulse/per-sample
# angle channel either, so unlike AGFD/UNSW there is no "precise" OT variant
# available here -- only the coarse, one-crossing-per-revolution method
# (order_track_signal), same as AGFD/UNSW's non-precise fallback.
FIRST_GEAR_RATIO = (
    29 / 95
)  # Stage-1 gearbox ratio: motor shaft -> intermediate (pinion) shaft
PINION_TEETH = (
    36  # Stage-2 pinion, on the intermediate shaft, where the faults are induced
)

# Key phase pulse: ~0V baseline, ~3.6V during the brief once-per-revolution
# pulse. 1.8V cleanly separates the two states regardless of speed/torque
# (checked against files at 500-3000 RPM and both circulation types).
KEY_PHASE_THRESHOLD = 1.8

# Samples per revolution of the pinion's own shaft in the order-tracked
# signal (order tracking below scales the motor-shaft key phase crossings by
# FIRST_GEAR_RATIO so the OT signal is referenced to the pinion, not the
# motor). A clean multiple of PINION_TEETH, so GMF harmonics land on exact
# order bins. At the highest tested speed (3000 RPM motor, ~15.3 Hz pinion)
# the raw 12800 Hz sampling only gives ~838 raw samples per pinion
# revolution, so that speed is slightly upsampled. Its order Nyquist (432)
# still leaves room for the anti-aliasing filter at 10 GMFs (order 360).
SAMPLES_PER_REV = PINION_TEETH * 24  # = 864

# Revolutions averaged into each one-revolution TSA window, so also the
# revolutions one TSA window consumes
WINDOWS_PER_TSA = 30
# Spectrum bins to keep: the 10 first full GMFs and the DC component
SPECTRUM_BINS = 10 * PINION_TEETH

TARGET_TSA_WINDOWS = 10

# Excluded entirely: not enough pinion revolutions per segment to be worth
# including.
SKIP_SPEEDS = {500}


def step_size_for_target_windows(n_revs: int, target: int = TARGET_TSA_WINDOWS) -> int:
    """Step size (in revolutions) that makes signal_TSA2 produce ~`target`
    windows from a segment with `n_revs` available revolutions, given
    WINDOWS_PER_TSA revolutions per window."""
    return max(1, round((n_revs - WINDOWS_PER_TSA) / (target - 1)))


# Upper/lower plateau pairs for the two circulation types (see the
# time-varying speed/load profiles in the dataset's documentation): a
# "speed_circulation" file's load is fixed but its speed alternates between
# these pairs, and vice versa for "torque_circulation".
speed_cycle_map = {
    1000: 500,
    2000: 1500,
    3000: 2500,
}
torque_cycle_map = {
    10: 5,
    20: 15,
}

fault_map = {
    "teeth_crack": "crack",
    "gear_wear": "wear",
    "gear_pitting": "pitting",
    "miss_teeth": "missing_tooth",
    "teeth_break": "broken_tooth",
}

# L/M/H/XL (raw filename codes) -> S/M/L/XL (severity convention used
# elsewhere in this project)
severity_map = {
    "L": "S",
    "M": "M",
    "H": "L",
    "XL": "XL",
}

SENSORS = [
    "motor_vibration_x",
    "motor_vibration_y",
    "motor_vibration_z",
    "gearbox_vibration_x",
    "gearbox_vibration_y",
    "gearbox_vibration_z",
]

# Seconds to cut out around each plateau for accelerations/decelerations.
# (upper, lower, upper) plateau time windows, matching the trapezoidal
# speed/load profiles: ramp 0-10s, plateau 10-20s, ramp down 20-25s, lower
# plateau 25-35s, ramp up 35-40s, plateau 40-50s, ramp down 50-60s.
margin = 0.5
plateau_windows = [(10, 20), (25, 35), (40, 50)]

# Raw data, and where the processed data is written
DATA_FOLDER = Path(__file__).parent.parent.parent.parent.resolve() / "data" / "MCC5-THU"
OUTPUT_FILE = Path(__file__).parent.parent.parent / "data" / "MCC5_V6.feather"

# HELPERS


def enc_angle_from_key_phase(key_phase, time, ratio):
    """
    Build a continuous, unwrapped shaft angle (degrees), referenced to a
    shaft OTHER than the one the key phase sensor is actually on, from a
    once-per-revolution key phase pulse train.

    `ratio` is that other shaft's revolutions per key-phase-shaft
    revolution (e.g. FIRST_GEAR_RATIO to go from the motor shaft, which has
    the sensor, to the pinion's own shaft). Angular velocity is assumed
    constant between consecutive key phase pulses (i.e. across one whole
    key-phase-shaft revolution) -- the same coarse assumption
    `order_track_signal` itself makes, just one level removed since it
    normally receives the tracked shaft's own crossings directly.

    Parameters:
    -----------
    key_phase : numpy.ndarray
        Raw key phase sensor signal, low baseline with a brief high pulse
        once per revolution of the shaft it is mounted on.
    time : numpy.ndarray
        Timestamps corresponding to each sample in `key_phase`.
    ratio : float
        Revolutions of the shaft to reference the output angle to, per one
        revolution of the key-phase shaft.

    Returns:
    --------
    numpy.ndarray
        Unwrapped angle (degrees) of the referenced shaft, at every input
        time sample.
    """

    above = key_phase > KEY_PHASE_THRESHOLD
    crossings = np.where(np.logical_and(~above[:-1], above[1:]))[0]
    assert len(crossings) >= 2, "Need at least two key phase pulses to order track"

    # Sub-sample crossing time via linear interpolation
    t_crossings = time[crossings] + (KEY_PHASE_THRESHOLD - key_phase[crossings]) / (
        key_phase[crossings + 1] - key_phase[crossings]
    ) * (time[crossings + 1] - time[crossings])

    # Each crossing = one full key-phase-shaft revolution
    angle_crossings = np.arange(len(t_crossings)) * 360 * ratio

    return np.interp(time, t_crossings, angle_crossings)


def parse_labels(stem):
    """
    The labels of a recording from its file name, e.g.
    health_speed_circulation_10Nm-1000rpm or
    gear_wear_M_torque_circulation_2000rpm_20Nm.
    None for the recordings that are left out (ERROR files, other faults
    and fault combinations).
    """
    p = stem.split("_")

    # Skip files with something wrong in them
    if p[0] == "ERROR":
        return None

    # Healthy
    if p[0] == "health":
        fault = "healthy"
        level = "-"
        circulation_parts = p[1:]
    # Faulty
    else:
        fault = f"{p[0]}_{p[1]}"

        # Skip other faults for now
        if fault not in fault_map or p[2] == "and":
            return None

        fault = fault_map[fault]
        if fault == "missing_tooth":
            # NOTE: Missing tooth doesn't have severity.
            p.insert(2, "L")

        level = severity_map[p[2]]  # L/M/H/XL -> S/M/L/XL
        circulation_parts = p[3:]

    if f"{circulation_parts[0]}_{circulation_parts[1]}" == "torque_circulation":
        circulation = "torque"
        rpm = int(circulation_parts[2].replace("rpm", ""))
        torque = int(circulation_parts[3].replace("Nm", ""))
    elif f"{circulation_parts[0]}_{circulation_parts[1]}" == "speed_circulation":
        circulation = "speed"
        torque, rpm = circulation_parts[2].split("-")
        torque = int(torque.replace("Nm", ""))
        rpm = int(rpm.replace("rpm", ""))
    else:
        return None

    return {
        "fault": fault,
        "level": level,
        "circulation": circulation,
        "rpm": rpm,
        "torque": torque,
    }


def plateau_spectra(signal, key_phase, time, plateau_rpm, name):
    """
    Magnitude spectra of the TSA windows of one sensor's signal on one
    plateau, as float32 (what Pytorch expects): shape (num_windows,
    SPECTRUM_BINS), with no windows when the plateau has too few
    revolutions. `name` identifies the signal in errors.
    """
    max_value = np.max(np.abs(signal))
    if max_value > 20:
        raise Exception(  # noqa: TRY002
            f"{name} contains values over 20. Max absolute value: {max_value}"
        )

    # Anti-aliasing filter
    antialiasing_cutoff = calc_antialiasing_cutoff(
        plateau_rpm / 60 * FIRST_GEAR_RATIO,
        pulses_per_rev=SAMPLES_PER_REV,
        order_of_interest=10 * PINION_TEETH,
        raw_sample_rate_hz=SAMPLING_RATE,
        order_margin_factor=1.0,
        nyquist_safety_factor=0.9,
    )
    sos = butter(
        4, antialiasing_cutoff, btype="lowpass", output="sos", fs=SAMPLING_RATE
    )
    filtered = sosfiltfilt(sos, signal)

    # Order track the signal, referenced to the pinion's own shaft
    enc_angle = enc_angle_from_key_phase(key_phase, time, FIRST_GEAR_RATIO)
    signal_OT = order_track_signal(
        filtered, time, enc_angle, num_samples_per_revolution=SAMPLES_PER_REV
    )

    # Remove first and last revolutions to remove any weird artifacts
    signal_OT = signal_OT[SAMPLES_PER_REV:-SAMPLES_PER_REV]

    n_revs_available = len(signal_OT) // SAMPLES_PER_REV
    if n_revs_available < WINDOWS_PER_TSA:
        return np.empty((0, SPECTRUM_BINS), dtype=np.float32)

    step_size = step_size_for_target_windows(n_revs_available)
    signal_OT_TSA = signal_TSA2(
        signal_OT,
        SAMPLES_PER_REV,
        1,  # One-revolution windows
        WINDOWS_PER_TSA,
        step_size,
    )

    return np.abs(np.fft.rfft(signal_OT_TSA, norm="forward", axis=1))[
        :, :SPECTRUM_BINS
    ].astype(np.float32)


def process_recording(path, labels):
    """One row per TSA window of a recording: its labels and the spectrum
    of every sensor, the upper plateaus' windows first."""
    rpm, torque, circulation = labels["rpm"], labels["torque"], labels["circulation"]

    df = pd.read_csv(path)
    time = np.arange(len(df)) / SAMPLING_RATE
    key_phase = df["speed"].to_numpy()

    # Actual motor speed (RPM) during each of the three plateaus -- differs
    # from the file-level `rpm` label on the plateau where the OTHER
    # condition is the one being cycled.
    plateau_rpms = [
        rpm,
        speed_cycle_map[rpm] if circulation == "speed" else rpm,
        rpm,
    ]

    # "upper"/"lower" -> the spectra of every sensor
    samples = {"upper": [], "lower": []}
    for sensor in SENSORS:
        part_samples = []  # One per plateau
        for (start_s, end_s), plateau_rpm in zip(plateau_windows, plateau_rpms):
            if plateau_rpm in SKIP_SPEEDS:
                part_samples.append(np.empty((0, SPECTRUM_BINS), dtype=np.float32))
                continue

            sl = slice(
                int((start_s + margin) * SAMPLING_RATE),
                int((end_s - margin) * SAMPLING_RATE),
            )
            part_samples.append(
                plateau_spectra(
                    df[sensor].to_numpy()[sl],
                    key_phase[sl],
                    time[sl],
                    plateau_rpm,
                    name=f"{sensor} in {path}",
                )
            )

        print(
            sensor,
            "upper",
            part_samples[0].shape,
            part_samples[2].shape,
            "lower",
            part_samples[1].shape,
        )
        samples["upper"].append(
            np.concatenate([part_samples[0], part_samples[2]], axis=0)
        )
        samples["lower"].append(part_samples[1])

    plateau_labels = {
        "upper": {"speed": rpm, "load": torque},
        "lower": {
            "speed": speed_cycle_map[rpm] if circulation == "speed" else rpm,
            "load": torque_cycle_map[torque] if circulation == "torque" else torque,
        },
    }

    # Make a new row for the DF for every TSA sample
    return pd.DataFrame([
        {
            **plateau_labels[plateau],
            "severity": labels["level"],
            "class": labels["fault"],
            "circulation": circulation,
            "OT_method": "signal",
            **{sensor: samples[plateau][i][j] for i, sensor in enumerate(SENSORS)},
        }
        for plateau in ("upper", "lower")
        for j in range(samples[plateau][0].shape[0])
    ])


# PROCESSING


def main(data_folder=DATA_FOLDER, output_file=OUTPUT_FILE):
    print(f"Loading MCC5-THU data from {data_folder}...")
    # Sorted, so the output's row order doesn't depend on the filesystem
    files = sorted(data_folder.glob("*.csv"))

    dfs = []
    for f in files:
        labels = parse_labels(f.stem)
        if labels is None:
            continue
        print(f, *labels.values())
        dfs.append(process_recording(f, labels))

    # Combine files
    dfs = pd.concat(dfs)
    dfs["TSA_size"] = 30  # FIXE: Old compatibility reasons
    dfs.to_feather(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-folder", type=Path, default=DATA_FOLDER)
    parser.add_argument("--output-file", type=Path, default=OUTPUT_FILE)
    args = parser.parse_args()
    main(args.data_folder.resolve(), args.output_file)
