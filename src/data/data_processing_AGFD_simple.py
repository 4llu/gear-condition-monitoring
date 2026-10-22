"""
AGFD processing that only writes the original format: the magnitude spectrum
of the TSA of the order tracked signal (the "spectrum" representation of
data_processing_AGFD.py, which also writes the envelope representations).

Data in Brief journal article: https://www.sciencedirect.com/science/article/pii/S2352340924011338
Data source: https://data.mendeley.com/datasets/fywnj597d8/2

Notes:
`acc2` in installation 3 are pure noice, as the sensor is damaged. DO NOT USE!
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
    order_track_signal_from_encoder_angle,
    signal_TSA2,
)

SAMPLING_RATE = 3012  # Hz
SAMPLES_PER_REV = 360  # Samples per revolution in the order tracked data
PINION_TEETH = 15
# The pinion turns at a third of the motor speed, which the directories give
MOTOR_TO_PINION_RATIO = 3
# Highest order the anti-aliasing filter keeps clean: the 9th GMF
ORDER_OF_INTEREST = 9 * PINION_TEETH
# AGFD has no raw encoder pulse train -- enc4_ang is already a continuous,
# per-sample angle channel -- so the precise path uses every enc4_ang sample
# directly (order_track_signal_from_encoder_angle) rather than detecting
# pulse crossings the way order_track_signal_from_encoder_pulses would.
USE_PRECISE_OT = True

# Revolutions averaged into each one-revolution TSA window, so also the
# revolutions one TSA window consumes
WINDOWS_PER_TSA = 60
# Spectrum bins to keep: the 10 first full GMFs and the DC component
SPECTRUM_BINS = 10 * PINION_TEETH

# Available revolutions per file vary a lot -- not just with nominal speed
# (higher RPM means more revolutions in the same wall-clock recording), but
# also between healthy (~110s) and faulty (~230s) recordings at the SAME
# speed, and even file-to-file at a fixed (speed, condition) due to real
# speed-control variation. A step size fixed per nominal speed can't balance
# both of those at once (e.g. faulty runs need roughly 2x the healthy step
# size at the same RPM just to land on the same window count). So step_size
# is computed per file below, from that file's own revolution count, aiming
# for TARGET_TSA_WINDOWS windows regardless of speed or healthy/faulty.
# 66 is the fewest windows that cover the longest files (~1978 revs) without
# gaps: (1978 - WINDOWS_PER_TSA) / WINDOWS_PER_TSA + 1, which leaves ~99% of
# every file's revolutions in some window. The shortest files (~300 revs)
# then overlap heavily, which balancing the window counts can't avoid.
TARGET_TSA_WINDOWS = 66


def step_size_for_target_windows(n_revs: int, target: int = TARGET_TSA_WINDOWS) -> int:
    """Step size (in revolutions) that makes signal_TSA2 produce ~`target`
    windows from a file with `n_revs` available revolutions, given
    WINDOWS_PER_TSA revolutions per window (see signal_TSA2's own docstring
    for how step_size and window count relate)."""
    return max(1, round((n_revs - WINDOWS_PER_TSA) / (target - 1)))


torque_map = {
    "0_12Nm": 1,
    "0_71Nm": 6,
    "1_31Nm": 11,
}

# S - Small, L - Large
fault_map = {
    "Mild_Wear_GP7": ("S", "wear"),
    "Severe_Wear_GP2": ("L", "wear"),
    "Mild_TFF_GP9": ("S", "crack"),
    "Severe_TFF_GP5": ("L", "crack"),
    "Mild_Pitting_GP1": ("S", "pitting"),
    "Severe_Pitting_GP6": ("L", "pitting"),
}


SENSORS = ["acc2", "acc3", "acc4"]
# These sensors not used in this study
UNUSED_COLUMNS = [
    "enc1_ang",
    "enc2_ang",
    "enc3_ang",
    "enc5_ang",
    "acc1",
    "Torq1",
    "Torq2",
]

# Raw data, and where the processed data is written
DATA_FOLDER = (
    Path(__file__).parent.parent.parent.parent.resolve() / "data" / "mendeley_AGFD3kHz"
)
# TSALEN1: one-revolution windows, the name configs with
# revs_per_TSA_window 1 look for
OUTPUT_FILE = Path(__file__).parent.parent.parent / "data" / "AGFD_V6.feather"

# HELPERS


def fix_time(X):
    """Repair timestamps that jump backwards or repeat (which would break the
    order tracking's spline): shift everything from the glitch on so it
    continues one sample period after the previous timestamp."""
    increment = 1 / SAMPLING_RATE
    loc = np.where(X[1:] <= X[:-1])[0] + 1  # + 1 needed to get the right index

    if len(loc) == 0:
        return X

    for l in loc:
        wrong = X[l - 1] - X[l]
        print(f"Wrong time at {l}")
        diff = np.concatenate([
            np.zeros(l),
            np.ones(len(X) - l) * (increment + wrong),
        ])
        X = X + diff

    return X


def parse_labels(parts):
    """
    The labels of a recording from its path inside the data folder:
    <speed>RPM/<torque>/Healthy/GP<n>.csv or
    <speed>RPM/<torque>/Faulty/Inst_<n>/<fault>.csv
    None for the recordings that are left out (250 RPM, and the faults not
    in fault_map).
    """
    speed = int(parts[0].replace("RPM", ""))
    torque = torque_map[parts[1]]

    if speed == 250:
        return None

    if parts[2] == "Healthy":
        installation = 0
        level = "-"
        GP = int(parts[3][2])
        fault = "healthy"
    elif parts[2] == "Faulty":
        installation = int(parts[3][-1])
        # Fault types we don't care about not in map
        level, fault = fault_map.get(parts[4].replace(".csv", ""), (None, None))
        GP = 0
    else:
        raise ValueError(f"Unknown type directory: {parts[2]}")

    # Skip the faults we don't want
    if fault is None:
        return None

    return {
        "speed": speed,
        "load": torque,
        "installation": installation,
        "severity": level,
        "healthy_GP": GP,
        "class": fault,
    }


def load_recording(path, installation):
    """A recording's time, enc4_ang and accelerometer columns, cleaned up."""
    df = pd.read_csv(path, sep=",", index_col=0, header=0)

    # Installation 3 special case
    if installation == 3:
        # Reset index
        df = df.reset_index(
            drop=False  # ! Installation 3 has time as index. This is done to add separate index, but not drop time
        )

        # Cut first and last 20%, which has already been done for other installations
        df = df.iloc[int(len(df) * 0.2) : int(len(df) * 0.8)].reset_index(drop=True)

    df = df.drop(columns=UNUSED_COLUMNS)

    # Enc4 runs the wrong way in most files
    if df["enc4_ang"].iloc[0] > df["enc4_ang"].iloc[1]:
        df["enc4_ang"] = -df["enc4_ang"]

    # Some measurements have glitches in time
    df["time"] = fix_time(df["time"].to_numpy())

    # Sometimes last value is NaN, which causes problems later
    if any(np.isnan(df[s].iloc[-1]) for s in SENSORS):
        df = df.iloc[:-1].reset_index(drop=True)

    # A NaN anywhere else would turn the whole filtered and order tracked
    # signal into NaNs without an error
    used_columns = ["time", "enc4_ang", *SENSORS]
    nan_counts = df[used_columns].isna().sum()
    assert not nan_counts.any(), (
        f"NaNs in {path}: {nan_counts[nan_counts > 0].to_dict()}"
    )

    return df


def antialiasing_filter(speed):
    """The anti-aliasing low-pass filter of a recording at `speed` RPM, the
    same for every sensor of the recording."""
    pinion_speed_hz = speed / 60 / MOTOR_TO_PINION_RATIO
    antialiasing_cutoff = calc_antialiasing_cutoff(
        pinion_speed_hz,
        pulses_per_rev=SAMPLES_PER_REV,
        order_of_interest=ORDER_OF_INTEREST,
        raw_sample_rate_hz=SAMPLING_RATE,
        order_margin_factor=1.15,
        nyquist_safety_factor=0.9,
    )
    return butter(
        4, antialiasing_cutoff, btype="lowpass", output="sos", fs=SAMPLING_RATE
    )


def order_track(signal, df):
    """Order track a signal with its recording's encoder."""
    if USE_PRECISE_OT:
        # enc4_ang already gives angle at every sample (not sparse
        # pulses), so use all of it directly instead of collapsing to
        # one crossing per revolution.
        return order_track_signal_from_encoder_angle(
            signal,
            df["enc4_ang"].to_numpy(),
            df["time"].to_numpy(),
            num_samples_per_revolution=SAMPLES_PER_REV,
        )
    return order_track_signal(
        signal,
        df["time"].to_numpy(),
        df["enc4_ang"].to_numpy(),
        num_samples_per_revolution=SAMPLES_PER_REV,
    )


def tsa_spectra(raw_signal, df, sos):
    """
    Magnitude spectra of the TSA windows of one sensor's signal, as float32
    (what Pytorch expects): shape (num_windows, SPECTRUM_BINS).
    """
    signal = sosfiltfilt(sos, raw_signal)

    signal_OT = order_track(signal, df)

    # Remove firstand last revolutions to remove any weird artifacts
    signal_OT = signal_OT[SAMPLES_PER_REV:-SAMPLES_PER_REV]

    # TSA the signal. step_size is computed from this file's own
    # revolution count (see step_size_for_target_windows) rather than
    # looked up by nominal speed, so healthy/faulty and file-to-file
    # length differences don't throw off the target window count.
    n_revs_available = len(signal_OT) // SAMPLES_PER_REV
    step_size = step_size_for_target_windows(n_revs_available)
    signal_OT_TSA = signal_TSA2(
        signal_OT,
        SAMPLES_PER_REV,
        1,  # One-revolution windows
        WINDOWS_PER_TSA,
        step_size,
    )
    print("Signal OT", signal_OT_TSA.shape)

    samples = np.abs(np.fft.rfft(signal_OT_TSA, norm="forward", axis=1))[
        :, :SPECTRUM_BINS
    ].astype(np.float32)
    print("spectrum", samples.shape)
    return samples


def process_recording(path, labels):
    """One row per TSA window of a recording: its labels and the spectrum
    of every sensor."""
    df = load_recording(path, labels["installation"])
    sos = antialiasing_filter(labels["speed"])
    sensor_samples = [tsa_spectra(df[sensor].to_numpy(), df, sos) for sensor in SENSORS]

    return pd.DataFrame([
        {
            **labels,
            "OT_method": "signal",
            **{sensor: sensor_samples[i][j] for i, sensor in enumerate(SENSORS)},
        }
        for j in range(sensor_samples[0].shape[0])
    ])


# PROCESSING


def main(data_folder=DATA_FOLDER, output_file=OUTPUT_FILE):
    # Sorted, so the output's row order doesn't depend on the filesystem
    files = sorted(data_folder.glob("**/*.csv"))

    dfs = []
    for f in files:
        labels = parse_labels(f.relative_to(data_folder).parts)
        if labels is None:
            continue
        print(str(f))
        dfs.append(process_recording(f, labels))

    # Combine files
    dfs = pd.concat(dfs)
    dfs["TSA_size"] = 40  # FIXE: Old compatibility reasons
    dfs.to_feather(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-folder", type=Path, default=DATA_FOLDER)
    parser.add_argument("--output-file", type=Path, default=OUTPUT_FILE)
    args = parser.parse_args()
    main(args.data_folder.resolve(), args.output_file)
