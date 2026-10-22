"""
UNSW gear crack processing that only writes the original format: the
magnitude spectrum of the one-revolution TSA of the order tracked signal (the
"spectrum" representation of data_processing_UNSW.py, which also writes the
envelope representations).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
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
    order_track_signal_from_encoder_pulses,
    signal_TSA2,
)

INCLUDE_NICOS = False  # Measurements already order tracked by Nico with Pietro's code

SAMPLING_RATE = 200_000  # Hz
SAMPLES_PER_REV = 27 * 48  # Samples per revolution in the order tracked data
# SAMPLES_PER_REV = 2700  # XXX
PINION_TEETH = 27
ENCODER_PULSES_PER_REV = 3600  # Input shaft encoder
USE_PRECISE_OT = True

# Revolutions averaged into each one-revolution TSA window, so also the
# revolutions one TSA window consumes
WINDOWS_PER_TSA = 40
# Spectrum bins to keep: the 10 first full GMFs and the DC component
SPECTRUM_BINS = 10 * PINION_TEETH
# Order of every kept bin
ORDERS = np.arange(SPECTRUM_BINS, dtype=float)

# Available revolutions per signal vary with speed (checked against the real
# data: ~102 at 5 Hz, ~119 at 2 Hz, ~208 at 10 Hz, ~313 at 15 Hz and ~418 at
# 20 Hz) and by 1-2 revolutions between Nico's precomputed OT signal and
# this script's own OT conversion of the same recording. A step size fixed
# per nominal speed can't track that per-signal difference, so -- same fix
# as data_processing_AGFD.py and data_processing_MCC5.py -- step_size is
# computed per signal below, from its own revolution count, aiming for
# TARGET_TSA_WINDOWS windows regardless of speed or which OT method produced
# it. 14 is the fewest windows that cover the longest signals (~419 revs)
# without gaps: (419 - WINDOWS_PER_TSA) / WINDOWS_PER_TSA + 1, which leaves
# ~95% of the revolutions in some window (the rest is each signal's tail).
# The 2 and 5 Hz signals then overlap ~80%, which balancing the window
# counts can't avoid.
TARGET_TSA_WINDOWS = 80

# Raw data (Nico's order tracked versions are in UNSW_gear_crack_Nico_OT
# next to it), and where the processed data is written
DATA_FOLDER = (
    Path(__file__).parent.parent.parent.parent.resolve() / "data" / "UNSW_gear_crack"
)
OUTPUT_FILE = Path(__file__).parent.parent.parent.resolve() / "data" / "UNSW_V6.feather"

print(OUTPUT_FILE)
quit()


COLUMNS = ["speed", "load", "severity", "OT_method", "signal", "freqs"]


def step_size_for_target_windows(n_revs: int, target: int = TARGET_TSA_WINDOWS) -> int:
    """Step size (in revolutions) that makes signal_TSA2 produce ~`target`
    windows from a signal with `n_revs` available revolutions, given
    WINDOWS_PER_TSA revolutions per window (see signal_TSA2's own docstring
    for how step_size and window count relate)."""
    return max(1, round((n_revs - WINDOWS_PER_TSA) / (target - 1)))


# HELPERS


def parse_labels(stem):
    """The labels of a recording from its file name, e.g. Data_05Hz_10Nm_M."""
    fname_parts = stem.split("_")
    return {
        "speed": int(fname_parts[1][:2]),
        "load": int(fname_parts[2][:2]),
        "severity": fname_parts[3],
    }


def tsa_spectra(signal_OT):
    """
    Magnitude spectra of the TSA windows of an order tracked signal, as
    float32 (what Pytorch expects): shape (num_windows, SPECTRUM_BINS).
    step_size is computed from this signal's own revolution count (see
    step_size_for_target_windows) rather than looked up by nominal speed.
    """
    n_revs_available = len(signal_OT) // SAMPLES_PER_REV
    step_size = step_size_for_target_windows(n_revs_available)
    signal_OT_TSA = signal_TSA2(
        signal_OT,
        SAMPLES_PER_REV,
        1,  # One-revolution windows
        WINDOWS_PER_TSA,
        step_size,
    )
    print("TSA", signal_OT_TSA.shape)

    return np.abs(np.fft.rfft(signal_OT_TSA, norm="forward", axis=1))[
        :, :SPECTRUM_BINS
    ].astype(np.float32)


def nico_spectra(path):
    """The spectra of Nico's order tracked version of a recording, None
    when there is none."""
    # e.g. UNSW_gear_crack/Data_05Hz_10Nm_M.mat ->
    # UNSW_gear_crack_Nico_OT/OT_Data_05Hz_10Nm_M.mat
    nico_f = path.parent.parent / f"{path.parent.name}_Nico_OT" / f"OT_{path.name}"
    if not nico_f.exists():
        print(f"{nico_f} does not exist in Nico's data, skipping...")
        return None

    mat = scipy.io.loadmat(nico_f)
    signal_OT = mat["signal_order_track"][0]
    # Get average torque
    # torqueAvg = mat["torqueAvg"][0][0] # NOTE: Not used
    return tsa_spectra(signal_OT)


def order_track(signal, enc_signal, time):
    """Order track a signal with its recording's input shaft encoder."""
    if USE_PRECISE_OT:
        return order_track_signal_from_encoder_pulses(
            signal,
            enc_signal,
            ENCODER_PULSES_PER_REV,
            time,
            SAMPLES_PER_REV,
        )
    # Prep encoder signal for order tracking
    crossing_idxs = np.where(np.logical_and(enc_signal[:-1] <= 0, enc_signal[1:] > 0))[
        0
    ]
    crossings_angles = np.arange(len(crossing_idxs)) * (360 / ENCODER_PULSES_PER_REV)
    enc_angle = np.interp(np.arange(len(enc_signal)), crossing_idxs, crossings_angles)
    return order_track_signal(
        signal, time, enc_angle, num_samples_per_revolution=SAMPLES_PER_REV
    )


def signal_spectra(path, speed):
    """The spectra of this script's own order tracked version of a
    recording, None for Yaxin's recordings (which don't have the signals
    this uses)."""
    non_OT_mat = scipy.io.loadmat(path)

    # Normal
    try:
        signal = non_OT_mat["signals"][:, 4]  # acceleration
        enc_signal = non_OT_mat["signals"][:, 1]  # Input shaft encoder
    # Yaxin's
    except KeyError:
        print(f"{path} is Yaxin's data, skipping...")
        return None

    time = np.arange(len(signal)) / SAMPLING_RATE

    # Anti-aliasing filter
    antialiasing_cutoff = calc_antialiasing_cutoff(
        speed,
        pulses_per_rev=SAMPLES_PER_REV,
        order_of_interest=10 * PINION_TEETH,
        raw_sample_rate_hz=SAMPLING_RATE,
        order_margin_factor=1.3,
        nyquist_safety_factor=0.9,
    )
    sos = butter(
        4, antialiasing_cutoff, btype="lowpass", output="sos", fs=SAMPLING_RATE
    )
    signal = sosfiltfilt(sos, signal)

    signal_OT = order_track(signal, enc_signal, time)

    # Remove first and last revolutions to remove any weird artifacts
    signal_OT = signal_OT[SAMPLES_PER_REV:-SAMPLES_PER_REV]

    return tsa_spectra(signal_OT)


def process_recording(path, labels):
    """One row per TSA window of a recording, Nico's order tracked version's
    windows first."""
    spectra = {}
    if INCLUDE_NICOS:
        spectra["nico"] = nico_spectra(path)
    spectra["signal"] = signal_spectra(path, labels["speed"])

    return [
        [labels["speed"], labels["load"], labels["severity"], OT_method, sample, ORDERS]
        for OT_method, samples in spectra.items()
        if samples is not None
        for sample in samples
    ]


# PROCESSING


def main(data_folder=DATA_FOLDER, output_file=OUTPUT_FILE):
    print(f"Loading UNSW gear crack data from {data_folder}...")
    # Sorted, so the output's row order doesn't depend on the filesystem
    files = sorted(data_folder.glob("*.mat"))

    rows = []
    for f in files:
        print(f)
        rows.extend(process_recording(f, parse_labels(f.stem)))

    df = pd.DataFrame(data=rows, columns=COLUMNS)
    print(len(df))
    df["TSA_size"] = 40  # FIXE: Old compatibility reasons
    df.to_feather(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-folder", type=Path, default=DATA_FOLDER)
    parser.add_argument("--output-file", type=Path, default=OUTPUT_FILE)
    args = parser.parse_args()
    main(args.data_folder.resolve(), args.output_file)
