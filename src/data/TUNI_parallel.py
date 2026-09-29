from pathlib import Path

import mdfreader
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from signal_analysis import order_track_signal, signal_TSA


def _load_mf4(file_path):
    mdf_data = mdfreader.Mdf(file_path)
    channel_list = list(mdf_data.keys())
    channel_data = mdf_data[channel_list[0]]["data"]
    return channel_data


def process_test(PW, run, run_num):
    print(f"Processing run {run_num + 1} for gear {PW} ({run['folder']})")
    # PW, run, i = args

    # TSA
    sampling_rate = 5000  # Hz
    # sampling_rate_enc = 500  # Hz
    revolutions_per_TSA = 99
    step_size = 99
    num_samples_per_revolution = 200

    # Specify raw data
    data_folder = Path("/Volumes/T7-Aalto/TUNI-KONG_comb")

    status = run["status"]
    status_opposite = run["status_opposite"]
    position = run["position"]
    start, end = run["take"]

    # LOAD
    ##

    signal_1 = _load_mf4(
        data_folder / run["folder"] / f"TC{position}_accelerometer_1_{PW}.mf4"
    )
    signal_2 = _load_mf4(
        data_folder / run["folder"] / f"TC{position}_accelerometer_2_{PW}.mf4"
    )
    tacho = _load_mf4(data_folder / run["folder"] / f"Encoder.mf4")

    # ENCODER TO ANGLE
    ##

    # Ensure tacho signals are binary (some 2's mixed in for some reason)
    if not np.all((tacho == 0) | (tacho == 1)):
        print(
            f"Encoder signal not binary (0 or 1)! ({np.unique_counts(tacho)}) ({run['folder']})"
        )
        tacho = tacho.astype(bool).astype(np.int16)  # Convert to [1, 0]

    # Check that all peaks are single width (a few double peaks in some runs)
    if not all(tacho[1:] + tacho[:-1] < 1.5):
        print(f"Encoder signal has multiple True values per peak! ({run['folder']})")
        # If tacho has multiple non-zero values in a row, keep only the first one
        keep_mask = (tacho == 1) & np.concatenate(([True], tacho[:-1] == 0))
        filtered_tacho = np.zeros_like(tacho)
        filtered_tacho[keep_mask] = 1
        tacho = filtered_tacho

    # Get rev start indices
    crossing_idxs = np.where(np.diff(tacho) == 1)[0]
    # Cumulative angle at rev start indices
    angle = np.arange(len(crossing_idxs)) * 360
    # Fill in the angle at every sample point
    # `* 10` because encoder sampling rate is 500 Hz and accelerometer is 5000 Hz
    interpolated_angle = np.interp(
        np.arange(len(tacho) * 10), crossing_idxs * 10, angle
    )
    tacho = None  # Free memory

    # CUT
    ##

    # Cut to signal length
    # Slightly different lengths due to different sampling rates,
    # but close enough (especially taking into account very long measurement) that cutting to length shouldn't be a problem
    # signal_1 = signal_1[200:-200]
    # signal_2 = signal_2[200:-200]
    # interpolated_angle = interpolated_angle[200:]
    interpolated_angle = interpolated_angle[: len(signal_1)]

    # FIXME: MAYBE DO THIS AT START?
    # Runs generally contain unstable conditions at the beginning and end
    signal_1 = signal_1[int(start * len(signal_1)) : int(end * len(signal_1))]
    signal_2 = signal_2[int(start * len(signal_2)) : int(end * len(signal_2))]
    interpolated_angle = interpolated_angle[
        int(start * len(interpolated_angle)) : int(end * len(interpolated_angle))
    ]

    # PARTITION
    ##

    # Split into parts, because otherwise order tracking crashes the kernel
    # signal_1_parts = []
    # signal_2_parts = []
    # interpolated_angle_parts = []

    # for i in range(5):
    #     signal_1_parts.append(
    #         signal_1[int(i * 0.2 * len(signal_1)) : int((i + 1) * 0.2 * len(signal_1))]
    #     )
    #     signal_2_parts.append(
    #         signal_2[int(i * 0.2 * len(signal_2)) : int((i + 1) * 0.2 * len(signal_2))]
    #     )
    #     interpolated_angle_parts.append(
    #         interpolated_angle[
    #             int(i * 0.2 * len(interpolated_angle)) : int(
    #                 (i + 1) * 0.2 * len(interpolated_angle)
    #             )
    #         ]
    #     )

    # OT & TSA (signal OT method)
    ##

    all_signals_OT = []
    # for signal_parts in [signal_1_parts, signal_2_parts]:
    for signal in [signal_1, signal_2]:
        signal_samples_OT = []

        # for j in range(len(signal_parts)):
        for j in range(5):

            # OT
            # signal_OT = order_track_signal(
            #     signal_parts[j],
            #     np.arange(len(signal_parts[j])) / sampling_rate,
            #     interpolated_angle_parts[j],
            #     num_samples_per_revolution=num_samples_per_revolution,
            # )
            signal_view = signal[
                int(j * 0.2 * len(signal)) : int((j + 1) * 0.2 * len(signal))
            ]
            interpolated_angle_view = interpolated_angle[
                int(j * 0.2 * len(interpolated_angle)) : int(
                    (j + 1) * 0.2 * len(interpolated_angle)
                )
            ]
            signal_OT = order_track_signal(
                signal_view,
                np.arange(len(signal_view)) / sampling_rate,
                interpolated_angle_view,
                num_samples_per_revolution=num_samples_per_revolution,
            )

            # TSA
            signal_OT_TSA = signal_TSA(
                signal_OT,
                num_samples_per_revolution,
                revolutions_per_TSA,
                step_size,
            )

            # FFT
            # Drop first few and last in case of interpolation artifacts
            samples = np.abs(np.fft.rfft(signal_OT_TSA, norm="forward", axis=1))[
                1:-1,
                :,
            ]

            samples = samples.astype(np.float32)
            signal_samples_OT.extend(samples)

        all_signals_OT.append(signal_samples_OT)

    signal_1 = None
    signal_2 = None
    interpolated_angle = None

    # (num_signals, num_samples, sample_len)
    all_signals_OT = np.array(all_signals_OT)

    # Make a new row for the DF for every TSA sample
    expanded_rows = []
    for j in range(all_signals_OT.shape[1]):
        expanded_rows.append(
            {
                "gear": PW,
                "position": position,
                "run": run["folder"],
                "run_num": run_num,
                "speed": run["speed"],
                "load": run["load"],
                "class": status,
                "class_opposite": status_opposite,
                # "OT_method": "signal",
                # "TSA_size": revolutions_per_TSA,
                "sample_index": j,
                "acc1": all_signals_OT[0, j, :],
                "acc2": all_signals_OT[1, j, :],
            }
        )

    print(f"> Completed run {run_num + 1} for gear {PW} ({run['folder']})")

    # Make DF of one file
    return pd.DataFrame(expanded_rows)


if __name__ == "__main__":
    print("Starting TUNI parallel processing")

    runs_by_gear = {
        # "18990": [  # TODO
        #     {
        #         "folder": "20220628_18990_20254_test1",
        #         "position": 1,
        #         "status": "healthy",
        #     },
        #     {
        #         "folder": "20220629_18990_20254_test2",
        #         "position": 1,
        #         "status": "healthy",
        #     },
        #     {"folder": "20220818_18990_20254_test3", "position": 1, "status": "faulty"},
        #     # {"folder": "20220819_18990_20254_test4", "position": 1, "status": "faulty"}, # ! POWER OUTAGE
        #     {"folder": "20220822_18990_20254_test5", "position": 1, "status": "faulty"},
        #     # {"folder": "20220826_18990_20254_test6", "position": 1, "status": "faulty"}, # ! ENCODER BROKE
        #     # {"folder": "20220829_18990_20254_test7", "position": 1, "status": "faulty"}, # ! CORRUPT
        #     # {"folder": "20220831_18990_20254_test8", "position": 1, "status": "faulty"}, # ! CORRUPT
        #     #
        #     {"folder": "20220922_18990_20253_test1", "position": 1, "status": "faulty"},
        #     {"folder": "20220926_18990_20253_test2", "position": 1, "status": "faulty"},
        #     {"folder": "20220928_18990_20253_test3", "position": 1, "status": "faulty"},
        #     {"folder": "20220929_18990_20253_test4", "position": 1, "status": "faulty"},
        #     {"folder": "20220930_18990_20253_test5", "position": 1, "status": "faulty"},
        #     {"folder": "20221003_18990_20253_test6", "position": 1, "status": "faulty"},
        # ],
        "20252": [
            # {"folder": "20221012_20252_20253_test1", "position": 1, "status": "healthy"}, # ! Particle alarm
            # {"folder": "20221013_20252_20253_test2", "position": 1, "status": "healthy"}, # Particle alarm
            {
                "folder": "20221014_20252_20253_test3",
                "position": 1,
                "status": "healthy",
                "status_opposite": "healthy",
                "take": [0.3, 0.7],
                "speed": 1500,
                "load": 1000,
            },
            {
                "folder": "20221014_20252_20253_test3",
                "position": 1,
                "status": "healthy",
                "status_opposite": "healthy",
                "take": [0.82, 0.98],
                "speed": 1500,
                "load": 600,
            },
            {
                "folder": "20221017_20252_20253_test4",
                "position": 1,
                "status": "healthy",
                "status_opposite": "healthy",
                "take": [0.3, 0.9],
                "speed": 1500,
                "load": 1000,
            },
            {
                "folder": "20221021_20252_20253_test5",
                "position": 1,
                "status": "healthy",
                "status_opposite": "healthy",
                "take": [0.3, 0.75],
                "speed": 1500,
                "load": 1125,
            },
            # {"folder": "20221025_20252_20253_test6", "position": 1}, # NO ENCODER
            {
                "folder": "20221026_20252_20253_test7",
                "position": 1,
                "status": "P_TIFF",  # ! Pinion TFF
                "status_opposite": "healthy",
                "take": [0.3, 0.98],
                "speed": 1500,
                "load": 1250,
            },
        ],
        "20253": [
            # 18990
            {
                "folder": "20220922_18990_20253_test1",
                "position": 2,
                "status": "healthy",
                "status_opposite": "W_indication",
                "take": [0.4, 0.8],
                "speed": 1500,
                "load": 1000,
            },
            {
                "folder": "20220926_18990_20253_test2",
                "position": 2,
                "status": "healthy",
                "status_opposite": "W_indication",
                "take": [0.3, 0.8],
                "speed": 1500,
                "load": 1000,
            },
            {
                "folder": "20220928_18990_20253_test3",
                "position": 2,
                "status": "healthy",
                "status_opposite": "W_indication",
                "take": [0.3, 0.8],
                "speed": 1500,
                "load": 1000,
            },
            {
                "folder": "20220929_18990_20253_test4",
                "position": 2,
                "status": "healthy",
                "status_opposite": "W_indication",
                "take": [0.3, 0.8],
                "speed": 1500,
                "load": 1000,
            },
            {
                "folder": "20220930_18990_20253_test5",
                "position": 2,
                "status": "healthy",
                "status_opposite": "W_indication",
                "take": [0.3, 0.60],
                "speed": 1500,
                "load": 1000,
            },
            {
                "folder": "20221003_18990_20253_test6",
                "position": 2,
                "status": "healthy",
                "status_opposite": "W_TIFF",
                "take": [0.3, 0.8],
                "speed": 1500,
                "load": 1000,
            },
            # 20252
            # { # ! PARTICLE ALARM
            #     "folder": "20221012_20252_20253_test1",
            #     "position": 2,
            #     "status": "healthy",
            #     "status_opposite": "healthy",
            #     "speed": 1500,
            #     "take": [0.20, 0.8],
            # },
            # { # ! PARTICLE ALARM
            #     "folder": "20221013_20252_20253_test2",
            #     "position": 2,
            #     "status": "healthy",
            #     "status_opposite": "healthy",
            #     "speed": 1500,
            #     "take": [0.20, 0.8],
            # },
            {
                "folder": "20221014_20252_20253_test3",
                "position": 2,
                "status": "healthy",
                "status_opposite": "healthy",
                "take": [0.3, 0.7],
                "speed": 1500,
                "load": 1000,
            },
            {
                "folder": "20221014_20252_20253_test3",
                "position": 2,
                "status": "healthy",
                "status_opposite": "healthy",
                "take": [0.85, 0.98],
                "speed": 1500,
                "load": 600,
            },
            {
                "folder": "20221017_20252_20253_test4",
                "position": 2,
                "status": "healthy",
                "status_opposite": "healthy",
                "take": [0.3, 0.8],
                "speed": 1500,
                "load": 1000,
            },
            {
                "folder": "20221021_20252_20253_test5",
                "position": 2,
                "status": "healthy",
                "status_opposite": "W_pre_indication",
                "take": [0.3, 0.75],
                "speed": 1500,
                "load": 1125,
            },
            # {"folder": "20221025_20252_20253_test6", "position": 2}, # ! NO ENCODER
            {
                "folder": "20221026_20252_20253_test7",
                "position": 2,
                "status": "healthy",
                "status_opposite": "W_TIFF",
                "take": [0.3, 0.8],
                "speed": 1500,
                "load": 1250,
            },
            # 20289
            {
                "folder": "20221102_20289_20253_test1",
                "position": 2,
                "status": "healthy",
                "status_opposite": "W_pre_indication",
                "take": [0.3, 0.6],
                "speed": 1500,
                "load": 1250,
            },
            {
                "folder": "20221102_20289_20253_test1",
                "position": 2,
                "status": "healthy",
                "status_opposite": "W_pre_indication",
                "take": [0.8, 0.94],
                "speed": 1500,
                "load": 1250,
            },
            {
                "folder": "20221104_20289_20253_test2",
                "position": 2,
                "status": "healthy",
                "status_opposite": "W_TIFF",
                "take": [0.3, 0.8],
                "speed": 1500,
                "load": 1250,
            },
            # 20290
            {
                "folder": "20221116_20290_20253_test1",
                "position": 2,
                "status": "healthy",
                "status_opposite": "healthy",
                "take": [0.2, 0.7],
                "speed": 1500,
                "load": 1125,
            },
            {
                "folder": "20221116_20290_20253_test1",
                "position": 2,
                "status": "healthy",
                "status_opposite": "healthy",
                "take": [0.77, 0.97],
                "speed": 1500,
                "load": 650,
            },
            {
                "folder": "20221118_20290_20253_test2",
                "position": 2,
                "status": "healthy",
                "status_opposite": "W_pre_indication",
                "take": [0.3, 0.8],
                "speed": 1500,
                "load": 1125,
            },
            {
                "folder": "20221121_20290_20253_test3",
                "position": 2,
                "status": "W_pre_indication",
                "status_opposite": "W_TIFF",
                "take": [0.3, 0.8],
                "speed": 1500,
                "load": 1125,
            },
            # 20292
            {
                "folder": "20230313_20292_20253_test1",
                "position": 2,
                "status": "W_indication",
                "status_opposite": "healthy",
                "take": [0.3, 0.8],
                "speed": 1500,
                "load": 1000,
            },
            {
                "folder": "20230315_20292_20253_test2",
                "position": 2,
                "status": "W_indication",
                "status_opposite": "healthy",
                "take": [0.3, 0.8],
                "speed": 1500,
                "load": 1000,
            },
            {
                "folder": "20230316_20292_20253_test3",
                "position": 2,
                "status": "W_indication",
                "status_opposite": "healthy",
                "take": [0.3, 0.8],
                "speed": 1500,
                "load": 1000,
            },
            {
                "folder": "20230316_20292_20253_test3",
                "position": 2,
                "status": "W_indication",
                "status_opposite": "healthy",
                "take": [0.85, 0.9],
                "speed": 1500,
                "load": 600,
            },
            {
                "folder": "20230317_20292_20253_test4",
                "position": 2,
                "status": "W_TIFF",
                "status_opposite": "healthy",
                "take": [0.3, 0.915],
                "speed": 1500,
                "load": 1000,
            },
            {
                "folder": "20230317_20292_20253_test4",
                "position": 2,
                "status": "W_TIFF",
                "status_opposite": "healthy",
                "take": [0.93, 0.99],
                "speed": 1500,
                "load": 600,
            },
        ],
        "20254": [
            # {"folder": "20220628_18990_20254_test1", "position": 2, "status": "healthy"}, # ! FERROUS PARTICLES
            {
                "folder": "20220629_18990_20254_test2",
                "position": 2,
                "status": "healthy",
                "status_opposite": "W_indication",
                "take": [0.3, 0.8],
                "speed": 1500,
                "load": 1000,
            },
            {
                "folder": "20220818_18990_20254_test3",
                "position": 2,
                "status": "healthy",
                "status_opposite": "W_indication",
                "take": [0.3, 0.8],
                "speed": 1500,
                "load": 1000,
            },
            # {"folder": "20220819_18990_20254_test4", "position": 2, "status": "healthy"}, # ! POWER OUTAGE
            {
                "folder": "20220822_18990_20254_test5",
                "position": 2,
                "status": "healthy",
                "status_opposite": "W_indication",
                "take": [0.3, 0.8],
                "speed": 1500,
                "load": 1000,
            },
            # {"folder": "20220826_18990_20254_test6", "position": 2, "status": "maybe_faulty"}, # ! ENCODER BROKE
            # ! Corrupt
            # {
            #     "folder": "20220829_18990_20254_test7",
            #     "position": 2,
            #     "status": "faulty",
            #     "take": [0.25, 0.90],
            #     "speed": 0,  # ? Corrupted or speed off (probably corrupted)
            # },
            # ! Corrupt
            # {
            #     "folder": "20220831_18990_20254_test8",
            #     "position": 2,
            #     "status": "faulty",
            #     "take": [0.25, 0.90],
            #     "speed": 0,  # ? Corrupted or speed off (probably corrupted)
            # },
        ],
        "20289": [
            {
                "folder": "20221102_20289_20253_test1",
                "position": 1,
                "status": "W_pre_indication",
                "status_opposite": "healthy",
                "take": [0.25, 0.6],
                "speed": 1500,
                "load": 1250,
            },
            {
                "folder": "20221102_20289_20253_test1",
                "position": 1,
                "status": "W_pre_indication",
                "status_opposite": "healthy",
                "take": [0.75, 0.94],
                "speed": 1500,
                "load": 650,
            },
            {
                "folder": "20221104_20289_20253_test2",
                "position": 1,
                "status": "W_TIFF",
                "status_opposite": "healthy",
                "take": [0.3, 0.89],
                "speed": 1500,
                "load": 1250,
            },
        ],
        "20290": [
            {
                "folder": "20221116_20290_20253_test1",
                "position": 1,
                "status": "healthy",
                "status_opposite": "healthy",
                "take": [0.2, 0.7],
                "speed": 1500,
                "load": 1125,
            },
            {
                "folder": "20221116_20290_20253_test1",
                "position": 1,
                "status": "healthy",
                "status_opposite": "healthy",
                "take": [0.8, 0.97],
                "speed": 1500,
                "load": 650,
            },
            {
                "folder": "20221118_20290_20253_test2",
                "position": 1,
                "status": "W_pre_indication",
                "status_opposite": "healthy",
                "take": [0.3, 0.8],
                "speed": 1500,
                "load": 1125,
            },
            {
                "folder": "20221121_20290_20253_test3",
                "position": 1,
                "status": "W_TIFF",
                "status_opposite": "W_pre_indication",
                "take": [0.3, 0.9],
                "speed": 1500,
                "load": 1125,
            },
        ],
        "20291": [
            {
                "folder": "20230503_20292_20291_test1",
                "position": 2,
                "status": "healthy",
                "status_opposite": "W_pre_indication",
                "take": [0.3, 0.8],
                "speed": 1500,
                "load": 1000,
            },
            {
                "folder": "20230505_20292_20291_test2",
                "position": 2,
                "status": "healthy",
                "status_opposite": "W_TIFF",
                "take": [0.3, 0.66],
                "speed": 1500,
                "load": 1000,
            },
        ],
        "20292": [
            {
                "folder": "20230313_20292_20253_test1",
                "position": 1,
                "status": "healthy",
                "status_opposite": "W_indication",
                "take": [0.3, 0.8],
                "speed": 1500,
                "load": 1000,
            },
            {
                "folder": "20230315_20292_20253_test2",
                "position": 1,
                "status": "healthy",
                "status_opposite": "W_indication",
                "take": [0.3, 0.8],
                "speed": 1500,
                "load": 1000,
            },
            {
                "folder": "20230316_20292_20253_test3",
                "position": 1,
                "status": "healthy",
                "status_opposite": "W_indication",
                "take": [0.3, 0.75],
                "speed": 1500,
                "load": 1000,
            },
            {
                "folder": "20230317_20292_20253_test4",
                "position": 1,
                "status": "healthy",
                "status_opposite": "W_TIFF",
                "take": [0.3, 0.8],
                "speed": 1500,
                "load": 1000,
            },
            #
            {
                "folder": "20230503_20292_20291_test1",
                "position": 1,
                "status": "W_pre_indication",
                "status_opposite": "healthy",
                "take": [0.3, 0.8],
                "speed": 1500,
                "load": 1000,
            },
            {
                "folder": "20230505_20292_20291_test2",
                "position": 1,
                "status": "W_TIFF",
                "status_opposite": "healthy",
                "take": [0.3, 0.675],
                "speed": 1500,
                "load": 1000,
            },
        ],
    }

    # Go through files
    with Parallel(n_jobs=2, backend="loky", timeout=99999) as parallel:
        # for PW in ["20252", "20253", "20254", "20289", "20290", "20291", "20292"]:
        for PW in ["20252", "20253", "20290"]:
            print(f"Processing gear {PW}")

            results = parallel(
                delayed(process_test)(PW, r, i) for i, r in enumerate(runs_by_gear[PW])
            )

            # Combine files
            dfs = pd.concat(results)

            # SAVING
            output_path = (
                Path(__file__).parent.parent.parent.absolute()
                / "data"
                / f"TUNI_OT_{PW}.feather"
            )
            dfs.to_feather(output_path)
            print(f"Gear {PW} saved to {output_path}")
            dfs = None
