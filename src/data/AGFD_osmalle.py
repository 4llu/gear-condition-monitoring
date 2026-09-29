from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow as pa

# SETUP
#######

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
    "Mild_Micropitting_GP4": ("S", "micropitting"),
    "Severe_Micropitting_GP3": ("L", "micropitting"),
}

# Specify raw data
data_folder = Path.cwd().parent.parent.parent / "data" / "mendeley_AGFD3kHz"
files = data_folder.glob("**/*.csv")


# FAULTS
########


# Go through files
dfs = []
for f in files:
    # SPECS
    ##

    # Get measurement specifications from file path
    p = f.parts
    # print(p)

    rpm = int(p[6].replace("RPM", ""))
    torque = torque_map[p[7]]

    # XXX SKIP HEALTHY
    if p[8] == "Healthy":
        continue
    elif p[8] == "Faulty":
        installation = int(p[9][-1])
        level, fault = fault_map.get(
            p[10].replace(".csv", ""), (None, None)
        )  # Fault types we don't care about not in map
        GP = 0
    else:
        raise ValueError(f"Unknown type directory: {p[8]}")

    # Skip the faults we don't want
    if fault is None:
        continue

    # SIGNAL
    ##

    # Read CSV
    df = pd.read_csv(f, sep=",", index_col=0, header=0)

    # Installation 3 special case
    if installation == 3:
        # Reset index
        df = df.reset_index(
            drop=False  # ! Installation 3 has time as index. This is done to add separate index, but not drop time
        )

        # Cut first and last 20%, which has already been done for other installations
        df = df.iloc[int(len(df) * 0.2) : int(len(df) * 0.8)].reset_index(drop=True)

    print(str(f))

    # These sensors not used in this study
    df.drop(
        columns=[
            "time",
            "enc1_ang",
            "enc2_ang",
            "enc3_ang",
            "enc4_ang",
            "enc5_ang",
            # "acc1",
            # "Torq1",
            # "Torq2",
        ],
        inplace=True,
    )

    # Enc4 runs the wrong way in most files
    # if df["enc4_ang"].iloc[0] > df["enc4_ang"].iloc[1]:
    #     df["enc4_ang"] = -df["enc4_ang"]

    # Some measurements have glitches in time
    # df["time"] = fix_time(df["time"].to_numpy())

    # Sometimes last value is NaN, which causes problems later
    if any([np.isnan(df[s].iloc[-1]) for s in ["acc1", "acc2", "acc3", "acc4"]]):
        df = df.iloc[:-1].reset_index(drop=True)

    # Flip torques and standardize naming
    df = df.rename({"Torq2": "torque1", "Torq1": "torque2"}, axis="columns")

    # Get float 64 columns to flaot 32
    # * Conversion done because final computations are done with float32 anyway
    float64_cols = list(df.select_dtypes(include="float64"))
    df[float64_cols] = df[float64_cols].astype("float32")

    # Convert categoricals
    # dfs["severity"] = dfs["severity"].astype("category")
    # dfs["fault"] = dfs["fault"].astype("category")

    df["rpm"] = rpm
    df["torque"] = torque
    df["severity"] = level
    df["fault"] = fault
    df["installation"] = installation

    dfs.append(df)


# SAVE FAULTY RESULTS
print("Writing partition")

dfs = pd.concat(dfs)

dfs = dfs.reset_index(drop=True)
table = pa.Table.from_pandas(dfs)

partitions = ds.partitioning(
    pa.schema([("rpm", pa.int16()), ("installation", pa.int8())]), flavor="hive"
)
ds.write_dataset(
    table,
    Path.cwd().parent.parent.parent / "data" / "AGFD_osmalle",
    format="feather",
    partitioning=partitions,
    existing_data_behavior="overwrite_or_ignore",
)


# BASELINE
##########


# Go through files
dfs = []
for f in files:
    # SPECS
    ##

    # Get measurement specifications from file path
    p = f.parts
    # print(p)

    rpm = int(p[6].replace("RPM", ""))
    torque = torque_map[p[7]]

    if p[8] == "Healthy":
        installation = 0
        level = "-"
        GP = int(p[9][2])
        fault = "baseline"
    # XXX SKIP FAULTY
    elif p[8] == "Faulty":
        continue
    else:
        raise ValueError(f"Unknown type directory: {p[8]}")

    # Skip the faults we don't want
    if fault is None:
        continue

    # SIGNAL
    ##

    # Read CSV
    df = pd.read_csv(f, sep=",", index_col=0, header=0)

    print(str(f))

    # These sensors not used in this study
    df.drop(
        columns=[
            "time",
            "enc1_ang",
            "enc2_ang",
            "enc3_ang",
            "enc4_ang",
            "enc5_ang",
            # "acc1",
            # "Torq1",
            # "Torq2",
        ],
        inplace=True,
    )

    # Enc4 runs the wrong way in most files
    # if df["enc4_ang"].iloc[0] > df["enc4_ang"].iloc[1]:
    #     df["enc4_ang"] = -df["enc4_ang"]

    # Some measurements have glitches in time
    # df["time"] = fix_time(df["time"].to_numpy())

    # Sometimes last value is NaN, which causes problems later
    if any([np.isnan(df[s].iloc[-1]) for s in ["acc1", "acc2", "acc3", "acc4"]]):
        df = df.iloc[:-1].reset_index(drop=True)

    # Flip torques and standardize naming
    df = df.rename({"Torq2": "torque1", "Torq1": "torque2"}, axis="columns")

    # Get float 64 columns to flaot 32
    # * Conversion done because final computations are done with float32 anyway
    float64_cols = list(df.select_dtypes(include="float64"))
    df[float64_cols] = df[float64_cols].astype("float32")

    # Convert categoricals
    # dfs["severity"] = dfs["severity"].astype("category")
    # dfs["fault"] = dfs["fault"].astype("category")

    df["rpm"] = rpm
    df["torque"] = torque
    df["severity"] = GP
    df["fault"] = fault

    dfs.append(df)


# SAVE BASELINE RESULTS
print("Writing partition")

dfs = pd.concat(dfs)

dfs = dfs.reset_index(drop=True)
table = pa.Table.from_pandas(dfs)

partitions = ds.partitioning(pa.schema([("rpm", pa.int16())]), flavor="hive")

ds.write_dataset(
    table,
    Path.cwd().parent.parent.parent / "data" / "AGFD_baseline_osmalle",
    format="feather",
    partitioning=partitions,
    existing_data_behavior="overwrite_or_ignore",
)
