import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from matplotlib import pyplot as plt
from scipy.stats import mode

from src.data.preprocessing_individual import get_important_frequencies
from src.data.signal_analysis import fold
from src.models.models import setup_model
from src.utils.init import setup_device

# CONFIG
###

# SETUP

UNSW_teeth = 27
AGFD_teeth = 15
MCC5_teeth = 36

embedding_columns = ["x", "y"]

embedding_specs = {
    "normalize_embeddings": False,
    # "normalize_embeddings": True,  # XXX
    "average_anchors": True,
    # "average_batch": False,  # XXX
    "average_batch": True,
    "drop_anchor_outliers": 0,
    # "drop_batch_outliers": 0,
    "drop_batch_outliers": 5,  # XXX
}

config = {
    "use_difference": True,
    "gear_harmonics_to_include": 8,
    "gear_sideband_block_size": 5,
    # "gear_sideband_block_size": 3,  # XXX
    "embedding_len": 2,
    "model": "embedding",
    "backbone": "MlpMixer",
    "model_args": {
        "intermediate_L2": False,
        # "intermediate_L2": True,  # XXX
        # NORMAL
        # "num_layers": 4,
        # "hidden_dim": 21,
        # "tokens_mlp_dim": 16,
        # "channels_mlp_dim": 42,
        # LARGER HIDDEN
        "num_layers": 3,
        "hidden_dim": 32,
        "tokens_mlp_dim": 16,
        "channels_mlp_dim": 64,
        # TUNI
        # "num_layers": 4,
        # "hidden_dim": 18,
        # "tokens_mlp_dim": 16,
        # "channels_mlp_dim": 18,
    },
    "dropout": 0.0,
    "loss": "triplet",
    "loss_args": {
        "margin": 1.4,  # Doesn't matter during inference
        "miner": False,  # Doesn't matter during inference
    },
    "similarity": "euclidean",  # Doesn't matter during inference
    "lpnorm_embeddings_training": False,  # All of these are False to do them manually
    "center_embeddings_inference": False,
    "lpnorm_embeddings_inference": False,
}

device = setup_device(override="cpu")

model = setup_model(config, device)
model.eval()

# Trained model

model_weight_dir = Path.cwd().resolve() / "model_weights"
ensemble_weight_dir_name = "<FILL_ME>"


# HELPERS
###


def remove_anchor_outliers(embeddings, num_outliers=5):
    filtered_embeddings = []
    # Calculate mean embedding
    mean_embedding = torch.mean(embeddings, dim=1, keepdim=True)

    # Calculate distances from each anchor to the mean
    distances = torch.norm(embeddings - mean_embedding, dim=2)

    # Find indices of closest anchors
    for i in range(distances.shape[0]):
        _, closest_indices = torch.topk(
            distances[i], distances.shape[1] - num_outliers, largest=False
        )
        filtered_embeddings.append(embeddings[i][closest_indices, :])

    filtered_embeddings = torch.stack(filtered_embeddings, dim=0)

    return filtered_embeddings


def remove_batch_outliers(embeddings, num_outliers=5):
    # Calculate mean embedding
    mean_embedding = torch.mean(embeddings, dim=0, keepdim=True)

    # Calculate distances from each anchor to the mean
    distances = torch.norm(embeddings - mean_embedding, dim=1)

    # Find indices of closest anchors
    _, closest_indices = torch.topk(
        distances, distances.shape[0] - num_outliers, largest=False
    )
    return embeddings[closest_indices, :]


def embed(
    samples,
    embedding_model,
    normalize_embeddings=False,
    average_anchors=False,
    average_batch=False,
    drop_anchor_outliers=0,
    drop_batch_outliers=0,
):
    assert not average_batch or average_anchors, (
        "Must average anchors if averaging batch"
    )

    assert drop_batch_outliers == 0 or average_anchors, (
        "Must average anchors if dropping batch outliers"
    )

    samples = torch.tensor(samples, device=device)

    embeddings = embedding_model(samples)

    if drop_anchor_outliers > 0:
        embeddings = remove_anchor_outliers(
            embeddings, num_outliers=drop_anchor_outliers
        )

    if average_anchors:
        embeddings = reduce(embeddings, "b n f -> b f", reduction="mean")

    if drop_batch_outliers > 0:
        embeddings = remove_batch_outliers(embeddings, num_outliers=drop_batch_outliers)

    if average_batch:
        embeddings = reduce(embeddings, "b f -> 1 f", reduction="mean")

    if normalize_embeddings:
        embeddings = F.normalize(embeddings, p=2, dim=-1)

    embeddings = embeddings.view(-1, embeddings.shape[-1])
    return embeddings.detach().numpy()


def embed_wrapper(samples, anchors, embedding_model, config, embedding_specs):
    if config["use_difference"]:
        samples = samples[:, np.newaxis, :] - anchors[np.newaxis, :, :]
    else:
        samples = samples[:, np.newaxis, :]

    embeddings = embed(samples, embedding_model, **embedding_specs)
    return pd.DataFrame(embeddings, columns=["x", "y"])


def get_samples_UNSW(df, speed, load, severity):
    samples = df[
        (df["speed"] == speed) & (df["load"] == load) & (df["severity"] == severity)
    ]
    samples = np.array(list(samples["signal"]))
    samples = samples[
        :,
        get_important_frequencies(
            UNSW_teeth,
            config["gear_harmonics_to_include"],
            config["gear_sideband_block_size"],
        ),
    ]
    samples = np.log10(samples)

    return samples


def get_samples_MCC5(df, speed, load, fault_class, severity, circulation=None):
    samples = df[
        (df["speed"] == speed)
        & (df["load"] == load)
        & (df["severity"] == severity)
        & (df["class"] == fault_class)
    ]

    # If None, take both torque and speed circulations
    if circulation is not None:
        samples = samples[(samples["circulation"] == circulation)]

    samples = np.array(list(samples["signal"]))
    samples = samples[
        :,
        get_important_frequencies(
            MCC5_teeth,
            config["gear_harmonics_to_include"],
            config["gear_sideband_block_size"],
        ),
    ]
    samples = np.log10(samples)

    return samples


def get_samples_AGFD(
    df, speed, load, fault_class, severity, installation=None, healthy_GP=None
):
    samples = df[
        (df["speed"] == speed)
        & (df["load"] == load)
        & (df["severity"] == severity)
        & (df["class"] == fault_class)
    ]

    if installation is not None:
        samples = samples[(samples["installation"] == installation)]

    if healthy_GP is not None:
        samples = samples[(samples["healthy_GP"] == healthy_GP)]

    samples = np.array(list(samples["signal"]))
    samples = samples[
        :,
        get_important_frequencies(
            AGFD_teeth,
            config["gear_harmonics_to_include"],
            config["gear_sideband_block_size"],
        ),
    ]
    samples = np.log10(samples)

    return samples


# DATA
###

# UNSW

# Load data
UNSW_crack_df = pd.read_feather(Path.cwd() / "data" / "UNSW_gear_crack_OT_V2.feather")

# Change S crack to second healthy
UNSW_crack_severity_fix_map = {"H": "H1", "S": "H2"}
UNSW_crack_df["severity"] = UNSW_crack_df["severity"].transform(
    lambda x: UNSW_crack_severity_fix_map.get(x, x)
)

# Need only one OT method
# UNSW_crack_df = UNSW_crack_df[UNSW_crack_df["OT_method"] == "nico"]
UNSW_crack_df = UNSW_crack_df[UNSW_crack_df["OT_method"] == "signal"]

# Drop unnecessary columns
UNSW_crack_df.drop(
    columns=["OT_method", "TSA_size"],
    inplace=True,
)

UNSW_crack_df[["speed", "load", "severity"]].value_counts().sort_index()

# MCC5

# Load data
MCC5_df = pd.read_feather(Path.cwd() / "data" / "MCC5-THU_OT_V3.feather")  # XXX
# MCC5_df = pd.read_feather(Path.cwd() / "data" / "MCC5-THU_OT_V2.feather")

# Need only one OT method
MCC5_df = MCC5_df[MCC5_df["OT_method"] == "signal"]
# Should only be 30 in this file, but just in case
MCC5_df = MCC5_df[MCC5_df["TSA_size"] == 30]

# Drop unnecessary columns
MCC5_df.drop(
    columns=["OT_method", "TSA_size"],
    inplace=True,
)
# Only use gearbox_vibration_x sensor
MCC5_df = MCC5_df[
    ["speed", "load", "severity", "class", "circulation", "gearbox_vibration_x"]  # XXX
]
MCC5_df.rename(columns={"gearbox_vibration_x": "signal"}, inplace=True)  # XXX

# Drop in-between speeds and loads
MCC5_df = MCC5_df[MCC5_df["speed"].isin([1000, 2000, 3000])]
MCC5_df = MCC5_df[MCC5_df["load"].isin([10, 20])]

# Keep only crack, wear, and healthy
MCC5_df = MCC5_df[MCC5_df["class"].isin(["crack", "wear", "healthy"])]
# MCC5_df = MCC5_df[MCC5_df["class"].isin(["broken_tooth", "wear", "healthy"])]  # XXX
# MCC5_df["class"] = MCC5_df["class"].map(
#     {
#         "broken_tooth": "crack",
#         "wear": "wear",
#         "healthy": "healthy",
#     }
# )

MCC5_df[["speed", "load", "severity", "class"]].value_counts().sort_index()

# AGFD

# Load data
AGFD_df = pd.read_feather(Path.cwd() / "data" / "AGFD_OT_V2.feather")


# Need only one OT method
AGFD_df = AGFD_df[AGFD_df["OT_method"] == "signal"]
# Should only be 40 in this file, but just in case
AGFD_df = AGFD_df[AGFD_df["TSA_size"] == 40]

# # Drop unnecessary columns
AGFD_df.drop(
    columns=["OT_method", "TSA_size", "acc2", "acc3"],
    inplace=True,
)
# Use only Acc4
AGFD_df.rename(columns={"acc4": "signal"}, inplace=True)

# Drop 250 RPM data
AGFD_df = AGFD_df[AGFD_df["speed"] != 250]


AGFD_df[
    ["speed", "load", "class", "severity", "healthy_GP", "installation"]
].value_counts().sort_index()
# AGFD_df.value_counts().sort_index()

# RESULTS
###

# Embed UNSW


# ensemble_checkpoint_num = 15
ensemble_checkpoint_num = 5
ensemble_size = 5

weight_files = sorted(
    glob.glob(
        str(
            model_weight_dir
            / ensemble_weight_dir_name
            / "**"
            / f"{ensemble_checkpoint_num}.pth"
        )
    )
)
print(
    f"Found weight files {len(weight_files)} ({len(weight_files) / ensemble_size} ensembles)."
)

UNSW_ensemble_emb_dfs = []
for ensemble_repeat_num in range(len(weight_files) // ensemble_size):
    ensemble_weight_files = weight_files[
        ensemble_repeat_num * ensemble_size : (ensemble_repeat_num + 1) * ensemble_size
    ]

    for wf in ensemble_weight_files:
        # Load ensemble models
        model.backbone.load_state_dict(
            torch.load(
                wf,
                weights_only=True,
            )
        )
        model.eval()

        # Embeddings for one ensemble model
        all_emb_dfs = []
        for speed in [5, 10, 15, 20]:
            for load in [5, 10, 15]:
                # ANCHOR
                #

                if config["use_difference"]:
                    # * Use 16 first samples for anchors. Slowest speed has 32 samples, so it's 50% of that.
                    # * 16 from each to have consistent number of anchor samples.
                    anchor_samples = get_samples_UNSW(UNSW_crack_df, speed, load, "H1")[
                        :16
                    ]

                    anchor_samples = reduce(anchor_samples, "b w -> 1 w", "mean")
                    # anchor_samples = rearrange(anchor_samples, "(a b) w -> a b w", a=4, b=4)
                    # anchor_samples = reduce(anchor_samples, "a b w -> a w", a=4, b=4)

                # HEALTHY 1
                #

                # * Drop 16 first from H1, as those are used for anchors
                healthy_samples_1 = get_samples_UNSW(UNSW_crack_df, speed, load, "H1")[
                    16:
                ]
                healthy_emb_df_1 = embed_wrapper(
                    healthy_samples_1, anchor_samples, model, config, embedding_specs
                )
                healthy_emb_df_1["class"] = "healthy"
                healthy_emb_df_1["severity"] = "H1"

                # HEALTHY 2
                #

                healthy_samples_2 = get_samples_UNSW(UNSW_crack_df, speed, load, "H2")
                healthy_emb_df_2 = embed_wrapper(
                    healthy_samples_2, anchor_samples, model, config, embedding_specs
                )
                healthy_emb_df_2["class"] = "healthy"
                healthy_emb_df_2["severity"] = "H2"

                # CRACK 1
                #

                crack_samples_1 = get_samples_UNSW(UNSW_crack_df, speed, load, "M")
                crack_emb_df_1 = embed_wrapper(
                    crack_samples_1, anchor_samples, model, config, embedding_specs
                )
                crack_emb_df_1["class"] = "crack"
                crack_emb_df_1["severity"] = "M"

                # CRACK 2
                #

                crack_samples_2 = get_samples_UNSW(UNSW_crack_df, speed, load, "L")
                crack_emb_df_2 = embed_wrapper(
                    crack_samples_2, anchor_samples, model, config, embedding_specs
                )
                crack_emb_df_2["class"] = "crack"
                crack_emb_df_2["severity"] = "L"

                # Combine all dataframes and add back operating condition info
                emb_df = pd.concat(
                    [
                        healthy_emb_df_1,
                        healthy_emb_df_2,
                        crack_emb_df_1,
                        crack_emb_df_2,
                    ],
                    ignore_index=True,
                )
                emb_df["speed"] = speed
                emb_df["load"] = load

                all_emb_dfs.append(emb_df)

        all_emb_dfs = pd.concat(all_emb_dfs, ignore_index=True)
        UNSW_ensemble_emb_dfs.append(all_emb_dfs)

# UNSW accuracy evaluation

# 1 severity for prototypes
# 1 severity for query

all_accuracies = []
# for offset in range(10):
for offset in range(len(UNSW_ensemble_emb_dfs) // ensemble_size):  # XXX
    UNSW_accuracies = []
    # For every operating conditions (consisting of speed and load)
    # for speed in [10]:
    #     for load in [5]:
    for speed in [5, 10, 15, 20]:
        for load in [5, 10, 15]:
            for support_severity, query_severity in [("M", "L"), ("L", "M")]:
                # Repeat with every ensemble model
                ensemble_distances = []
                for e in range(5):
                    round_df = UNSW_ensemble_emb_dfs[offset * 5 + e]  # XXX
                    round_df = round_df[
                        (round_df["speed"] == speed) & (round_df["load"] == load)
                    ]

                    # Get possible prototypes

                    healthy_prototype_H1 = (
                        round_df[round_df["severity"] == "H1"][["x", "y"]]
                        .to_numpy()
                        .mean(axis=0)
                    )

                    crack_prototype = (
                        round_df[round_df["severity"] == support_severity][["x", "y"]]
                        .to_numpy()
                        .mean(axis=0)
                    )

                    # Use everything else for query
                    # (Not H1 [used for anchor] and not samples from operating condition used for prototypes)
                    query_df = round_df[
                        round_df["severity"].isin([query_severity, "H2"])
                    ]

                    # Convert severities to target labels [0, 1]
                    targets = (
                        query_df["severity"].map({"H2": 0, "M": 1, "L": 1}).to_numpy()
                    )

                    query_embs = query_df[["x", "y"]].to_numpy()

                    # XXX
                    # Center embeddings
                    # proto_mean = np.mean(
                    #     [healthy_prototype_H1, crack_prototype], axis=0
                    # )
                    # healthy_prototype_H1 -= proto_mean
                    # crack_prototype -= proto_mean
                    # query_embs -= proto_mean

                    # # L2-norm embeddings
                    # healthy_prototype_H1 /= np.linalg.norm(healthy_prototype_H1)
                    # crack_prototype /= np.linalg.norm(crack_prototype)
                    # query_embs /= np.linalg.norm(query_embs, axis=1, keepdims=True)
                    # XXX

                    # Calculate query distances to prototypes
                    #   H1 always used as healthy prototype
                    #   M & L crack severities used both ways
                    distances = np.stack(
                        [
                            np.linalg.norm(
                                query_embs - healthy_prototype_H1[np.newaxis, :],
                                axis=1,
                            ),
                            np.linalg.norm(
                                query_embs - crack_prototype[np.newaxis, :], axis=1
                            ),
                        ],
                        axis=1,
                    )

                    ensemble_distances.append(distances)

                ensemble_distances = np.stack(ensemble_distances, axis=-1)
                # Soft voting
                voted_distances = np.mean(ensemble_distances, axis=-1)
                acc = np.mean(np.argmin(voted_distances, axis=1) == targets)
                # Hard voting
                # voted_targets, _ = mode(np.argmin(ensemble_distances, axis=1), axis=1)
                # acc = np.mean(voted_targets == targets)

                UNSW_accuracies.append([acc])
                print(
                    f"{speed:<2} Hz, {load:<2} Nm: \033[95m{acc:>10.2%}\033[0m ({support_severity} -> {query_severity})"
                )
                print(
                    "  ",
                    ", ".join([
                        f"{np.mean(np.argmin(ensemble_distances[:, :, i], axis=1) == targets):.1%}"
                        for i in range(5)
                    ]),
                )

    print(f"UNSW Average Accuracy: \033[92m{np.mean(UNSW_accuracies):.4%}\033[0m")
    all_accuracies.append(np.mean(UNSW_accuracies))
    # break  # XXX
print(f"All accuracies: {np.mean(all_accuracies):.4%}")

# Embed MCC5

# Embed everything with ensemble

# ensemble_weight_dir_name = "SCENARIO_UNSW_UA"
# ensemble_weight_dir_name = "SCENARIO_UNSW_UNSW_ONLY"
# ensemble_weight_dir_name = "SCENARIO_UNSW_AGFD_ONLY"
ensemble_weight_dir_name = "EXPERIMENTAL_UA_PERM_SC_BLOCK"
# ensemble_weight_dir_name = "EXPERIMENTAL_SUPCON_UA"

# ensemble_repeat_num = 1
ensemble_checkpoint_num = 5

MCC5_ensemble_emb_dfs = []
for ensemble_repeat_num in range(1, 4):  # XXX
    # for ensemble_repeat_num in range(1, 11):
    ensemble_weight_dirs = sorted(
        list(os.walk(model_weight_dir / ensemble_weight_dir_name))[0][1]
    )[(ensemble_repeat_num - 1) * 5 : ensemble_repeat_num * 5]

    for e in range(5):
        # Load ensemble models
        model.backbone.load_state_dict(
            torch.load(
                model_weight_dir
                / ensemble_weight_dir_name
                / ensemble_weight_dirs[e]
                / f"{ensemble_checkpoint_num}.pth",
                weights_only=True,
            )
        )
        model.eval()

        # Embeddings for one ensemble model
        all_emb_dfs = []
        for speed in [1000, 2000, 3000]:
            for load in [10, 20]:
                # ANCHOR
                #

                if config["use_difference"]:
                    # * Use 7 first samples for anchors. Slowest speed has 14 samples, so it's 50% of that.
                    # * 7 from each to have consistent number of anchor samples.
                    anchor_samples = get_samples_MCC5(
                        MCC5_df, speed, load, "healthy", "-", circulation="torque"
                    )[:7]

                    anchor_samples = reduce(anchor_samples, "b w -> 1 w", "mean")
                    # anchor_samples = rearrange(anchor_samples, "(a b) w -> a b w", a=4, b=4)
                    # anchor_samples = reduce(anchor_samples, "a b w -> a w", a=4, b=4)

                # HEALTHY 1
                #

                # Get samples from specific conditions
                # * Drop 7 first from H1, as those are used for anchors
                healthy_samples_1 = get_samples_MCC5(
                    MCC5_df, speed, load, "healthy", "-", circulation="torque"
                )[7:]
                healthy_emb_df_1 = embed_wrapper(
                    healthy_samples_1, anchor_samples, model, config, embedding_specs
                )
                healthy_emb_df_1["class"] = "healthy"
                healthy_emb_df_1["severity"] = "H1"

                # HEALTHY 2
                #

                # Get samples from specific conditions
                healthy_samples_2 = get_samples_MCC5(
                    MCC5_df, speed, load, "healthy", "-", circulation="speed"
                )
                healthy_emb_df_2 = embed_wrapper(
                    healthy_samples_2, anchor_samples, model, config, embedding_specs
                )
                healthy_emb_df_2["class"] = "healthy"
                healthy_emb_df_2["severity"] = "H2"

                # CRACK 1
                #

                # Get samples from specific conditions
                crack_samples_1 = get_samples_MCC5(MCC5_df, speed, load, "crack", "M")
                crack_emb_df_1 = embed_wrapper(
                    crack_samples_1, anchor_samples, model, config, embedding_specs
                )
                crack_emb_df_1["class"] = "crack"
                crack_emb_df_1["severity"] = "M"

                # CRACK 2
                #

                # Get samples from specific conditions
                crack_samples_2 = get_samples_MCC5(MCC5_df, speed, load, "crack", "L")
                crack_emb_df_2 = embed_wrapper(
                    crack_samples_2, anchor_samples, model, config, embedding_specs
                )
                crack_emb_df_2["class"] = "crack"
                crack_emb_df_2["severity"] = "L"

                # WEAR 1
                #

                # Get samples from specific conditions
                wear_samples_1 = get_samples_MCC5(MCC5_df, speed, load, "wear", "M")
                wear_emb_df_1 = embed_wrapper(
                    wear_samples_1, anchor_samples, model, config, embedding_specs
                )
                wear_emb_df_1["class"] = "wear"
                wear_emb_df_1["severity"] = "M"

                # WEAR 2
                #

                # Get samples from specific conditions
                wear_samples_2 = get_samples_MCC5(MCC5_df, speed, load, "wear", "L")
                wear_emb_df_2 = embed_wrapper(
                    wear_samples_2, anchor_samples, model, config, embedding_specs
                )
                wear_emb_df_2["class"] = "wear"
                wear_emb_df_2["severity"] = "L"

                # Combine all dataframes and add back operating condition info
                emb_df = pd.concat(
                    [
                        healthy_emb_df_1,
                        healthy_emb_df_2,
                        crack_emb_df_1,
                        crack_emb_df_2,
                        wear_emb_df_1,
                        wear_emb_df_2,
                    ],
                    ignore_index=True,
                )
                emb_df["speed"] = speed
                emb_df["load"] = load

                all_emb_dfs.append(emb_df)

        all_emb_dfs = pd.concat(all_emb_dfs, ignore_index=True)
        MCC5_ensemble_emb_dfs.append(all_emb_dfs)

# MCC5 accuracy evaluation

# 1 severity for prototypes
# 1 severity for query

all_accuracies = []
# for offset in range(10):
for offset in range(3):  # XXX
    MCC5_accuracies = []
    for speed in [1000, 2000, 3000]:
        for load in [10, 20]:
            for support_severity, query_severity in [["M", "L"], ["L", "M"]]:
                ensemble_distances = []
                for e in range(5):
                    round_df = MCC5_ensemble_emb_dfs[offset * 5 + e]  # XXX
                    round_df = round_df[
                        (round_df["speed"] == speed) & (round_df["load"] == load)
                    ]

                    healthy_prototype_H1 = (
                        round_df[
                            (round_df["class"] == "healthy")
                            & (round_df["severity"] == "H1")
                        ][["x", "y"]]
                        .to_numpy()
                        .mean(axis=0)
                    )

                    # crack_prototype = (
                    #     round_df[
                    #         (round_df["class"] == "crack")
                    #         & (round_df["severity"] == support_severity)
                    #     ][["x", "y"]]
                    #     .to_numpy()
                    #     .mean(axis=0)
                    # )

                    wear_prototype = (
                        round_df[
                            (round_df["class"] == "wear")
                            & (round_df["severity"] == support_severity)
                        ][["x", "y"]]
                        .to_numpy()
                        .mean(axis=0)
                    )

                    query_df = round_df[
                        (round_df["severity"].isin([query_severity, "H2"]))
                        # & (round_df["class"].isin(["healthy", "crack"]))
                        & (round_df["class"].isin(["healthy", "wear"]))
                    ]

                    # Convert severities to target labels [0, 1, 2]
                    targets = (
                        query_df["class"]
                        .map({
                            "healthy": 0,
                            # "crack": 1,
                            # "wear": 2,
                            "wear": 1,
                        })
                        # .map({"healthy": 0, "crack": 1, "wear": 2})
                        .to_numpy()
                    )
                    query_embs = query_df[["x", "y"]].to_numpy()

                    # query_embs - healthy_prototype_H1
                    distances = np.stack(
                        [
                            np.linalg.norm(
                                query_embs - healthy_prototype_H1[np.newaxis, :],
                                axis=1,
                            ),
                            # np.linalg.norm(
                            #     query_embs - crack_prototype[np.newaxis, :], axis=1
                            # ),
                            np.linalg.norm(
                                query_embs - wear_prototype[np.newaxis, :], axis=1
                            ),
                        ],
                        axis=1,
                    )
                    ensemble_distances.append(distances)

                ensemble_distances = np.stack(ensemble_distances, axis=-1)
                # Soft voting
                voted_distances = np.mean(ensemble_distances, axis=-1)
                acc = np.mean(np.argmin(voted_distances, axis=1) == targets)
                # Hard voting
                # voted_targets, _ = mode(np.argmin(ensemble_distances, axis=1), axis=1)
                # acc = np.mean(voted_targets == targets)

                MCC5_accuracies.append(acc)
                print(
                    f"{speed:<2} Hz, {load:<2} Nm: \033[95m{acc:>10.2%}\033[0m ({support_severity}->{query_severity})"
                )
                print(
                    "  ",
                    ", ".join([
                        f"{np.mean(np.argmin(ensemble_distances[:, :, i], axis=1) == targets):.1%}"
                        for i in range(5)
                    ]),
                )

    print(f"MCC5 average accuracy: \033[92m{np.mean(MCC5_accuracies):.4%}\033[0m")
    all_accuracies.append(np.mean(MCC5_accuracies))
print(f"All accuracies: {np.mean(all_accuracies):.4%}")


# Embed AGFD

# Embed everything with ensemble

# ensemble_weight_dir_name = "SCENARIO_UNSW_UM"
# ensemble_weight_dir_name = "SCENARIO_UNSW_UNSW_ONLY"
# ensemble_weight_dir_name = "SCENARIO_UNSW_MCC5_ONLY"
# ensemble_weight_dir_name = "EXPERIMENTAL_U_PERM"
# ensemble_weight_dir_name = "EXPERIMENTAL_U_PERM_BLOCK"
# ensemble_weight_dir_name = "EXPERIMENTAL_U_PERM_SC_BLOCK"
# ensemble_weight_dir_name = "EXPERIMENTAL_U_BLOCK"
# ensemble_weight_dir_name = "EXPERIMENTAL_U_PERM_BLOCK_40"
ensemble_weight_dir_name = "EXP_UM_PERM_BLOCK_40_AMP50"
# ensemble_weight_dir_name = "EXPERIMENTAL_U_PERM_SC_BLOCK_40"

# ensemble_repeat_num = 1
ensemble_checkpoint_num = 5

AGFD_ensemble_emb_dfs = []
for ensemble_repeat_num in range(1, 11):
    # for ensemble_repeat_num in range(1, 5):  # XXX
    ensemble_weight_dirs = sorted(
        list(os.walk(model_weight_dir / ensemble_weight_dir_name))[0][1]
    )[(ensemble_repeat_num - 1) * 5 : ensemble_repeat_num * 5]

    for e in range(5):
        # Load ensemble models
        model.backbone.load_state_dict(
            torch.load(
                model_weight_dir
                / ensemble_weight_dir_name
                / ensemble_weight_dirs[e]
                / f"{ensemble_checkpoint_num}.pth",
                weights_only=True,
            )
        )
        model.eval()

        # Embeddings for one ensemble model
        all_emb_dfs = []
        for speed in [500, 750, 1000, 1250, 1500]:
            for load in [1, 6, 11]:
                # ANCHOR
                #

                if config["use_difference"]:
                    # * 16 should be enough
                    anchor_samples = get_samples_AGFD(
                        AGFD_df, speed, load, "healthy", "-", healthy_GP=4
                    )[:16]

                    anchor_samples = reduce(anchor_samples, "b w -> 1 w", "mean")
                    # anchor_samples = rearrange(anchor_samples, "(a b) w -> a b w", a=4, b=4)
                    # anchor_samples = reduce(anchor_samples, "a b w -> a w", a=4, b=4)

                # HEALTHY 1
                #

                # Get samples from specific conditions
                # * Drop 7 first from H1, as those are used for anchors
                healthy_samples_1 = get_samples_AGFD(
                    AGFD_df, speed, load, "healthy", "-", healthy_GP=5
                )[16:]

                healthy_emb_df_1 = embed_wrapper(
                    healthy_samples_1, anchor_samples, model, config, embedding_specs
                )
                healthy_emb_df_1["class"] = "healthy"
                healthy_emb_df_1["severity"] = "H1"
                healthy_emb_df_1["installation"] = 0

                # HEALTHY 2
                #

                # Get samples from specific conditions
                healthy_samples_2 = get_samples_AGFD(
                    AGFD_df, speed, load, "healthy", "-", healthy_GP=9
                )
                healthy_emb_df_2 = embed_wrapper(
                    healthy_samples_2, anchor_samples, model, config, embedding_specs
                )
                healthy_emb_df_2["class"] = "healthy"
                healthy_emb_df_2["severity"] = "H2"
                healthy_emb_df_2["installation"] = 0

                crack_emb_dfs_1 = []
                crack_emb_dfs_2 = []
                wear_emb_dfs_1 = []
                wear_emb_dfs_2 = []
                for installation in [1, 2, 3]:
                    # CRACK 1
                    #

                    # Get samples from specific conditions
                    crack_samples_1 = get_samples_AGFD(
                        AGFD_df, speed, load, "crack", "S", installation=installation
                    )
                    crack_emb_df_1 = embed_wrapper(
                        crack_samples_1, anchor_samples, model, config, embedding_specs
                    )
                    crack_emb_df_1["class"] = "crack"
                    crack_emb_df_1["severity"] = "S"
                    crack_emb_df_1["installation"] = installation
                    crack_emb_dfs_1.append(crack_emb_df_1)

                    # CRACK 2
                    #

                    # Get samples from specific conditions
                    crack_samples_2 = get_samples_AGFD(
                        AGFD_df, speed, load, "crack", "L", installation=installation
                    )
                    crack_emb_df_2 = embed_wrapper(
                        crack_samples_2, anchor_samples, model, config, embedding_specs
                    )
                    crack_emb_df_2["class"] = "crack"
                    crack_emb_df_2["severity"] = "L"
                    crack_emb_df_2["installation"] = installation
                    crack_emb_dfs_2.append(crack_emb_df_2)

                    # WEAR 1
                    #

                    # Get samples from specific conditions
                    wear_samples_1 = get_samples_AGFD(
                        AGFD_df, speed, load, "wear", "S", installation=installation
                    )
                    wear_emb_df_1 = embed_wrapper(
                        wear_samples_1, anchor_samples, model, config, embedding_specs
                    )
                    wear_emb_df_1["class"] = "wear"
                    wear_emb_df_1["severity"] = "S"
                    wear_emb_df_1["installation"] = installation
                    wear_emb_dfs_1.append(wear_emb_df_1)

                    # WEAR 2
                    #

                    # Get samples from specific conditions
                    wear_samples_2 = get_samples_AGFD(
                        AGFD_df, speed, load, "wear", "L", installation=installation
                    )
                    wear_emb_df_2 = embed_wrapper(
                        wear_samples_2, anchor_samples, model, config, embedding_specs
                    )
                    wear_emb_df_2["class"] = "wear"
                    wear_emb_df_2["severity"] = "L"
                    wear_emb_df_2["installation"] = installation
                    wear_emb_dfs_2.append(wear_emb_df_2)

                crack_emb_df_1 = pd.concat(crack_emb_dfs_1, ignore_index=True)
                crack_emb_df_2 = pd.concat(crack_emb_dfs_2, ignore_index=True)
                wear_emb_df_1 = pd.concat(wear_emb_dfs_1, ignore_index=True)
                wear_emb_df_2 = pd.concat(wear_emb_dfs_2, ignore_index=True)

                # Combine all dataframes and add back operating condition info
                emb_df = pd.concat(
                    [
                        healthy_emb_df_1,
                        healthy_emb_df_2,
                        crack_emb_df_1,
                        crack_emb_df_2,
                        wear_emb_df_1,
                        wear_emb_df_2,
                    ],
                    ignore_index=True,
                )
                emb_df["speed"] = speed
                emb_df["load"] = load

                all_emb_dfs.append(emb_df)

        all_emb_dfs = pd.concat(all_emb_dfs, ignore_index=True)
        AGFD_ensemble_emb_dfs.append(all_emb_dfs)

# AGFD accuracy evaluation

# 1 installation for prototypes
# 1 installation for queries

all_accuracies = []
# for offset in range(10):
# for offset in range(2): # XXX
for offset in range(len(AGFD_ensemble_emb_dfs) // 5):  # XXX
    AGFD_accuracies = []
    for support_installation in [1, 2, 3]:
        for query_installation in [1, 2, 3]:
            if support_installation == query_installation:
                continue

            for speed in [500, 750, 1000, 1250, 1500]:
                for load in [1, 6, 11]:
                    ensemble_distances = []
                    for e in range(5):
                        round_df = AGFD_ensemble_emb_dfs[offset * 5 + e]  # XXX
                        round_df = round_df[
                            (round_df["speed"] == speed) & (round_df["load"] == load)
                        ]

                        healthy_prototype_H1 = (
                            round_df[
                                (round_df["class"] == "healthy")
                                & (round_df["severity"] == "H1")
                            ][["x", "y"]]
                            .to_numpy()
                            .mean(axis=0)
                        )

                        crack_prototype = (
                            round_df[
                                (round_df["class"] == "crack")
                                & (round_df["severity"] == "S")
                                & (round_df["installation"] == support_installation)
                            ][["x", "y"]]
                            .to_numpy()
                            .mean(axis=0)
                        )

                        wear_prototype = (
                            round_df[
                                (round_df["class"] == "wear")
                                & (round_df["severity"] == "L")
                            ][["x", "y"]]
                            .to_numpy()
                            .mean(axis=0)
                        )

                        query_df = round_df[
                            (round_df["installation"].isin([0, query_installation]))
                            & (round_df["severity"].isin(["H1", "M", "L"]))
                        ]

                        # Convert severities to target labels [0, 1, 2]
                        targets = (
                            query_df["class"]
                            .map({"healthy": 0, "crack": 1, "wear": 2})
                            .to_numpy()
                        )

                        query_embs = query_df[["x", "y"]].to_numpy()

                        distances = np.stack(
                            [
                                np.linalg.norm(
                                    query_embs - healthy_prototype_H1[np.newaxis, :],
                                    # query_embs - healthy_prototype_H2[np.newaxis, :],
                                    axis=1,
                                ),
                                np.linalg.norm(
                                    query_embs - crack_prototype[np.newaxis, :],
                                    axis=1,
                                ),
                                np.linalg.norm(
                                    query_embs - wear_prototype[np.newaxis, :],
                                    axis=1,
                                ),
                            ],
                            axis=1,
                        )

                        ensemble_distances.append(distances)

                    ensemble_distances = np.stack(ensemble_distances, axis=-1)
                    # Soft voting
                    voted_distances = np.mean(ensemble_distances, axis=-1)
                    acc = np.mean(np.argmin(voted_distances, axis=1) == targets)
                    # Hard voting
                    # voted_targets, _ = mode(np.argmin(ensemble_distances, axis=1), axis=1)
                    # acc = np.mean(voted_targets == targets)

                    AGFD_accuracies.append((acc))
                    # print(
                    #     f"{speed:<4} RPM, {load/100:3.0%}, s_inst. {support_installation}, q_inst. {query_installation}: \033[95m{acc:>10.2%}\033[0m ({severity})"
                    # )
                    # print(
                    #     "  ",
                    #     ", ".join(
                    #         [
                    #             f"{np.mean(np.argmin(ensemble_distances[:, :, i], axis=1) == targets):.1%}"
                    #             for i in range(5)
                    #         ]
                    #     ),
                    # )

    print(f"AGFD average accuracy: \033[92m{np.mean(AGFD_accuracies):.4%}\033[0m")
    all_accuracies.append(np.mean(AGFD_accuracies))
print(f"All accuracies: {np.mean(all_accuracies):.4%}")
