import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from einops import reduce
from ruamel.yaml import YAML

from src.data.preprocessing_individual import get_important_frequencies
from src.models.models import setup_model
from src.training.training import CHECKPOINT_INTERVAL
from src.utils.init import setup_device

# CONFIG
###

# SETUP
ENSEMBLE_SIZE = 3
# How ensemble members are combined: "soft" averages the members' distances to each
# prototype, "hard" takes a majority vote of the members' predictions
VOTING_TYPE = "soft"
ENSEMBLE_CHECKPOINT_NUM = 5  # Batch 500, the last checkpoint of max_batches: 501

device = setup_device(override="cpu")
MODEL_WEIGHT_DIR = Path.cwd().resolve() / "model_weights"
# Keeps log10 finite for zero-valued spectrum bins
LOG_EPS = 1e-12

embedding_specs = {
    # L2-normalize every window embedding before averaging (and the average after it).
    # The triplet loss is trained on L2-normalized embeddings, so only their direction
    # is meaningful
    "normalize_embeddings": True,
    "average_anchors": True,
    "average_batch": True,
    "drop_anchor_outliers": 0,
    "drop_batch_outliers": 0,
}

config = {
    # Defaults for testing; the TRAINING_CONFIG_KEYS entries are replaced with the
    # values the models were trained with (run_info.yaml)
    "dropout": 0.0,
    "loss": "triplet",
    "similarity": "euclidean",
    "loss_args": {
        "margin": 1.4,  # Doesn't matter during inference
        "miner": False,  # Doesn't matter during inference
    },
    "lpnorm_embeddings_training": False,
    "center_embeddings_inference": False,
    "lpnorm_embeddings_inference": False,
}

UNSW_teeth = 27
AGFD_teeth = 15
MCC5_teeth = 36

# Fraction of the H1 windows of each condition used for the anchor, the rest form the
# healthy prototype (UNSW_V6 has 62-92 H1 windows per condition)
UNSW_ANCHOR_FRACTION = 0.5
MCC5_ANCHOR_NUM = 7
# The only severity of missing tooth in MCC5-THU
MCC5_MISSING_TOOTH_SEVERITY = "S"

# Sensors used when testing on each dataset: the training sensor by default. With
# use_all_sensors() (--all-sensors), every (model, sensor) pair votes, i.e. the
# prototype distances are averaged over ensemble members and all sensors (AGFD acc2 is
# broken for installation 3, so it isn't used)
AGFD_SENSORS = ["acc4"]
MCC5_SENSORS = ["gearbox_vibration_x"]
ALL_AGFD_SENSORS = ["acc3", "acc4"]
ALL_MCC5_SENSORS = ["gearbox_vibration_x", "gearbox_vibration_y", "gearbox_vibration_z"]


def use_all_sensors():
    """Test with all sensors voting instead of the training sensor only."""
    global AGFD_SENSORS, MCC5_SENSORS
    AGFD_SENSORS = ALL_AGFD_SENSORS
    MCC5_SENSORS = ALL_MCC5_SENSORS

AGFD_SPEEDS = [500, 750, 1000, 1250, 1500]
AGFD_LOADS = [1, 6, 11]
AGFD_INSTALLATIONS = [1, 2, 3]
# Healthy gears that never had faults seeded in them for a class in the test
# (missing tooth is GP5, mild crack GP9, and GP8 doesn't exist)
AGFD_CLEAN_HEALTHY_GPS = [1, 2, 3, 4, 6, 7]

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
        # topk orders by distance, keep the original order instead
        filtered_embeddings.append(embeddings[i][closest_indices.sort().values, :])

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
    # topk orders by distance, keep the original order so that the same row is the
    # same sample for every ensemble member (needed for voting without batch averaging)
    return embeddings[closest_indices.sort().values, :]


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

    if normalize_embeddings:
        embeddings = F.normalize(embeddings, p=2, dim=-1)

    if drop_batch_outliers > 0:
        embeddings = remove_batch_outliers(embeddings, num_outliers=drop_batch_outliers)

    if average_batch:
        embeddings = reduce(embeddings, "b f -> 1 f", reduction="mean")

    if normalize_embeddings:
        embeddings = F.normalize(embeddings, p=2, dim=-1)

    embeddings = embeddings.view(-1, embeddings.shape[-1])
    return embeddings.detach().numpy()


def embedding_columns():
    """DataFrame columns holding the embedding, one per embedding dimension."""
    return [f"emb_{i}" for i in range(config["embedding_len"])]


def embed_wrapper(samples, anchors, embedding_model, config, embedding_specs):
    if config["use_difference"]:
        samples = samples[:, np.newaxis, :] - anchors[np.newaxis, :, :]
    else:
        samples = samples[:, np.newaxis, :]

    # Batch outlier removal keeps n - drop_batch_outliers samples, so it needs at least one left
    min_samples = embedding_specs["drop_batch_outliers"] + 1
    if len(samples) < min_samples:
        raise ValueError(
            f"Need at least {min_samples} samples to embed, got {len(samples)}"
        )

    embeddings = embed(samples, embedding_model, **embedding_specs)
    return pd.DataFrame(embeddings, columns=embedding_columns())


def prepare_samples(samples, teeth):
    if len(samples) == 0:
        raise ValueError("No samples match the requested operating condition")

    samples = np.array(list(samples["signal"]))
    samples = samples[
        :,
        get_important_frequencies(
            teeth,
            config["gear_harmonics_to_include"],
            config["gear_sideband_block_size"],
        ),
    ]
    return np.log10(samples + LOG_EPS)


def get_samples_UNSW(df, speed, load, severity):
    samples = df[
        (df["speed"] == speed) & (df["load"] == load) & (df["severity"] == severity)
    ]
    return prepare_samples(samples, UNSW_teeth)


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

    return prepare_samples(samples, MCC5_teeth)


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

    return prepare_samples(samples, AGFD_teeth)


def check_checkpoint_available(weight_dir):
    """Fail clearly if no ensemble member has the checkpoint ENSEMBLE_CHECKPOINT_NUM."""
    member_dirs = [d for d in weight_dir.iterdir() if d.is_dir()]
    if any((d / f"{ENSEMBLE_CHECKPOINT_NUM}.pth").is_file() for d in member_dirs):
        return

    available = sorted({
        int(p.stem) for d in member_dirs for p in d.glob("*.pth") if p.stem.isdigit()
    })
    if not available:
        raise SystemExit(f"No ensemble members with checkpoints found in {weight_dir}")
    raise SystemExit(
        f"No ensemble member in {weight_dir} has {ENSEMBLE_CHECKPOINT_NUM}.pth, the "
        f"members have checkpoints {available[0]}-{available[-1]}. Set "
        f"ENSEMBLE_CHECKPOINT_NUM in test.py to at most {available[-1]}, or train with "
        f"max_batches > {ENSEMBLE_CHECKPOINT_NUM * CHECKPOINT_INTERVAL} (a checkpoint "
        f"is saved every {CHECKPOINT_INTERVAL} batches)."
    )


def find_ensemble_members(weight_dir):
    # Only subdirectories holding the selected checkpoint count as ensemble members
    member_dirs = sorted(
        d.name
        for d in weight_dir.iterdir()
        if (d / f"{ENSEMBLE_CHECKPOINT_NUM}.pth").is_file()
    )
    ensemble_repeats = len(member_dirs) // ENSEMBLE_SIZE
    print(
        f"Found {len(member_dirs)} ensemble members ({ensemble_repeats} complete ensembles)."
    )
    if ensemble_repeats == 0:
        raise SystemExit(
            f"Not enough members with {ENSEMBLE_CHECKPOINT_NUM}.pth for one ensemble in {weight_dir}"
        )

    return member_dirs, ensemble_repeats


# Config entries that define the input features, model architecture and how
# embeddings are compared. They are read from the trained models' run_info.yaml, the
# values in `config` above are only used for older weights without one.
TRAINING_CONFIG_KEYS = [
    "use_difference",
    "gear_harmonics_to_include",
    "gear_sideband_block_size",
    "embedding_len",
    "model",
    "backbone",
    "model_args",
    "similarity",
    "center_embeddings_inference",
    "lpnorm_embeddings_inference",
]


def apply_training_config(weight_dir):
    """Update `config` with the settings the ensemble members were trained with."""
    member_dirs = [
        d
        for d in sorted(weight_dir.iterdir())
        if (d / f"{ENSEMBLE_CHECKPOINT_NUM}.pth").is_file()
    ]
    run_info_paths = [
        d / "run_info.yaml" for d in member_dirs if (d / "run_info.yaml").is_file()
    ]

    if not run_info_paths:
        print(
            "No run_info.yaml found, using the input and model settings hard-coded in test.py."
        )
        return
    if len(run_info_paths) < len(member_dirs):
        raise SystemExit(
            f"Only {len(run_info_paths)} of {len(member_dirs)} members in {weight_dir} have a run_info.yaml"
        )

    yaml = YAML(typ="safe")
    training_settings = None
    for path in run_info_paths:
        with open(path) as stream:
            training_config = yaml.load(stream)["config"]
        settings = {key: training_config[key] for key in TRAINING_CONFIG_KEYS}

        # All members of the ensembles must share the same inputs and architecture
        if training_settings is None:
            training_settings = settings
        elif settings != training_settings:
            raise SystemExit(
                f"{path.parent.name} was trained with different settings than "
                f"{run_info_paths[0].parent.name}:\n{settings}\nvs\n{training_settings}"
            )

    config.update(training_settings)
    print(
        f"Using settings from run_info.yaml: {training_settings['gear_harmonics_to_include']} "
        f"harmonics, sideband block size {training_settings['gear_sideband_block_size']}, "
        f"{training_settings['embedding_len']}-D embeddings, "
        f"{training_settings['similarity']} similarity"
        + (", centered" if training_settings["center_embeddings_inference"] else "")
        + (
            ", L2-normalized"
            if training_settings["similarity"] == "euclidean"
            and training_settings["lpnorm_embeddings_inference"]
            else ""
        )
        + "."
    )


def vote(ensemble_distances):
    """
    Combine the ensemble members' distances, (n_queries, n_prototypes, n_members),
    into one predicted prototype index per query.

    Soft: prototype with the lowest distance averaged over members.
    Hard: prototype predicted by the most members. Ties (e.g. 1-1-1 with 3 members
        and 3 classes) go to the tied prototype with the lowest average distance,
        instead of always the first prototype (healthy).
    """
    mean_distances = np.mean(ensemble_distances, axis=-1)

    if VOTING_TYPE == "soft":
        return np.argmin(mean_distances, axis=1)
    elif VOTING_TYPE == "hard":
        n_queries, n_prototypes, _ = ensemble_distances.shape
        # (n_queries, n_members), each member's prediction
        member_predictions = np.argmin(ensemble_distances, axis=1)
        # (n_queries, n_prototypes), number of members voting for each prototype
        vote_counts = np.stack([
            np.bincount(p, minlength=n_prototypes) for p in member_predictions
        ])
        # Only prototypes with the most votes can win, ties broken by distance
        is_winner = vote_counts == vote_counts.max(axis=1, keepdims=True)
        return np.argmin(np.where(is_winner, mean_distances, np.inf), axis=1)
    else:
        raise ValueError(f"Unknown VOTING_TYPE '{VOTING_TYPE}', use 'soft' or 'hard'")


def prototype_distances(query_embs, prototypes):
    """
    Distance of every query to every prototype, (n_queries, n_prototypes), lower is
    closer. Prototypes are means of raw embeddings. Follows Embedding.predict in
    training:
        1. If center_embeddings_inference, center both on the mean of the prototypes
        2. If cosine similarity or lpnorm_embeddings_inference, L2-normalize both
        3. Euclidean distance, or cosine distance (1 - cosine similarity) so that soft
           voting and argmin work the same way for both
    """
    if config["similarity"] not in ("euclidean", "cosine"):
        raise ValueError(f"Unknown similarity type '{config['similarity']}'!")

    prototypes = np.stack(prototypes)

    if config["center_embeddings_inference"]:
        mean = np.mean(prototypes, axis=0)
        prototypes = prototypes - mean
        query_embs = query_embs - mean

    if config["similarity"] == "cosine" or config["lpnorm_embeddings_inference"]:
        # Same epsilon as F.normalize
        prototypes = prototypes / np.maximum(
            np.linalg.norm(prototypes, axis=-1, keepdims=True), 1e-12
        )
        query_embs = query_embs / np.maximum(
            np.linalg.norm(query_embs, axis=-1, keepdims=True), 1e-12
        )

    if config["similarity"] == "euclidean":
        return np.stack(
            [
                np.linalg.norm(query_embs - prototype[np.newaxis, :], axis=1)
                for prototype in prototypes
            ],
            axis=1,
        )
    return 1 - query_embs @ prototypes.T


# UNSW
###


def load_UNSW():
    UNSW_crack_df = pd.read_feather(
        Path.cwd() / "data" / "UNSW_V6.feather"
    )

    # Change S crack to second healthy
    UNSW_crack_severity_fix_map = {"H": "H1", "S": "H2"}
    UNSW_crack_df["severity"] = UNSW_crack_df["severity"].transform(
        lambda x: UNSW_crack_severity_fix_map.get(x, x)
    )

    # Need only one OT method
    UNSW_crack_df = UNSW_crack_df[UNSW_crack_df["OT_method"] == "signal"]

    # Drop unnecessary columns
    UNSW_crack_df.drop(
        columns=["OT_method", "TSA_size"],
        inplace=True,
    )

    return UNSW_crack_df


def run_UNSW(model, weight_dir):
    UNSW_crack_df = load_UNSW()

    all_weight_dirs, ensemble_repeats = find_ensemble_members(weight_dir)

    UNSW_ensemble_emb_dfs = []
    for ensemble_repeat_num in range(1, ensemble_repeats + 1):
        ensemble_weight_dirs = all_weight_dirs[
            (ensemble_repeat_num - 1) * ENSEMBLE_SIZE : ensemble_repeat_num
            * ENSEMBLE_SIZE
        ]

        for e in range(ENSEMBLE_SIZE):
            # Load ensemble models
            model.backbone.load_state_dict(
                torch.load(
                    weight_dir
                    / ensemble_weight_dirs[e]
                    / f"{ENSEMBLE_CHECKPOINT_NUM}.pth",
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

                    H1_samples = get_samples_UNSW(UNSW_crack_df, speed, load, "H1")
                    num_anchors = int(len(H1_samples) * UNSW_ANCHOR_FRACTION)

                    if config["use_difference"]:
                        # First part of H1 for the anchor
                        anchor_samples = H1_samples[:num_anchors]

                        anchor_samples = reduce(anchor_samples, "b w -> 1 w", "mean")

                    # HEALTHY 1
                    #

                    # Rest of H1, the anchor windows are left out
                    healthy_samples_1 = H1_samples[num_anchors:]
                    healthy_emb_df_1 = embed_wrapper(
                        healthy_samples_1,
                        anchor_samples,
                        model,
                        config,
                        embedding_specs,
                    )
                    healthy_emb_df_1["class"] = "healthy"
                    healthy_emb_df_1["severity"] = "H1"

                    # HEALTHY 2
                    #

                    healthy_samples_2 = get_samples_UNSW(
                        UNSW_crack_df, speed, load, "H2"
                    )
                    healthy_emb_df_2 = embed_wrapper(
                        healthy_samples_2,
                        anchor_samples,
                        model,
                        config,
                        embedding_specs,
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
    for offset in range(ensemble_repeats):
        UNSW_accuracies = []
        # For every operating conditions (consisting of speed and load)
        for speed in [5, 10, 15, 20]:
            for load in [5, 10, 15]:
                for support_severity, query_severity in [("M", "L"), ("L", "M")]:
                    # Repeat with every ensemble model
                    ensemble_distances = []
                    for e in range(ENSEMBLE_SIZE):
                        round_df = UNSW_ensemble_emb_dfs[offset * ENSEMBLE_SIZE + e]
                        round_df = round_df[
                            (round_df["speed"] == speed) & (round_df["load"] == load)
                        ]

                        # Get possible prototypes

                        healthy_prototype_H1 = (
                            round_df[round_df["severity"] == "H1"][embedding_columns()]
                            .to_numpy()
                            .mean(axis=0)
                        )

                        crack_prototype = (
                            round_df[round_df["severity"] == support_severity][
                                embedding_columns()
                            ]
                            .to_numpy()
                            .mean(axis=0)
                        )

                        # Queries: H2 (healthy) and the crack severity not used for the prototype
                        query_df = round_df[
                            round_df["severity"].isin([query_severity, "H2"])
                        ]

                        # Convert severities to target labels [0, 1]
                        targets = (
                            query_df["severity"]
                            .map({"H2": 0, "M": 1, "L": 1})
                            .to_numpy()
                        )

                        query_embs = query_df[embedding_columns()].to_numpy()

                        # Calculate query distances to prototypes
                        #   H1 always used as healthy prototype
                        #   M & L crack severities used both ways
                        distances = prototype_distances(
                            query_embs, [healthy_prototype_H1, crack_prototype]
                        )

                        ensemble_distances.append(distances)

                    ensemble_distances = np.stack(ensemble_distances, axis=-1)
                    # Combine ensemble members (soft or hard voting)
                    acc = np.mean(vote(ensemble_distances) == targets)

                    UNSW_accuracies.append([acc])
                    print(
                        f"{speed:<2} Hz, {load:<2} Nm: \033[95m{acc:>10.2%}\033[0m ({support_severity} -> {query_severity})"
                    )
                    print(
                        "  ",
                        ", ".join([
                            f"{np.mean(np.argmin(ensemble_distances[:, :, i], axis=1) == targets):.1%}"
                            for i in range(ENSEMBLE_SIZE)
                        ]),
                    )

        print(f"UNSW Average Accuracy: \033[92m{np.mean(UNSW_accuracies):.4%}\033[0m")
        all_accuracies.append(np.mean(UNSW_accuracies))
    print(f"All accuracies: {np.mean(all_accuracies):.4%}")

    # Average accuracy of each ensemble
    return all_accuracies


# MCC5
###


def load_MCC5(sensor="gearbox_vibration_x"):
    MCC5_df = pd.read_feather(Path.cwd() / "data" / "MCC5_V6.feather")

    # Need only one OT method
    MCC5_df = MCC5_df[MCC5_df["OT_method"] == "signal"]
    # Should only be 30 in this file, but just in case
    MCC5_df = MCC5_df[MCC5_df["TSA_size"] == 30]

    # Drop unnecessary columns
    MCC5_df.drop(
        columns=["OT_method", "TSA_size"],
        inplace=True,
    )
    # Only use one sensor
    MCC5_df = MCC5_df[["speed", "load", "severity", "class", "circulation", sensor]]
    MCC5_df = MCC5_df.rename(columns={sensor: "signal"})

    # Drop in-between speeds and loads
    MCC5_df = MCC5_df[MCC5_df["speed"].isin([1000, 2000, 3000])]
    MCC5_df = MCC5_df[MCC5_df["load"].isin([10, 20])]

    # Keep only crack, missing tooth, and healthy
    MCC5_df = MCC5_df[MCC5_df["class"].isin(["crack", "missing_tooth", "healthy"])]

    return MCC5_df


def run_MCC5(model, weight_dir):
    MCC5_dfs = [load_MCC5(sensor) for sensor in MCC5_SENSORS]

    all_weight_dirs, ensemble_repeats = find_ensemble_members(weight_dir)

    MCC5_ensemble_emb_dfs = []
    for ensemble_repeat_num in range(1, ensemble_repeats + 1):
        ensemble_weight_dirs = all_weight_dirs[
            (ensemble_repeat_num - 1) * ENSEMBLE_SIZE : ensemble_repeat_num
            * ENSEMBLE_SIZE
        ]

        for e in range(ENSEMBLE_SIZE):
            # Load ensemble models
            model.backbone.load_state_dict(
                torch.load(
                    weight_dir
                    / ensemble_weight_dirs[e]
                    / f"{ENSEMBLE_CHECKPOINT_NUM}.pth",
                    weights_only=True,
                )
            )
            model.eval()

            # One set of embeddings per sensor
            sensor_emb_dfs = []
            for MCC5_df in MCC5_dfs:
                # Embeddings for one ensemble model
                all_emb_dfs = []
                for speed in [1000, 2000, 3000]:
                    for load in [10, 20]:
                        # ANCHOR
                        #

                        if config["use_difference"]:
                            # First 7 torque-run windows for the anchor (the same number for
                            # every condition; V6 test units have 18-28 windows)
                            anchor_samples = get_samples_MCC5(
                                MCC5_df, speed, load, "healthy", "-", circulation="torque"
                            )[:MCC5_ANCHOR_NUM]

                            anchor_samples = reduce(anchor_samples, "b w -> 1 w", "mean")

                        # HEALTHY 1
                        #

                        # Get samples from specific conditions
                        # Rest of the torque run, the anchor windows are left out
                        healthy_samples_1 = get_samples_MCC5(
                            MCC5_df, speed, load, "healthy", "-", circulation="torque"
                        )[MCC5_ANCHOR_NUM:]
                        healthy_emb_df_1 = embed_wrapper(
                            healthy_samples_1,
                            anchor_samples,
                            model,
                            config,
                            embedding_specs,
                        )
                        healthy_emb_df_1["class"] = "healthy"
                        healthy_emb_df_1["severity"] = "H1"
                        healthy_emb_df_1["circulation"] = "torque"

                        # HEALTHY 2
                        #

                        # Get samples from specific conditions
                        healthy_samples_2 = get_samples_MCC5(
                            MCC5_df, speed, load, "healthy", "-", circulation="speed"
                        )
                        healthy_emb_df_2 = embed_wrapper(
                            healthy_samples_2,
                            anchor_samples,
                            model,
                            config,
                            embedding_specs,
                        )
                        healthy_emb_df_2["class"] = "healthy"
                        healthy_emb_df_2["severity"] = "H2"
                        healthy_emb_df_2["circulation"] = "speed"

                        # FAULTS
                        #

                        # Crack severities and missing tooth from both runs. Prototypes
                        # come from the torque run and queries from the speed run, like
                        # healthy (H1 torque, H2 speed).
                        fault_emb_dfs = []
                        for fault_class, severities in [
                            ("crack", ["M", "L"]),
                            ("missing_tooth", [MCC5_MISSING_TOOTH_SEVERITY]),
                        ]:
                            for severity in severities:
                                for circulation in ["torque", "speed"]:
                                    fault_samples = get_samples_MCC5(
                                        MCC5_df,
                                        speed,
                                        load,
                                        fault_class,
                                        severity,
                                        circulation=circulation,
                                    )
                                    fault_emb_df = embed_wrapper(
                                        fault_samples,
                                        anchor_samples,
                                        model,
                                        config,
                                        embedding_specs,
                                    )
                                    fault_emb_df["class"] = fault_class
                                    fault_emb_df["severity"] = severity
                                    fault_emb_df["circulation"] = circulation
                                    fault_emb_dfs.append(fault_emb_df)

                        # Combine all dataframes and add back operating condition info
                        emb_df = pd.concat(
                            [healthy_emb_df_1, healthy_emb_df_2, *fault_emb_dfs],
                            ignore_index=True,
                        )
                        emb_df["speed"] = speed
                        emb_df["load"] = load

                        all_emb_dfs.append(emb_df)

                all_emb_dfs = pd.concat(all_emb_dfs, ignore_index=True)
                sensor_emb_dfs.append(all_emb_dfs)
            MCC5_ensemble_emb_dfs.append(sensor_emb_dfs)

    # MCC5 accuracy evaluation

    # Healthy: H1 (torque run) for the prototype, H2 (speed run) for queries
    # Crack: one severity from the torque run for the prototype, the other severity
    #   from the speed run for queries
    # Missing tooth: torque run for the prototype, speed run for queries

    all_accuracies = []
    for offset in range(ensemble_repeats):
        MCC5_accuracies = []
        for speed in [1000, 2000, 3000]:
            for load in [10, 20]:
                for support_severity, query_severity in [["M", "L"], ["L", "M"]]:
                    ensemble_distances = []
                    for e in range(ENSEMBLE_SIZE):
                        for round_df in MCC5_ensemble_emb_dfs[offset * ENSEMBLE_SIZE + e]:
                            round_df = round_df[
                                (round_df["speed"] == speed) & (round_df["load"] == load)
                            ]

                            def mean_embedding(fault_class, severity, circulation):
                                return (
                                    round_df[
                                        (round_df["class"] == fault_class)
                                        & (round_df["severity"] == severity)
                                        & (round_df["circulation"] == circulation)
                                    ][embedding_columns()]
                                    .to_numpy()
                                    .mean(axis=0)
                                )

                            prototypes = [
                                mean_embedding("healthy", "H1", "torque"),
                                mean_embedding("crack", support_severity, "torque"),
                                mean_embedding(
                                    "missing_tooth", MCC5_MISSING_TOOTH_SEVERITY, "torque"
                                ),
                            ]
                            query_embs = np.stack([
                                mean_embedding("healthy", "H2", "speed"),
                                mean_embedding("crack", query_severity, "speed"),
                                mean_embedding(
                                    "missing_tooth", MCC5_MISSING_TOOTH_SEVERITY, "speed"
                                ),
                            ])
                            # healthy: 0, crack: 1, missing tooth: 2
                            targets = np.array([0, 1, 2])

                            distances = prototype_distances(query_embs, prototypes)
                            ensemble_distances.append(distances)

                    ensemble_distances = np.stack(ensemble_distances, axis=-1)
                    # Combine ensemble members (soft or hard voting)
                    acc = np.mean(vote(ensemble_distances) == targets)

                    MCC5_accuracies.append(acc)
                    print(
                        f"{speed:<4} RPM, {load:<2} Nm: \033[95m{acc:>10.2%}\033[0m ({support_severity}->{query_severity})"
                    )
                    print(
                        "  ",
                        ", ".join([
                            f"{np.mean(np.argmin(ensemble_distances[:, :, i], axis=1) == targets):.1%}"
                            for i in range(ENSEMBLE_SIZE)
                        ]),
                    )

        print(f"MCC5 average accuracy: \033[92m{np.mean(MCC5_accuracies):.4%}\033[0m")
        all_accuracies.append(np.mean(MCC5_accuracies))
    print(f"All accuracies: {np.mean(all_accuracies):.4%}")

    # Average accuracy of each ensemble
    return all_accuracies


# AGFD
###


def load_AGFD(sensor="acc4"):
    AGFD_df = pd.read_feather(Path.cwd() / "data" / "AGFD_V6.feather")

    # Need only one OT method
    AGFD_df = AGFD_df[AGFD_df["OT_method"] == "signal"]
    # Should only be 40 in this file, but just in case
    AGFD_df = AGFD_df[AGFD_df["TSA_size"] == 40]

    # Use only one sensor
    AGFD_df = AGFD_df.drop(
        columns=["OT_method", "TSA_size"]
        + [other for other in ["acc2", "acc3", "acc4"] if other != sensor]
    )
    AGFD_df = AGFD_df.rename(columns={sensor: "signal"})

    # Drop 250 RPM data
    AGFD_df = AGFD_df[AGFD_df["speed"] != 250]

    # Severe "crack" (Severe_TFF_GP5) is an entirely missing tooth
    missing_tooth = (AGFD_df["class"] == "crack") & (AGFD_df["severity"] == "L")
    AGFD_df.loc[missing_tooth, "class"] = "missing_tooth"

    return AGFD_df


def embed_AGFD(AGFD_df, model, anchor_GPs, healthy_GPs):
    """
    Embeddings of one model for every operating condition, with the anchor averaged
    over `anchor_GPs` (equal weight per gear). Returns
    {(speed, load, unit, GP or installation): embedding}, one batch-averaged
    embedding per entry.
    """
    embeddings = {}
    for speed in AGFD_SPEEDS:
        for load in AGFD_LOADS:
            anchor_samples = None
            if config["use_difference"]:
                anchor_samples = np.mean(
                    [
                        get_samples_AGFD(
                            AGFD_df, speed, load, "healthy", "-", healthy_GP=GP
                        ).mean(axis=0)
                        for GP in anchor_GPs
                    ],
                    axis=0,
                )[np.newaxis, :]

            units = [
                (("healthy", GP), dict(fault_class="healthy", severity="-", healthy_GP=GP))
                for GP in healthy_GPs
            ]
            for installation in AGFD_INSTALLATIONS:
                units.append((
                    ("missing_tooth", installation),
                    dict(fault_class="missing_tooth", severity="L", installation=installation),
                ))
                units.append((
                    ("crack_S", installation),
                    dict(fault_class="crack", severity="S", installation=installation),
                ))

            for (unit, key), selection in units:
                samples = get_samples_AGFD(AGFD_df, speed, load, **selection)
                embeddings[(speed, load, unit, key)] = embed_wrapper(
                    samples, anchor_samples, model, config, embedding_specs
                ).to_numpy()[0]

    return embeddings


def run_AGFD(model, weight_dir):
    """
    Healthy gears: support and query come from two different gears in
    AGFD_CLEAN_HEALTHY_GPS, and the anchor is averaged over the remaining ones. All
    ordered (support, query) pairs are used. These gears never had faults seeded in
    them for a class in the test (missing tooth is GP5, mild crack GP9).
    Faults: prototype from one installation, queries from another.

    Main task: healthy vs missing tooth. Mild crack is below the detection limit and
    is reported separately (healthy vs mild crack).
    """
    assert embedding_specs["average_batch"], (
        "AGFD evaluation expects one batch-averaged embedding per unit"
    )
    AGFD_dfs = [load_AGFD(sensor) for sensor in AGFD_SENSORS]

    all_weight_dirs, ensemble_repeats = find_ensemble_members(weight_dir)

    role_pairs = list(itertools.permutations(AGFD_CLEAN_HEALTHY_GPS, 2))

    # Embeddings for every ensemble member and anchor (one anchor per unordered pair)
    # member_embeddings[member][anchor_GPs] = [embed_AGFD(...) for every sensor]
    member_embeddings = []
    for member_dir in all_weight_dirs[: ensemble_repeats * ENSEMBLE_SIZE]:
        model.backbone.load_state_dict(
            torch.load(
                weight_dir / member_dir / f"{ENSEMBLE_CHECKPOINT_NUM}.pth",
                weights_only=True,
            )
        )
        model.eval()

        embeddings = {}
        for support_GP, query_GP in role_pairs:
            anchor_GPs = tuple(
                GP for GP in AGFD_CLEAN_HEALTHY_GPS if GP not in (support_GP, query_GP)
            )
            if anchor_GPs not in embeddings:
                embeddings[anchor_GPs] = [
                    embed_AGFD(AGFD_df, model, anchor_GPs, [support_GP, query_GP])
                    for AGFD_df in AGFD_dfs
                ]
        member_embeddings.append(embeddings)

    def evaluate(fault_unit):
        """Accuracies of healthy vs `fault_unit` for each ensemble, plus recalls."""
        ensemble_accuracies = []
        healthy_hits, fault_hits = [], []
        healthy_hits_per_GP = {GP: [] for GP in AGFD_CLEAN_HEALTHY_GPS}
        for offset in range(ensemble_repeats):
            accuracies = []
            for support_GP, query_GP in role_pairs:
                anchor_GPs = tuple(
                    GP
                    for GP in AGFD_CLEAN_HEALTHY_GPS
                    if GP not in (support_GP, query_GP)
                )
                for support_installation, query_installation in itertools.permutations(
                    AGFD_INSTALLATIONS, 2
                ):
                    for speed in AGFD_SPEEDS:
                        for load in AGFD_LOADS:
                            ensemble_distances = []
                            for e in range(ENSEMBLE_SIZE):
                                for E in member_embeddings[offset * ENSEMBLE_SIZE + e][anchor_GPs]:
                                    prototypes = [
                                        E[(speed, load, "healthy", support_GP)],
                                        E[(speed, load, fault_unit, support_installation)],
                                    ]
                                    query_embs = np.stack([
                                        E[(speed, load, "healthy", query_GP)],
                                        E[(speed, load, fault_unit, query_installation)],
                                    ])
                                    ensemble_distances.append(
                                        prototype_distances(query_embs, prototypes)
                                    )

                            ensemble_distances = np.stack(ensemble_distances, axis=-1)
                            hits = vote(ensemble_distances) == np.array([0, 1])

                            accuracies.append(np.mean(hits))
                            healthy_hits.append(hits[0])
                            fault_hits.append(hits[1])
                            healthy_hits_per_GP[query_GP].append(hits[0])
            ensemble_accuracies.append(np.mean(accuracies))

        return ensemble_accuracies, healthy_hits, fault_hits, healthy_hits_per_GP

    print(
        f"Healthy gears {AGFD_CLEAN_HEALTHY_GPS} rotated as support/query, anchor "
        f"averaged over the other {len(AGFD_CLEAN_HEALTHY_GPS) - 2}"
    )
    for fault_unit, name in [
        ("crack_S", "Detection limit, healthy vs mild crack"),
        ("missing_tooth", "Healthy vs missing tooth"),
    ]:
        accuracies, healthy_hits, fault_hits, healthy_hits_per_GP = evaluate(fault_unit)
        print("")
        print(name)
        print(
            "  Ensembles: "
            + ", ".join(f"{acc:.1%}" for acc in accuracies)
            + f" | healthy recall {np.mean(healthy_hits):.1%}, "
            f"fault recall {np.mean(fault_hits):.1%}"
        )
        print(
            "  Healthy recall per query gear: "
            + ", ".join(
                f"GP{GP} {np.mean(hits):.0%}" for GP, hits in healthy_hits_per_GP.items()
            )
        )
        print(f"  Average accuracy: \033[92m{np.mean(accuracies):.4%}\033[0m")

    # Average accuracy of each ensemble (main task)
    return accuracies


# MAIN
###

RUNNERS = {"UNSW": run_UNSW, "AGFD": run_AGFD, "MCC5": run_MCC5}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained embedding-model ensemble on a gear fault dataset."
    )
    parser.add_argument(
        "weight_dir",
        help=(
            "Directory containing the ensemble weights. Relative paths are resolved "
            f"against {MODEL_WEIGHT_DIR}."
        ),
    )
    parser.add_argument(
        "dataset",
        type=str.upper,
        choices=list(RUNNERS),
        help="Dataset to test with.",
    )
    parser.add_argument(
        "--all-sensors",
        action="store_true",
        help="Let all sensors of the dataset vote instead of only the training sensor.",
    )
    return parser.parse_args()


def evaluate(weight_dir, dataset):
    """Test the ensembles in `weight_dir` on `dataset`, returns each ensemble's accuracy."""
    if VOTING_TYPE not in ("soft", "hard"):
        raise ValueError(f"Unknown VOTING_TYPE '{VOTING_TYPE}', use 'soft' or 'hard'")
    print(f"Voting: {VOTING_TYPE}, sensors: AGFD {AGFD_SENSORS}, MCC5 {MCC5_SENSORS}")

    check_checkpoint_available(weight_dir)
    apply_training_config(weight_dir)

    model = setup_model(config, device)
    model.eval()

    return RUNNERS[dataset](model, weight_dir)


def main():
    args = parse_args()

    weight_dir = MODEL_WEIGHT_DIR / args.weight_dir
    if not weight_dir.is_dir():
        raise SystemExit(f"Weight directory not found: {weight_dir}")

    if args.all_sensors:
        use_all_sensors()
    evaluate(weight_dir, args.dataset)


if __name__ == "__main__":
    main()
