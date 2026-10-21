import logging
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pytorch_warmup as warmup
import torch
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from ruamel.yaml import YAML
from torch.utils.tensorboard import SummaryWriter

from src.models.models import setup_model

log = logging.getLogger("gear-cm")

# Weights are saved every CHECKPOINT_INTERVAL batches as 0.pth, 1.pth, ...
CHECKPOINT_INTERVAL = 100


def last_checkpoint_num(max_batches):
    """Number of the last checkpoint a training with `max_batches` batches saves."""
    return (max_batches - 1) // CHECKPOINT_INTERVAL

# HELPERS
#########


def _get_grad_norm(model):
    """Total L2 norm of all gradients, same as what clip_grad_norm_ clips against."""
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1.0 / 2)
    return total_norm


def _get_git_state():
    repo_dir = Path(__file__).resolve().parent.parent.parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_dir,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def _save_run_info(run_dir, config):
    """Store everything needed to reproduce the run next to its weights."""
    run_info = {
        "seed": config.get("seed"),
        "git": _get_git_state(),
        "config": config,
    }
    yaml = YAML()
    with open(run_dir / "run_info.yaml", "w") as stream:
        yaml.dump(run_info, stream)


# MAIN TRAINING FUNCTION
########################


def run_single_training(
    train_loaders,
    validation_loader,
    config,
    device,
    model_weight_dir=None,
):
    """
    Train one model. `validation_loader` (or None) is only monitored: its loss and
    retrieval metrics are logged, no checkpoint is selected with it.
    """
    run_start_time = datetime.now().strftime("%m-%d_%H-%M-%S")  # noqa: DTZ005

    # Use if debugging NaN errors
    # NOTE: This is a bit slow, so only use if necessary
    # torch.autograd.set_detect_anomaly(True)

    # MODEL INITIALIZATION
    ######################

    model = setup_model(config, device)

    # TODO: print model summary?

    # TRAINING INITIALIZATION
    #########################

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        betas=[config["momentum"], config["b2"]],
    )


    # Learning rate scheduler
    # TODO: Cosine annealing with warm restarts?
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        config["sch_gamma"],
    )

    # Warmup scheduler
    warmup_scheduler = None
    if config["warmup_batches"] > 0:
        warmup_scheduler = warmup.LinearWarmup(
            optimizer, warmup_period=config["warmup_batches"]
        )

    AccCalc = AccuracyCalculator(k="max_bin_count")

    # TENSORBOARD
    #############

    TB_writer = None
    if config["log"]:
        # Initialization
        ##

        # Check and create TB log directory
        TB_log_dir = Path(__file__).resolve().parent.parent.parent / "logs" / "TB_logs"
        TB_log_dir.mkdir(parents=True, exist_ok=True)

        # Create TB writer
        run_name = (
            f"{config['name']}-{config['model']}-{config['backbone']}-{run_start_time}"
        )
        TB_writer = SummaryWriter(
            log_dir=TB_log_dir / run_name,
        )

    # TRAINING TRACKING
    #########################

    if model_weight_dir is None:
        model_weight_dir = (
            Path(__file__).resolve().parent.parent.parent / "model_weights"
        )
    model_weight_dir.mkdir(parents=True, exist_ok=True)
    # Seed in the name keeps parallel runs started in the same second apart
    run_dir = model_weight_dir / f"{run_start_time}_s{config.get('seed')}"
    if config["save"]:
        run_dir.mkdir(parents=True, exist_ok=True)
        _save_run_info(run_dir, config)
    checkpoint_i = 0

    # TRAINING LOOP
    ###############

    # Batch loop (No separate epoch loop)
    for batch_i in range(config["max_batches"]):
        # Reset
        model.train()
        optimizer.zero_grad(set_to_none=True)

        # Forward pass
        full_loss = 0

        for train_loader in train_loaders:
            for _ in range(config["episodes_per_dataset_per_batch"]):
                # Get batch
                samples, labels, _ = next(train_loader)

                # Results and loss
                outputs = model(samples)

                # Check for NaN in outputs
                if torch.isnan(outputs).any():
                    log.error("NaN values detected in model outputs")
                    raise ValueError("NaN values detected in model outputs")

                loss = model.loss(outputs, labels)

                # Per-sample loss; gradients accumulate over datasets and episodes
                loss = loss / samples.shape[0]
                loss.backward()
                full_loss += loss.item()

        # First, last and every 5th batch
        if batch_i % 5 == 0 or batch_i == config["max_batches"] - 1:
            # Check for NaN in loss
            if np.isnan(full_loss):
                log.error("NaN values detected in loss calculation")
                raise ValueError("NaN values detected in loss calculation")

            # Logging
            log.debug(f"TRAIN batch {batch_i + 1} - loss: {full_loss:.6f}")

            # Tensorboard logging
            if TB_writer:
                # Track training loss
                TB_writer.add_scalar("train/loss", full_loss, batch_i)
                # Track learning rate
                TB_writer.add_scalar(
                    "z-other/lr", optimizer.param_groups[0]["lr"], batch_i
                )

        if config["save"] and batch_i % CHECKPOINT_INTERVAL == 0:
            torch.save(
                model.backbone.state_dict(),
                run_dir / f"{checkpoint_i}.pth",
            )
            checkpoint_i += 1

        # Backward pass
        # * Skip first batch learning to get a baseline
        if batch_i != 0:
            # Gradient clipping (of the whole batch's accumulated gradient)
            # * clip_grad_norm_ returns the total norm before clipping
            if config["gradient_clip"] > 0:
                grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=config["gradient_clip"],
                    norm_type=2,
                )
            else:
                grad_norm_before_clip = None

            # Same cadence as the training loss logging
            if TB_writer and (
                batch_i % 5 == 0 or batch_i == config["max_batches"] - 1
            ):
                # Norm of the gradient actually used for the update
                grad_norm = _get_grad_norm(model)
                if grad_norm_before_clip is None:
                    # No clipping, so the norm is unchanged
                    grad_norm_before_clip = grad_norm
                TB_writer.add_scalar(
                    "train/grad_norm_before_clip", grad_norm_before_clip, batch_i
                )
                TB_writer.add_scalar("train/grad_norm", grad_norm, batch_i)

            # Backpropagation and weight update
            optimizer.step()

        # Learning rate control
        if warmup_scheduler is not None:
            with warmup_scheduler.dampening():
                if warmup_scheduler.last_step + 1 >= config["warmup_batches"]:
                    scheduler.step()
        elif scheduler is not None:
            scheduler.step()

        ##############
        # VALIDATION #
        ##############

        # First, last, and every validation_interval batches (monitoring only)
        if validation_loader is not None and (
            batch_i == 0
            or batch_i % config["validation_interval"] == 0
            or batch_i == config["max_batches"] - 1
        ):
            model.eval()
            val_loss = 0.0
            val_metrics = {"precision_at_1": 0.0, "r_precision": 0.0, "AMI": 0.0}

            with torch.no_grad():
                for _ in range(config["validation_batches"]):
                    samples, labels, _ = next(validation_loader)
                    outputs = model(samples)

                    val_loss += model.loss(outputs, labels).item()

                    # Retrieval metrics within the batch, every sample against all others
                    metrics = AccCalc.get_accuracy(outputs, labels, outputs, labels, True)
                    for key in val_metrics:
                        val_metrics[key] += metrics[key]

            val_loss /= config["validation_batches"]
            val_metrics = {
                key: value / config["validation_batches"]
                for key, value in val_metrics.items()
            }

            log.debug("")
            log.debug(f"\x1b[36;20mVALIDATION\x1b[0m batch {batch_i + 1}")
            log.debug(
                f"\x1b[36;20mVALIDATION\x1b[0m Loss: \x1b[1m{val_loss:>15.4f}\x1b[0m"
            )
            log.debug(
                f"\x1b[36;20mVALIDATION\x1b[0m Precision at 1: "
                f"\x1b[1m{val_metrics['precision_at_1'] * 100:>6.2f}%\x1b[0m"
            )
            log.debug("")

            if TB_writer:
                TB_writer.add_scalar("validation/loss", val_loss, batch_i)
                for key, value in val_metrics.items():
                    TB_writer.add_scalar(f"validation/{key}", value, batch_i)

    # Ensure tensorboard is closed correctly
    if TB_writer:
        TB_writer.flush()
        TB_writer.close()
