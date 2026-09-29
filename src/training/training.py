import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytorch_warmup as warmup
import torch
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from torch.utils.tensorboard import SummaryWriter
from zclip import ZClip

from src.models.models import setup_model
from src.training.utils import fix_embedding_labels

log = logging.getLogger("gear-cm")

# HELPERS
#########


def _get_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1.0 / 2)
    return total_norm


class autoclip_gradient:
    def __init__(self, clip_percentile=95):
        self.clip_percentile = clip_percentile
        self.grad_history = []

    def __call__(self, model):
        obs_grad_norm = _get_grad_norm(model)
        self.grad_history.append(obs_grad_norm)
        clip_value = np.percentile(self.grad_history, self.clip_percentile)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)


# MAIN TRAINING FUNCTION
########################


def run_single_training(
    train_loaders,
    validation_loader,
    test_loader,
    config,
    device,
    trial=None,
    model_weight_dir=None,
):
    run_start_time = datetime.now().strftime("%m-%d_%H-%M-%S")

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
    # optimizer = torch.optim.RAdam(
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        betas=[config["momentum"], config["b2"]],
    )

    # Initialize ZClip
    zclip = ZClip(alpha=0.97, z_thresh=2.5)
    # auto_clip = autoclip_gradient(clip_percentile=10)

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

    # loss_fn = torch.nn.CrossEntropyLoss()
    # loss_fn = loss_fn.to(device=device)

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

        # Add model graph to TensorBoard
        ##

        # FIXME: Temp solution to only use the first one here for now
        # sample_input = next(train_loaders[0])[0].to(device)
        # TB_writer.add_graph(model, sample_input, verbose=False)

        # torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Add raw samples as embeddings to TensorBoard
        ##

        # FIXME: Broken for now (because of anchors)
        # embeddings = []
        # labels = []
        # for _ in range(100):
        #     try:
        #         samples, batch_labels = next(train_loaders[0])
        #     except KeyError as e:
        #         log.warning(f"No operating condition matching to: {e}")
        #         continue
        #     embeddings.append(samples)
        #     labels.append(torch.round(batch_labels).to(torch.int32))

        # # TODO Add speed and load to metadata when added to samples
        # embeddings = torch.cat(embeddings, dim=0).cpu().detach().numpy()
        # embeddings = embeddings.reshape(embeddings.shape[0] * embeddings.shape[1], embeddings.shape[2])
        # labels = torch.cat(labels, dim=0).cpu().detach().numpy()

        # # Add embeddings to Tensorboard
        # TB_writer.add_embedding(embeddings, metadata=labels, tag="raw_input_embeddings")

    # TRAINING TRACKING
    #########################

    best_epoch = 0
    best_val_loss = 99999999999.9
    # NOTE: Technically not best, but one corresponding to best loss
    best_val_accuracy = 0.0
    best_test_loss = 99999999999.9
    # NOTE: Technically not best, but one corresponding to best loss
    best_test_accuracy = 0.0
    running_test_accuracy = []

    # TODO: Implement early stopping
    patience_counter = 0

    # TODO: Model weight saving
    if model_weight_dir is None:
        model_weight_dir = (
            Path(__file__).resolve().parent.parent.parent / "model_weights"
        )
    model_weight_dir.mkdir(parents=True, exist_ok=True)
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
        train_accuracy = 0.0

        for train_loader in train_loaders:
            for _ in range(config["episodes_per_dataset_per_batch"]):
                # Get batch
                samples, labels, _ = next(train_loader)

                # for i in range(len(samples)):
                #     plt.plot(samples[i, 0, :].cpu().numpy())
                # plt.show()

                # Results and loss
                outputs = model(samples)

                # outputs[:5, :] = 0
                # zero_healthy = torch.zeros_like(outputs, device=device)[:5, :]
                # outputs = torch.concat([zero_healthy, outputs], dim=0)
                # labels = torch.concat(
                #     [torch.zeros_like(labels, device=device)[:5], labels], dim=0
                # )

                # Check for NaN in outputs
                if torch.isnan(outputs).any():
                    log.error("NaN values detected in model outputs")
                    raise ValueError("NaN values detected in model outputs")

                loss = model.loss(outputs, labels)

                # # XXX
                # # Divide loss to get multiple episodes per dataset for
                # # each "batch"
                # loss = loss / (
                #     len(train_loaders) * config["episodes_per_dataset_per_batch"]
                # )
                # loss.backward()
                # full_loss += loss.item() / samples.shape[0]
                # # XXX

                # YYY
                loss = (
                    loss
                    / (
                        samples.shape[0]  # * config["episodes_per_dataset_per_batch"]
                    )
                )
                loss.backward()
                zclip.step(model)  # XXX
                # auto_clip(model)  # XXX
                # HERE: Try the other clipping method https://github.com/pseeth/autoclip/blob/master/autoclip.py
                full_loss += loss.item()
                # YYY

                # Accuracy
                # logits = model.predict(outputs)
                # y_pred = torch.argmax(logits, dim=-1)
                # train_accuracy += (y_pred == query_labels).float().mean()

        # First, last and every 5th batch
        if batch_i % 5 == 0 or batch_i == config["max_batches"] - 1:
            # Check for NaN in loss
            if np.isnan(full_loss):
                log.error("NaN values detected in loss calculation")
                raise ValueError("NaN values detected in loss calculation")

            # Average accuracy
            # train_accuracy = (
            #     train_accuracy
            #     / (len(train_loaders) * config["episodes_per_dataset_per_batch"])
            # ) * 100

            # Logging
            log.debug(f"TRAIN batch {batch_i + 1} - loss: {full_loss:.6f}")
            # log.debug(f"TRAIN batch {batch_i + 1} - accuracy: {train_accuracy:.2f}")

            # Tensorboard logging
            if TB_writer:
                # Track training loss
                TB_writer.add_scalar("train/loss", full_loss, batch_i)
                # Track training accuracy
                TB_writer.add_scalar("train/accuracy", train_accuracy, batch_i)
                # Track learning rate
                TB_writer.add_scalar(
                    "z-other/lr", optimizer.param_groups[0]["lr"], batch_i
                )

                # # Log model parameter weights and biases to TensorBoard
                # for tag, value in model.named_parameters():
                #     TB_writer.add_histogram(
                #         "weights/" + tag, value.detach().cpu(), batch_i
                #     )
                #     # Keep gradient logging as well
                #     if value.grad is not None:
                #         TB_writer.add_histogram(
                #             "grad/" + tag, value.grad.cpu(), batch_i
                #         )

        if config["save"] and batch_i % 100 == 0:
            run_dir = model_weight_dir / run_start_time
            run_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                model.backbone.state_dict(),
                model_weight_dir / run_dir / f"{checkpoint_i}.pth",
            )
            checkpoint_i += 1

        # Backward pass
        # * Skip first batch learning to get a baseline
        if batch_i != 0:
            # Gradient clipping
            if config["gradient_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=config["gradient_clip"],
                    norm_type=2,
                )

            # Backpropagation and weight update
            # print(">> Step")
            optimizer.step()
            # optimizer.zero_grad(set_to_none=True)

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

        # First, last, and every validation_interval batches
        # NOTE: Only if validation_loader exists (i.e. not a hyperparameter searach run)
        if validation_loader is not None and (
            batch_i == 0
            or batch_i % config["validation_interval"] == 0
            or batch_i == config["max_batches"] - 1
        ):
            # Reset
            model.eval()
            val_loss = 0.0
            val_accuracy = 0.0

            with torch.no_grad():
                # Only one dataset at a time supported for validation
                for _ in range(config["validation_batches"]):
                    samples, labels, _ = next(validation_loader)

                    outputs = model(samples)

                    # Loss
                    loss = model.loss(outputs, labels)
                    val_loss += loss.item()

                    # Drop support labels if using embeddings
                    if config["model"] == "classical":
                        query_labels = labels
                    else:
                        query_labels = fix_embedding_labels(labels, config)

                    # Accuracy
                    logits = model.predict(outputs)
                    y_pred = torch.argmax(logits, dim=-1)

                    val_accuracy += (y_pred == query_labels).float().mean()

            # Average over batches
            val_loss = val_loss / config["validation_batches"]
            val_accuracy = (val_accuracy / config["validation_batches"]) * 100

            # Save best
            if val_accuracy > best_val_accuracy:
                best_val_loss = val_loss
                best_val_accuracy = val_accuracy
                best_epoch = batch_i

                # Save model weights
                # TODO: Save model weights

            # Logging
            log.debug("")
            log.debug(f"\x1b[36;20mVALIDATION\x1b[0m batch {batch_i + 1}")
            log.debug(
                f"\x1b[36;20mVALIDATION\x1b[0m Loss: \x1b[1m{val_loss:>15.2f}\x1b[0m"
            )
            log.debug(
                f"\x1b[36;20mVALIDATION\x1b[0m Accuracy: \x1b[1m{val_accuracy:>11.2f}%\x1b[0m"
            )
            log.debug("")

            # Tensorboard logging
            ##
            if TB_writer:
                TB_writer.add_scalar("validation/loss", val_loss, batch_i)
                TB_writer.add_scalar("validation/accuracy", val_accuracy, batch_i)

            # For last batch, print samples of predictions and labels
            if batch_i == config["max_batches"] - 1:
                log.debug("")
                log.debug("VALIDATION")
                log.debug("Predictions:")
                log.debug(torch.round(outputs.cpu(), decimals=3))
                log.debug("")
                log.debug("Real labels:")
                log.debug(labels)

        ########
        # TEST #
        ########

        # First, last, and every test_interval batches
        if (
            batch_i % config["test_interval"] == 0
            or batch_i == config["max_batches"] - 1
        ):
            # Reset
            model.eval()
            test_loss = 0.0
            test_accuracy = 0.0

            test_AMI = 0.0
            test_precision_at_1 = 0.0
            test_r_precision = 0.0

            with torch.no_grad():
                for _ in range(config["test_batches"]):
                    # Only one dataset at a time supported for validation
                    samples, labels, _ = next(test_loader)

                    outputs = model(samples)

                    # Loss
                    loss = model.loss(outputs, labels)
                    test_loss += loss.item()

                    # XXX
                    # Drop support labels if using embeddings
                    # if config["model"] == "classical":
                    #     query_labels = labels
                    # else:
                    #     query_labels = fix_embedding_labels(labels, config)

                    # Accuracy

                    # logits = model.predict(outputs)
                    # y_pred = torch.argmax(logits, dim=-1)
                    # test_accuracy += (y_pred == query_labels).float().mean()
                    # XXX
                    # print("Outputs:", outputs.shape)
                    # print("Query labels:", query_labels.shape)
                    # print("Labels:", labels.shape)
                    # reshaped_outputs = outputs.reshape(
                    # outputs.shape[0] * outputs.shape[1], outputs.shape[2]
                    # )
                    # print("Reshaped outputs:", reshaped_outputs.shape)
                    test_metrics = AccCalc.get_accuracy(
                        outputs, labels, outputs, labels, True
                    )

                    test_AMI += test_metrics["AMI"]
                    test_precision_at_1 += test_metrics["precision_at_1"]
                    test_r_precision += test_metrics["r_precision"]

                    # print(test_metrics)
                    # quit()
            # Average over batches
            test_loss = test_loss / config["test_batches"]
            test_accuracy = (test_accuracy / config["test_batches"]) * 100

            test_AMI = test_AMI / config["test_batches"]
            test_precision_at_1 = test_precision_at_1 / config["test_batches"]
            test_r_precision = test_r_precision / config["test_batches"]

            test_accuracy = test_r_precision * 100  # XXX

            running_test_accuracy.append(test_accuracy)
            if len(running_test_accuracy) > 3:
                # Remove the oldest accuracy value
                # NOTE: This is used to smooth the accuracy curve
                running_test_accuracy.pop(0)

            # Save best
            # NOTE: Independent of validation
            # FIXME: This should probably be handled differently (tied to validation)
            if (
                sum(running_test_accuracy) / len(running_test_accuracy)
                > best_test_accuracy
            ):
                # if test_accuracy > best_test_accuracy:
                best_test_loss = test_loss
                best_test_accuracy = sum(running_test_accuracy) / len(
                    running_test_accuracy
                )
                best_epoch = batch_i

            # For Optuna trials
            if trial is not None:
                trial.report(
                    sum(running_test_accuracy) / len(running_test_accuracy), batch_i
                )

                # If the trial is pruned, stop the training
                if trial.should_prune():
                    # Imported here to not be required for normal runs
                    import optuna

                    raise optuna.TrialPruned()

            # Logging
            log.debug(f"\x1b[35;20mTEST\x1b[0m batch {batch_i + 1}")
            log.debug(f"\x1b[35;20mTEST\x1b[0m Loss: \x1b[1m{test_loss:>21.2f}\x1b[0m")
            log.debug(
                f"\x1b[35;20mTEST\x1b[0m Accuracy: \x1b[1m{test_accuracy:>17.2f}%\x1b[0m"
            )
            log.debug("")

            # For last batch, print samples of predictions and labels
            if batch_i == config["max_batches"] - 1:
                log.debug("")
                log.debug("TEST")
                log.debug("Predictions:")
                log.debug(torch.round(outputs.cpu(), decimals=3))
                log.debug("")
                log.debug("Real labels:")
                log.debug(labels)

            # Tensorboard logging
            ##
            if TB_writer:
                TB_writer.add_scalar("test/loss", test_loss, batch_i)
                TB_writer.add_scalar("test/accuracy", test_accuracy, batch_i)

                TB_writer.add_scalar("test/AMI", test_AMI, batch_i)
                TB_writer.add_scalar(
                    "test/precision_at_1", test_precision_at_1, batch_i
                )
                TB_writer.add_scalar("test/r_precision", test_r_precision, batch_i)

    # Ensure tensorboard is closed correctly
    if TB_writer:
        TB_writer.flush()
        TB_writer.close()

    # Used for hyperparameter search
    return best_test_accuracy
