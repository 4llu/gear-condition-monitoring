import torch
import numpy as np


def fig_to_tensor(fig):
    """
    Convert matplotlib figure to tensor for TensorBoard.
    """

    fig.canvas.draw()
    data = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
    return torch.from_numpy(data.transpose(2, 0, 1))


def fix_embedding_labels(batch_labels, config):
    """
    Pick out batch labels that match the query samples.
    """

    batch_labels = batch_labels.reshape(
        batch_labels.shape[0] // (config["k_shot"] + config["n_query"]),
        config["k_shot"] + config["n_query"],
    )[:, config["k_shot"] :].flatten()

    return batch_labels
