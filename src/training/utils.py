def fix_embedding_labels(batch_labels, config):
    """
    Pick out batch labels that match the query samples.
    """

    batch_labels = batch_labels.reshape(
        batch_labels.shape[0] // (config["k_shot"] + config["n_query"]),
        config["k_shot"] + config["n_query"],
    )[:, config["k_shot"] :].flatten()

    return batch_labels
