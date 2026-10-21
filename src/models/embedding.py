from einops import rearrange
from pytorch_metric_learning import distances, losses, miners
from torch import nn


class Embedding(nn.Module):
    """Backbone that maps a sample to an embedding, trained with a triplet loss."""

    def __init__(self, backbone, config, device):
        super().__init__()
        self.backbone = backbone
        self.config = config
        self.device = device

        if self.config["loss"] != "triplet":
            raise ValueError(f"Unknown loss '{self.config['loss']}', only 'triplet' is supported")

        # Similarity metric
        if self.config["similarity"] == "cosine":
            self.distance_fn = distances.CosineSimilarity()
        elif self.config["similarity"] == "euclidean":
            self.distance_fn = distances.LpDistance(
                normalize_embeddings=self.config["lpnorm_embeddings_training"],
                p=2,
            )
        else:
            raise ValueError(f"Unknown similarity type '{self.config['similarity']}'!")

        # NOTE: Without `distance`, pytorch-metric-learning defaults to
        # LpDistance(normalize_embeddings=True), ignoring `similarity` and
        # `lpnorm_embeddings_training`
        self.loss_fn = losses.TripletMarginLoss(
            margin=self.config["loss_args"]["margin"],
            distance=self.distance_fn,
        ).to(self.device)

        self.miner = None
        if self.config["loss_args"]["miner"]:
            self.miner = miners.TripletMarginMiner(
                type_of_triplets=self.config["loss_args"]["miner_type"],
                margin=self.config["loss_args"]["margin"],
                distance=self.distance_fn,
            )

    def forward(self, x):
        # Input shape: (batch_size, input_size), or (batch_size, n, input_size) at test
        # time, where n is the number of anchors
        orig_shape = x.shape

        if len(orig_shape) == 3:
            x = rearrange(x, "b n i -> (b n) i")

        x = self.backbone(x)

        if len(orig_shape) == 3:
            x = rearrange(x, "(b n) e -> b n e", b=orig_shape[0], n=orig_shape[1])

        return x

    def loss(self, embeddings, labels):
        if len(embeddings.shape) == 3:
            embeddings = rearrange(embeddings, "b n e -> (b n) e")

        if self.miner is None:
            return self.loss_fn(embeddings, labels)

        # Use the miner to get the triplets
        miner_output = self.miner(embeddings, labels)
        return self.loss_fn(embeddings, labels, miner_output)
