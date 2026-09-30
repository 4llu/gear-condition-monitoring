import torch
import torch.nn.functional as F
from einops import rearrange, reduce
from pytorch_metric_learning import distances, losses, miners
from torch import nn

from src.training.utils import fix_embedding_labels


class Embedding(nn.Module):
    def __init__(self, backbone, config, device):
        super().__init__()
        self.backbone = backbone
        self.config = config
        self.device = device

        # Loss
        ##

        self.miner = None
        self.loss_fn = None
        self.eps = torch.finfo(torch.float32).eps

        # Similarity metric
        if self.config["loss"] in ["contrastive", "triplet", "supcon"]:
            if self.config["similarity"] == "cosine":
                self.distance_fn = distances.CosineSimilarity()
            elif self.config["similarity"] == "euclidean":
                self.distance_fn = distances.LpDistance(
                    normalize_embeddings=self.config["lpnorm_embeddings_training"],
                    p=2,
                )
            else:
                raise ValueError(
                    f"Unknown similarity type '{self.config['similarity']}'!"
                )

        # NCA
        if self.config["loss"] == "NCA":
            self.loss_fn = losses.NCALoss(
                distance=distances.LpDistance(
                    normalize_embeddings=self.config["lpnorm_embeddings_training"],
                    p=2,
                    power=2,  # NOTE: **Squared** Euclidean distance
                ),
                softmax_scale=1
                / self.config["loss_args"][
                    "temperature"
                ],  # Inverted to get scale from temperature
            )
        # Contrastive
        elif self.config["loss"] == "contrastive":
            self.loss_fn = losses.ContrastiveLoss(
                distance=self.distance_fn,
                pos_margin=self.config["loss_args"]["pos_margin"],
                neg_margin=self.config["loss_args"]["neg_margin"],
            )

            if self.config["loss_args"]["miner"]:
                self.miner = miners.PairMarginMiner(
                    pos_margin=self.config["loss_args"]["pos_margin"],
                    neg_margin=self.config["loss_args"]["neg_margin"],
                )
        # Triplet
        elif self.config["loss"] == "triplet":
            # NOTE: Without `distance`, pytorch-metric-learning defaults to
            # LpDistance(normalize_embeddings=True), ignoring `similarity` and
            # `lpnorm_embeddings_training`
            # sum_reducer = reducers.SumReducer()  # XXX
            self.loss_fn = losses.TripletMarginLoss(
                margin=self.config["loss_args"]["margin"],
                distance=self.distance_fn,
                # reducer=sum_reducer,
            )

            if self.config["loss_args"]["miner"]:
                self.miner = miners.TripletMarginMiner(
                    type_of_triplets=self.config["loss_args"]["miner_type"],
                    margin=self.config["loss_args"]["margin"],
                    distance=self.distance_fn,
                )
        # Circle loss
        elif self.config["loss"] == "circle":
            # NOTE: Always uses cosine similarity
            self.loss_fn = losses.CircleLoss(
                m=self.config["loss_args"]["m"],
                gamma=self.config["loss_args"]["gamma"],
            )
        # Supervised contrastive
        elif self.config["loss"] == "supcon":
            self.loss_fn = losses.SupConLoss(
                distance=self.distance_fn,
                temperature=self.config["loss_args"]["temperature"],
            )
        # Prototypical (CrossEntropy)
        elif self.config["loss"] == "proto_loss":
            self.loss_fn = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown loss type '{self.config['loss']}'!")

        self.loss_fn = self.loss_fn.to(self.device)

    def forward(self, x):
        # Input shape: (n_way, k_shot + n_query, input_size)
        orig_shape = x.shape

        # NOTE: Non-episodic training doesn't have a 3rd dimension
        if len(orig_shape) == 3:
            # Flatten
            x = rearrange(x, "w n i -> (w n) i")

        # Compute embeddings
        x = self.backbone(x)

        # Return to original shape
        if len(orig_shape) == 3:
            x = rearrange(
                x,
                "(w n) e -> w n e",
                w=orig_shape[0],
                n=orig_shape[1],
            )

        return x

    def loss(self, embeddings, labels):
        if self.config["loss"] == "proto_loss":
            y_pred = self.predict(embeddings)
            labels = fix_embedding_labels(labels, self.config)
            return self.loss_fn(y_pred, labels)
        else:
            # NOTE: Non-episodic training doesn't have a 3rd dimension
            if len(embeddings.shape) == 3:
                embeddings = rearrange(embeddings, "w n e -> (w n) e")

            if self.miner is None:
                return self.loss_fn(embeddings, labels)
            else:
                # Use the miner to get pairs/triplets
                miner_output = self.miner(embeddings, labels)
                return self.loss_fn(embeddings, labels, miner_output)

    def predict(self, embeddings):
        # Separate supports
        ##

        # (n_way, k_shot, embedding_len)
        supports = embeddings[:, : self.config["k_shot"], :]
        # Average over k_shot
        prototypes = reduce(supports, "w k e -> w e", "mean")

        # Separate queries
        ##

        # (n_way, n_query, embedding_len)
        queries = embeddings[:, self.config["k_shot"] :, :]
        # Flatten
        queries = rearrange(queries, "w q e -> (w q) e")

        # Normalize
        ##

        if self.config["center_embeddings_inference"]:
            # NOTE: same as mean of supports, as there are the same amount from each class
            mean = torch.mean(prototypes, dim=0)
            prototypes = prototypes - mean
            queries = queries - mean

        if (
            self.config["similarity"] == "cosine"
            or self.config["lpnorm_embeddings_inference"]
        ):
            prototypes = F.normalize(prototypes, p=2, dim=-1)
            queries = F.normalize(queries, p=2, dim=-1)

        # Distances
        ##

        # Euclidean
        if self.config["similarity"] == "euclidean":
            # Each query to each prototype
            distances = queries.unsqueeze(1) - prototypes.unsqueeze(0)
            # ((n_way * n_query), n_way)
            distances = (
                torch.sum(distances * distances, dim=-1).clamp(min=self.eps).sqrt()
            )

            # y_pred = torch.argmin(distances, dim=-1)
            return -1 * distances
        # Cosine
        elif self.config["similarity"] == "cosine":
            similarities = torch.einsum("q i, p i -> q p", queries, prototypes)

            # y_pred = torch.argmax(similarities, dim=-1)
            return similarities
        else:
            raise ValueError(f"Unknown similarity type '{self.config['similarity']}'!")
