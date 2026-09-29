import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce


class Prototypical(nn.Module):
    def __init__(self, backbone, config):
        super(Prototypical, self).__init__()
        self.config = config
        self.backbone = backbone
        self.eps = torch.finfo(torch.float32).eps

        assert config["similarity"] in [
            "euclidean",
            "cosine",
        ], "Unknown similarity type '{config[\"similarity\"]}'!"

        assert config["similarity"] == "euclidean" or (
            config["similarity"] == "cosine" and config["lp_norm"] == 2
        ), "Cosine similarity only works with L2 normalization!"

    def forward(self, x):
        # Input shape: (n_way, k_shot + n_query, input_size)
        orig_shape = x.shape

        # Reshape to fit (batch, features) shape
        x = rearrange(x, "w n i -> (w n) i")

        # COMPUTE EMBEDDINGS
        ####################

        # Compute embeddings
        x = self.backbone(x)
        # Return to original shape (except feature length is now embedding length)
        x = rearrange(
            x,
            "(w n) e -> w n e",
            w=orig_shape[0],
            n=orig_shape[1],
        )

        # SUPPORT EMBEDDING PROCESSING
        ##############################

        # Separate supports
        # (n_way, k_shot, embedding_len)
        supports = x[:, : self.config[f"k_shot"], :]
        # Average over k_shot
        prototypes = reduce(supports, "w k e -> w e", "mean")

        # QUERY EMBEDDING PROCESSING
        ############################

        # Separate queries
        # (n_way, n_query, embedding_len)
        queries = x[:, self.config[f"k_shot"] :, :]
        # Flatten
        queries = rearrange(queries, "w q e -> (w q) e")

        # NORMALIZATION
        ###############

        if self.config["lp_norm"] > 0:
            prototypes = F.normalize(prototypes, p=self.config["lp_norm"], dim=-1)
            queries = F.normalize(queries, p=self.config["lp_norm"], dim=-1)

        # DISTANCES
        ###########

        # Euclidean
        ##

        if self.config["similarity"] == "euclidean":
            # queries = self.config["embedding_multiplier"] * queries
            # prototypes = self.config["embedding_multiplier"] * prototypes

            # Calculate element-wise distances
            # Unsqueezes to get broadcasting to work
            # (n_way * n_query, 1, embedding_len)
            # (1, k_shot, embedding_len)
            # = (n_way * n_query, k_shot, embedding_len)
            distances = queries.unsqueeze(1) - prototypes.unsqueeze(0)
            # Euclidean distance
            distances = torch.sum(distances * distances, dim=-1).sqrt()
            # Flip to get similarity
            similarities = -1 * distances

        # Cosine
        ##

        if self.config["similarity"] == "cosine":
            # Dot product to get cosine similarity
            # (L2 normalization required beforehand)
            # Results in shape (n_way * n_queries, n_way)
            cos_ij = torch.einsum("q i, s i -> q s", queries, prototypes)
            similarities = cos_ij

        return similarities
