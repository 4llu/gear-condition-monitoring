from torch import nn
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange


class Normalize(nn.Module):
    def __init__(self, p=2, dim=-1):
        super(Normalize, self).__init__()
        self.normalize = nn.functional.normalize
        self.p = p
        self.dim = dim

    def forward(self, x):
        x = self.normalize(x, p=self.p, dim=self.dim)
        return x


class MlpBlock(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MixerBlock(nn.Module):
    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout):
        super().__init__()

        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange("b s c -> b c s"),
            MlpBlock(num_patch, token_dim, dropout),
            Rearrange("b c s -> b s c"),
        )

        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            MlpBlock(dim, channel_dim, dropout),
        )

    def forward(self, x):
        x = x + self.token_mix(x)  # x + ... because of residual connection
        x = x + self.channel_mix(x)  # x + ... because of residual connection

        return x


class MlpMixer(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Each GMF is a patch
        self.num_patch = config["gear_harmonics_to_include"]
        # (higher + lower sidebands) * symmetrical + GMF itself
        self.patch_size = config["model_args"].get(
            "patch_size", config.get("gear_sideband_block_size", 0) * 2 * 2 + 1
        )

        # Configurable
        self.hidden_dim = config["model_args"]["hidden_dim"]
        self.tokens_mlp_dim = config["model_args"]["tokens_mlp_dim"]
        self.channels_mlp_dim = config["model_args"]["channels_mlp_dim"]

        # Stem (linear projection of patches)
        self.linear_project_patch = nn.Linear(self.patch_size, self.hidden_dim)

        # Create blocks/layers for feature extraction
        blocks = []
        for _ in range(config["model_args"]["num_layers"]):
            blocks.append(
                MixerBlock(
                    self.hidden_dim,
                    self.num_patch,  # "channels"
                    self.tokens_mlp_dim,
                    self.channels_mlp_dim,
                    config["dropout"],
                )
            )
        self.features = nn.Sequential(*blocks)

        # Classification/embedding head
        self.head = [
            nn.LayerNorm(self.hidden_dim),
            Reduce("b s c -> b c", reduction="mean"),  # GlobalAvgPool over patches
        ]
        # NOTE: Only for SupCon
        if config["model_args"]["intermediate_L2"]:
            self.head.append(Normalize(p=2, dim=-1))

        self.head = nn.Sequential(
            *self.head,
            # nn.Linear(self.hidden_dim, 24, bias=False),
            # nn.GELU(),
            nn.Linear(self.hidden_dim, config["embedding_len"], bias=False),
        )
        # self.head = nn.Sequential(
        #     nn.LayerNorm(self.hidden_dim),
        #     Reduce("b s c -> b c", reduction="mean"),  # GlobalAvgPool over patches
        #     nn.Linear(self.hidden_dim, config["embedding_len"], bias=False),
        # )

    def forward(self, x):
        # Input shape: (batch_size, input_len)

        # 1. Get rid of the channel dimension (should always be 1 for this model)
        # 2. Separate GMFs into separate patches
        # (batch_size, num_patch, patch_size)
        x = rearrange(x, "b (s c) -> b s c", s=self.num_patch, c=self.patch_size)

        # Patch projection
        # (batch_size, num_patch, hidden_dim)
        x = self.linear_project_patch(x)

        # Blocks
        # (batch_size, num_patch, hidden_dim)
        x = self.features(x)

        # NOTE: Only use for SupCon
        # x = nn.functional.normalize(x, p=2, dim=-1)

        # Classification/embedding head
        # (batch_size, embedding_len)
        x = self.head(x)

        return x
