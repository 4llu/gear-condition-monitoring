from einops import rearrange
from torch import nn


class Classical(nn.Module):
    def __init__(self, backbone, config, device):
        super(Classical, self).__init__()

        self.backbone = backbone
        self.config = config
        self.device = device

        self.loss_fn = nn.CrossEntropyLoss().to(self.device)

    def forward(self, x):
        # Input shape: (n_way, k_shot + n_query, input_size)
        orig_shape = x.shape

        # Flatten
        x = rearrange(x, "w n i -> (w n) i")

        # Actual model
        x = self.backbone(x)

        # Return to original shape
        x = rearrange(x, "(w n) e -> w n e", w=orig_shape[0], n=orig_shape[1])

        return x

    def loss(self, embeddings, labels):
        return self.loss_fn(embeddings.view(-1, embeddings.shape[-1]), labels)

    def predict(self, embeddings):
        # NOTE: Just flatten
        return embeddings.view(-1, embeddings.shape[-1])
