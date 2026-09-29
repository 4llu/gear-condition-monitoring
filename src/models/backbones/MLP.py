import torch.nn as nn
from einops import rearrange


class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()

        model_args = config["model_args"]
        input_size = model_args["input_size"]

        # Create layers for feature extraction
        layers = []
        for i, hidden_size in enumerate(model_args["hidden_sizes"]):
            layers.append(nn.Linear(input_size, hidden_size))
            # Add batch normalization after all but the last hidden layer
            # if i < len(model_args["hidden_sizes"]) - 1:
            #     layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config["dropout"]))

            input_size = hidden_size

        self.features = nn.Sequential(*layers)

        # Classification/embedding head
        self.head = nn.Linear(input_size, config["embedding_len"])

    def forward(self, x):
        # Input shape: (batch_size, input_size)

        # Get rid of the channel dimension (should always be 1 for this model)
        # (batch_size, input_size)
        # x = rearrange(x, "b 1 i -> b i")

        # Feature extraction
        # (batch_size, last_hidden_size)
        x = self.features(x)

        # (batch_size, embedding_len)
        x = self.head(x)

        return x
