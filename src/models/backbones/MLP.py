import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()

        model_args = config["model_args"]
        # Defaults to the number of selected order bins, as for the MLP-Mixer
        input_size = model_args.get(
            "input_size",
            config["gear_harmonics_to_include"]
            * (config["gear_sideband_block_size"] * 4 + 1),
        )

        # Create layers for feature extraction
        layers = []
        for i, hidden_size in enumerate(model_args["hidden_sizes"]):
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config["dropout"]))

            input_size = hidden_size

        self.features = nn.Sequential(*layers)

        # Classification/embedding head
        self.head = nn.Linear(input_size, config["embedding_len"])

    def forward(self, x):
        # Input shape: (batch_size, input_size)

        # Feature extraction
        # (batch_size, last_hidden_size)
        x = self.features(x)

        # (batch_size, embedding_len)
        x = self.head(x)

        return x
