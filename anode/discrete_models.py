import torch.nn as nn


class ResidualBlock(nn.Module):
    """Following the structure of the one implemented in
    https://arxiv.org/pdf/1806.10909.pdf
    """
    def __init__(self, data_dim, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, data_dim),
            nn.ReLU(True)
        )

    def forward(self, x):
        return x + self.mlp(x)


class ResNet(nn.Module):
    """ResNet which maps data_dim dimensional points to an output_dim
    dimensional output.
    """
    def __init__(self, data_dim, hidden_dim, num_layers, output_dim=1,
                 is_img=False):
        super(ResNet, self).__init__()
        residual_blocks = \
            [ResidualBlock(data_dim, hidden_dim) for _ in range(num_layers)]
        self.residual_blocks = nn.Sequential(*residual_blocks)
        self.linear_layer = nn.Linear(data_dim, output_dim)
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.is_img = is_img

    def forward(self, x, return_features=False):
        if self.is_img:
            # Flatten image, i.e. (batch_size, channels, height, width) to
            # (batch_size, channels * height * width)
            features = self.residual_blocks(x.view(x.size(0), -1))
        else:
            features = self.residual_blocks(x)
        pred = self.linear_layer(features)
        if return_features:
            return features, pred
        return pred

    @property
    def hidden_dim(self):
        return self.residual_blocks.hidden_dim


class MLPNet(nn.Module):
    """
    """
    def __init__(self, data_dim, hidden_dim):
        super(MLPNet, self).__init__()
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.mlp(x)
