import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(
            self,
            in_features: int = 32 * 64 * 2,
            hidden_features: int = 32 * 64 * 4,
            num_hidden_blocks: int = 8,
            out_features: int = 32 * 64,
    ):
        super().__init__()
        linear = nn.Linear(in_features=in_features, out_features=hidden_features)
        self.layers = [linear, nn.ReLU()]

        for i in range(num_hidden_blocks):
            linear = nn.Linear(in_features=hidden_features, out_features=hidden_features)
            self.layers += [linear, nn.ReLU()]

        linear = nn.Linear(in_features=hidden_features, out_features=out_features)
        self.layers += [linear]
        self.net = nn.Sequential(*self.layers)

    def forward(self, x):
        batch, channel, height, weight = x.shape

        x = self.net(torch.flatten(x, start_dim=1))
        x = x.view(batch, 1, height, weight)

        return x