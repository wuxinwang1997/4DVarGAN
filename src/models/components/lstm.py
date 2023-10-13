import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(
            self,
            in_features: int = 32 * 64 * 2,
            hidden_features: int = 32 * 64,
            num_layers: int = 8,
            out_features: int = 32 * 64,
    ):
        super().__init__()
        self.linear1 = nn.Linear(in_features=in_features, out_features=hidden_features)
        self.lstm = nn.LSTM(input_size=hidden_features, num_layers=num_layers, hidden_size=hidden_features, batch_first=True)
        self.linear2 = nn.Linear(in_features=hidden_features, out_features=out_features)

    def forward(self, x):
        batch, channel, height, weight = x.shape
        x = self.linear1(torch.flatten(x, start_dim=1))
        x, (h, c) = self.lstm(x)
        x = self.linear2(x)
        x = x.view(batch, 1, height, weight)

        return x