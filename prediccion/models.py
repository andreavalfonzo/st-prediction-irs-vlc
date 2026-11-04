"""CNN-GRU con cabezal angular residual y dos ablaciones comparables."""
import torch
from torch import nn


class Controller(nn.Module):
    def __init__(self, n_angles, kind='cnn_gru'):
        super().__init__()
        if kind not in ('cnn_gru', 'coord_gru', 'cnn'):
            raise ValueError(kind)
        self.kind = kind
        self.encoder = (nn.Sequential(nn.Linear(8, 64), nn.ReLU()) if kind == 'coord_gru'
                        else nn.Sequential(nn.Conv2d(8, 12, 3, padding=1), nn.ReLU(),
                                           nn.AvgPool2d(2), nn.Conv2d(12, 16, 3, padding=1),
                                           nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
                                           nn.Linear(256, 64), nn.ReLU()))
        if kind != 'cnn':
            self.gru = nn.GRU(64, 64, batch_first=True)
        self.head = nn.Linear(64, n_angles)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        # Corrections are bounded to +/- 0.15 rad (~8.6 degrees) per angle.
        # The known current geometric solution is a skip connection, not a
        # future observation. All three networks use the same connection.

    def forward(self, maps, coords, base):
        if self.kind == 'coord_gru':
            encoded = self.encoder(coords)
        elif self.kind == 'cnn':
            return base + 0.15 * torch.tanh(self.head(self.encoder(maps[:, -1])))
        else:
            b, length, c, h, w = maps.shape
            encoded = self.encoder(maps.reshape(b*length, c, h, w)).reshape(b, length, -1)
        encoded, _ = self.gru(encoded)
        return base + 0.15 * torch.tanh(self.head(encoded[:, -1]))
