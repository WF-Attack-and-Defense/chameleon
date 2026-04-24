"""
Surakav (WF-GAN) generator — adapted from pendding/surakav/src/model.py
(without optional torchsummaryX dependency).
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """Conditional WGAN generator for burst-sequence synthesis."""

    def __init__(
        self,
        seq_size: int,
        class_dim: int,
        latent_dim: int,
        scaler_min: float,
        scaler_max: float,
        is_gpu: bool = False,
    ):
        super().__init__()
        self.seq_size = seq_size
        self.class_dim = class_dim
        self.latent_dim = latent_dim
        self.LongTensor = torch.cuda.LongTensor if is_gpu else torch.LongTensor
        self.scaler_min = scaler_min
        self.scaler_max = scaler_max

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.ReLU(inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.latent_dim + self.class_dim, 512, normalize=False),
            *block(512, 1024),
            *block(1024, 2048),
            nn.Linear(2048, self.seq_size),
            nn.Sigmoid(),
        )

    def forward(self, z, c):
        inp = torch.cat([z, c], 1)
        trace = self.model(inp)
        burst_length = trace[:, 0] * (self.scaler_max - self.scaler_min) + self.scaler_min
        mask = torch.zeros_like(trace)
        mask[(torch.arange(trace.shape[0]), burst_length.type(self.LongTensor) + 1)] = 1
        mask = 1 - mask.cumsum(dim=-1)
        trace = trace * mask
        return trace
