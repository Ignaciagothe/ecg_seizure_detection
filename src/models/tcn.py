
from typing import List
import torch
from torch import nn
"""
models/tcn.py

Backbones for 1‑D time‑series classification:
* TCNClassifier     (causal, dilated Temporal Convolutional Network)

Both expose the same forward interface that returns logits (B,).
"""
# ----------------------------------------------------------------------
# Temporal Convolutional Network (causal, dilated)
# ----------------------------------------------------------------------
class Chomp1d(nn.Module):
    def __init__(self, chomp: int):
        super().__init__()
        self.chomp = chomp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., :-self.chomp] if self.chomp > 0 else x


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      padding=pad, dilation=dilation),
            Chomp1d(pad),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size,
                      padding=pad, dilation=dilation),
            Chomp1d(pad),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.down = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        out = out + self.down(x)
        return self.relu(out)


class TCNClassifier(nn.Module):
    """
    Dilated causal‐TCN followed by global average pooling.
    """
    def __init__(
        self,
        in_channels: int = 1,
        n_classes: int = 1,
        channels: tuple[int, ...] = (16, 32, 32, 64),
        kernel_size: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        prev = in_channels
        for i, ch in enumerate(channels):
            layers.append(
                TemporalBlock(
                    prev, ch, kernel_size,
                    dilation=2 ** i,
                    dropout=dropout
                )
            )
            prev = ch
        self.tcn = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(prev, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tcn(x)
        x = self.gap(x).squeeze(-1)
        return self.head(x).squeeze(-1)