import math
import torch
from torch import nn

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, dim: int, dropout: float = 0.1, max_len: int = 10000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2) * (-math.log(10000.0) / dim)
        )
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.pe.size(1):
            # Extend positional encoding if input is longer than expected
            extra_len = x.size(1) - self.pe.size(1)
            position = torch.arange(
                self.pe.size(1), self.pe.size(1) + extra_len, device=x.device
            ).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, self.pe.size(2), 2, device=x.device)
                * (-math.log(10000.0) / self.pe.size(2))
            )
            extra_pe = torch.zeros(extra_len, self.pe.size(2), device=x.device)
            extra_pe[:, 0::2] = torch.sin(position * div_term)
            extra_pe[:, 1::2] = torch.cos(position * div_term)
            self.pe = torch.cat([self.pe, extra_pe.unsqueeze(0)], dim=1)

        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class AttentionDenoiser(nn.Module):
    """Multi-head attention based denoising layer."""
    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        return x + attn_out


class HybridTransformerCNN(nn.Module):
    """CNN feature extractor followed by Transformer encoder for seizure detection."""
    def __init__(self,
                 in_channels: int = 1,
                 conv_channels: tuple = (32, 64),
                 num_heads: int = 4,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 n_classes: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        layers = []
        prev = in_channels
        for ch in conv_channels:
            layers += [
                nn.Conv1d(prev, ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(ch),
                nn.ReLU()
            ]
            prev = ch
        self.cnn = nn.Sequential(*layers)
        self.embedding_dim = prev
        self.pos_enc = PositionalEncoding(prev, dropout)
        self.denoiser = AttentionDenoiser(prev, num_heads)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=prev,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(prev, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, channels, time)
        features = self.cnn(x)
        features = features.permute(0, 2, 1)  # (batch, seq_len, embed)
        features = self.pos_enc(features)
        features = self.denoiser(features)
        features = self.transformer(features)
        features = features.permute(0, 2, 1)
        pooled = self.pool(features).squeeze(-1)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        if logits.size(-1) == 1:
            return logits.squeeze(-1)
        return logits
