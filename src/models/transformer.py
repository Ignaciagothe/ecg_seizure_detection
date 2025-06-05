import torch
from torch import nn

class TransformerSequenceModel(nn.Module):
    """Lightweight Transformer encoder for sequences of embeddings."""
    def __init__(self, input_dim: int, num_heads: int = 4, hidden_dim: int = 128,
                 num_layers: int = 2, n_classes: int = 1, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(input_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, seq_len, input_dim]
        enc = self.encoder(x)  # [B, seq_len, input_dim]
        logits = self.classifier(enc)
        if logits.size(-1) == 1:
            return logits.squeeze(-1)
        return logits
