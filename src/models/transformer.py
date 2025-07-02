import torch
from torch import nn

class TransformerSequenceModel(nn.Module):
 
    def __init__(self, 
                 input_dim: int, 
                 num_heads: int = 4, 
                 hidden_dim: int = 128,
                 num_layers: int = 2, 
                 n_classes: int = 1, 
                 dropout: float = 0.1):
        super().__init__()
        
        if input_dim % num_heads != 0:
            raise ValueError(f"input_dim ({input_dim}) must be divisible by num_heads ({num_heads})")
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.projection = nn.Linear(input_dim, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:    
        enc = self.encoder(x)     
        enc = self.dropout(enc)
        logits = self.projection(enc)  
        
        if logits.size(-1) == 1:
            return logits.squeeze(-1)
        return logits