import torch
import torch.nn as nn
from typing import List, Optional, Tuple


def conv_bn(in_channels: int, out_channels: int, kernel_size: int) -> nn.Sequential:
    padding = (kernel_size - 1) // 2
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True),
    )


class SqueezeExcite1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.avg_pool(x)
        weights = self.fc(weights)
        return x * weights


class InceptionBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: List[int],
        bottleneck_channels: int,
        use_se: bool,
        use_residual: bool = True,
        dropout: float = 0.1,  # Made configurable
    ):
        super().__init__()
        self.use_se = use_se
        total_channels = (len(kernel_sizes) + 1) * out_channels
   
        if bottleneck_channels > 0:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        else:
            self.bottleneck = None
            bottleneck_channels = in_channels
   
        self.branches = nn.ModuleList(
            [conv_bn(bottleneck_channels, out_channels, k) for k in kernel_sizes]
        )
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            conv_bn(bottleneck_channels, out_channels, kernel_size=1),
            )
        
        branch_count  = len(self.branches) + 1               # +1 por la rama de pooling
        total_channels = branch_count * out_channels
        self.bn = nn.BatchNorm1d(total_channels)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        if use_se:
            self.se = SqueezeExcite1D(total_channels)

        if use_residual:
            if in_channels != total_channels:
                self.shortcut = nn.Conv1d(in_channels, total_channels, kernel_size=1, bias=False)
            else:
                self.shortcut = nn.Identity()
        else:
            self.shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        if self.bottleneck is not None:
            y = self.bottleneck(x)
        else:
            y = x
        branch_outputs = [branch(y) for branch in self.branches]
        pool_out = self.pool_branch(y)
        min_len = min(t.shape[-1] for t in branch_outputs + [pool_out])
        branch_outputs = [t[..., :min_len] for t in branch_outputs]
        pool_out      = pool_out[..., :min_len]
        out = torch.cat(branch_outputs + [pool_out], dim=1)
        out = self.bn(out)
        out = self.dropout(out)

        if self.use_se:
            out = self.se(out)

        if self.shortcut is not None:
            identity = self.shortcut(identity)
            if identity.size(-1) != out.size(-1):
                identity = identity[..., :out.size(-1)]
            out = out + identity

        return self.relu(out)


class InceptionTimeSE(nn.Module):
    def __init__(
        self,
        n_blocks: int = 6,
        in_channels: int = 1,
        n_classes: int = 1,
        out_channels: int = 32,
        bottleneck_channels: int = 32,
        kernel_sizes: Optional[List[int]] = None,
        use_se: bool = False,
        use_residual: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [9, 19, 39]

        self.blocks = nn.ModuleList()
        current_channels = in_channels

        for _ in range(n_blocks):
            block = InceptionBlock(
                in_channels=current_channels,
                out_channels=out_channels,
                kernel_sizes=kernel_sizes,
                bottleneck_channels=bottleneck_channels,
                use_se=use_se,
                use_residual=use_residual,
                dropout=dropout,
            )
            self.blocks.append(block)
            current_channels = (len(kernel_sizes) + 1) * out_channels

        self.embedding_dim = current_channels
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(current_channels, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        x = self.global_avg_pool(x).squeeze(-1)
        logits = self.classifier(x)
        if logits.size(-1) == 1:
            return logits.squeeze(-1)
        return logits

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        x = self.global_avg_pool(x).squeeze(-1)
        return x


class AttentionPool(nn.Module):
    """Attention-based pooling over temporal dimension."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, seq_len, hidden_dim)
        scores = self.attn(torch.tanh(x)).squeeze(-1)  # (batch, seq_len)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return context, weights


class HierarchicalSeizureModel(nn.Module):
    def __init__(
        self,
        window_encoder: InceptionTimeSE,
        hidden_size: int = 64,
        seq_model_type: str = 'gru',
        num_layers: int = 1,
        n_classes: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.window_encoder = window_encoder
        self.seq_model_type = seq_model_type.lower()
        embedding_dim = window_encoder.embedding_dim

        if self.seq_model_type == 'gru':
            self.seq_model = nn.GRU(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=True
            )
            seq_output_dim = hidden_size * 2  # bidirectional
            
        elif self.seq_model_type == 'lstm':
            self.seq_model = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=True
            )
            seq_output_dim = hidden_size * 2
            
        elif self.seq_model_type == 'transformer':
           
            from .transformer import TransformerSequenceModel
            self.seq_model = TransformerSequenceModel(
                input_dim=embedding_dim,
                num_heads=4,
                hidden_dim=hidden_size * 2,
                num_layers=num_layers,
                n_classes=hidden_size,
                dropout=dropout,
            )
            seq_output_dim = hidden_size
        else:
            raise ValueError(f"Unsupported seq_model_type: {seq_model_type}")

        self.attention = AttentionPool(seq_output_dim)
        self.classifier = nn.Linear(seq_output_dim, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, seq_len, c, t = x.size()
        x_flat = x.view(b * seq_len, c, t)
        embeddings = self.window_encoder.get_embedding(x_flat)
        embeddings = embeddings.view(b, seq_len, -1)
        embeddings = self.dropout(embeddings)
        
        if self.seq_model_type in ['gru', 'lstm']:
            seq_out, _ = self.seq_model(embeddings)
        else:  
            seq_out = self.seq_model(embeddings)
        
        seq_out = self.dropout(seq_out)
        context, attn_weights = self.attention(seq_out)
        context = self.dropout(context)
        logits = self.classifier(context)

        if logits.size(-1) == 1:
            return logits.squeeze(-1)
        return logits