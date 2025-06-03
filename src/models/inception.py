import torch
import torch.nn as nn
from typing import List, Optional


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
     

        self.bn = nn.BatchNorm1d(total_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.1)

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
        y = x
        if self.bottleneck is not None:
            y = self.bottleneck(x)

        branch_outputs = [branch(y) for branch in self.branches]
        branch_outputs.append(self.pool_branch(y))
 

    

        out = torch.cat(branch_outputs, dim=1)
        out = self.bn(out)
        out = self.dropout(out)

        if self.use_se:
            out = self.se(out)

        if self.shortcut is not None:
            identity = self.shortcut(identity)
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
        use_se: bool = True,
        use_residual: bool = True,
    ):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [10, 20, 40]

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
            )
            self.blocks.append(block)
            current_channels = (len(kernel_sizes) + 1) * out_channels

        # Expose embedding dimension for outside use
        self.embedding_dim = current_channels
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(current_channels, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        x = self.global_avg_pool(x).squeeze(-1)
        return self.classifier(x).squeeze(-1)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute and return the window-level embedding before classification.
        Input:
            x: Tensor of shape (batch_size, in_channels, time)
        Output:
            Tensor of shape (batch_size, embedding_dim)
        """
        for block in self.blocks:
            x = block(x)
        x = self.global_avg_pool(x).squeeze(-1)
        return x


# HierarchicalSeizureModel
class HierarchicalSeizureModel(nn.Module):
    """
    Two-stage hierarchical model:
    - First stage: window encoder based on InceptionTimeSE to produce embeddings.
    - Second stage: GRU (or LSTM) over the sequence of embeddings to capture longer-term patterns.
    Predicts one label per window in the sequence.
    """
    def __init__(
        self,
        window_encoder: InceptionTimeSE,
        hidden_size: int = 64,
        seq_model_type: str = 'gru',
        num_layers: int = 1,
        n_classes: int = 1,
    ):
        super().__init__()
        # Window encoder produces embeddings of size window_encoder.embedding_dim
        self.window_encoder = window_encoder
        embedding_dim = window_encoder.embedding_dim

        # Choose sequence model type
        if seq_model_type.lower() == 'gru':
            self.seq_model = nn.GRU(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
        elif seq_model_type.lower() == 'lstm':
            self.seq_model = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
        else:
            raise ValueError(f"Unsupported seq_model_type: {seq_model_type}")

        # Final classifier per window
        self.classifier = nn.Linear(hidden_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:
            x: Tensor of shape (batch_size, seq_len, in_channels, time)
        Output:
            preds: Tensor of shape (batch_size, seq_len) if n_classes == 1,
                   otherwise (batch_size, seq_len, n_classes)
        """
        b, seq_len, c, t = x.size()
        # Flatten batch & sequence for window encoding
        x_flat = x.view(b * seq_len, c, t)
        # Compute embeddings: shape (b * seq_len, embedding_dim)
        embeddings = self.window_encoder.get_embedding(x_flat)
        # Reshape back to sequence form: (batch_size, seq_len, embedding_dim)
        embeddings = embeddings.view(b, seq_len, -1)
        # Pass through sequence model: output shape (batch_size, seq_len, hidden_size)
        seq_out, _ = self.seq_model(embeddings)
        # Classifier on each time step
        logits = self.classifier(seq_out)  # shape (batch_size, seq_len, n_classes)
        if logits.size(-1) == 1:
            return logits.squeeze(-1)  # shape (batch_size, seq_len)
        return logits
