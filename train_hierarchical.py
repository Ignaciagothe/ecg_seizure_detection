import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.models.inception import InceptionTimeSE, HierarchicalSeizureModel
from src.models.transformer import TransformerSequenceModel
from src.utils import train_one_epoch, evaluate, FocalLoss

class WindowSequenceDataset(Dataset):
    """Create sequences of windows directly from NPZ."""
    def __init__(self, npz_path: Path, seq_len: int):
        arr = np.load(npz_path)
        x = arr["x"]
        y = arr["y"]
        usable = (len(y) // seq_len) * seq_len
        self.x = x[:usable].reshape(-1, seq_len, x.shape[1])
        self.y = y[:usable].reshape(-1, seq_len)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        xb = torch.from_numpy(self.x[idx]).float().unsqueeze(1)
        yb = torch.from_numpy(self.y[idx]).float()
        return xb, yb

def build_model(args):
    window_encoder = InceptionTimeSE(
        n_blocks=args.n_blocks,
        in_channels=1,
        n_classes=1,
        out_channels=args.out_channels,
        bottleneck_channels=args.bottleneck_channels,
        kernel_sizes=args.kernel_sizes,
        use_se=not args.no_se,
    )
    if args.seq_model == "transformer":
        seq_model = TransformerSequenceModel(
            input_dim=window_encoder.embedding_dim,
            num_heads=args.num_heads,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
        )
    else:
        seq_model = "gru"
    model = HierarchicalSeizureModel(
        window_encoder=window_encoder,
        hidden_size=args.hidden_size,
        seq_model_type=args.seq_model,
        num_layers=args.num_layers,
    )
    if args.seq_model == "transformer":
        model.seq_model = TransformerSequenceModel(
            input_dim=window_encoder.embedding_dim,
            num_heads=args.num_heads,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
        )
    return model

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = WindowSequenceDataset(args.train_npz, args.seq_len)
    val_ds = WindowSequenceDataset(args.val_npz, args.seq_len)
    test_ds = WindowSequenceDataset(args.test_npz, args.seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    model = build_model(args).to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        print(f"[Epoch {epoch}] train_loss={train_loss:.4f} val_auroc={val_metrics['auroc']:.3f}")
        scheduler.step(val_metrics['loss'])
        if val_metrics['auroc'] > best:
            best = val_metrics['auroc']
            torch.save(model.state_dict(), args.out_dir / "best_hier.pt")
    model.load_state_dict(torch.load(args.out_dir / "best_hier.pt", map_location=device))
    test_metrics = evaluate(model, test_loader, criterion, device)
    print(test_metrics)

if __name__ == "__main__":
    p = argparse.ArgumentParser("Train hierarchical seizure model")
    p.add_argument("--train_npz", type=Path, required=True)
    p.add_argument("--val_npz", type=Path, required=True)
    p.add_argument("--test_npz", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--seq_len", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden_size", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--seq_model", choices=["gru", "lstm", "transformer"], default="gru")
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--n_blocks", type=int, default=6)
    p.add_argument("--out_channels", type=int, default=32)
    p.add_argument("--bottleneck_channels", type=int, default=32)
    p.add_argument("--kernel_sizes", type=int, nargs="*", default=[10, 20, 40])
    p.add_argument("--no_se", action="store_true")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    main(args)
