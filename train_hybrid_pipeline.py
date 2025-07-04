import argparse
from pathlib import Path
import csv
import json
import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.preprocessing import preprocess_dataset
from src.datasets import ECGWindowDataset, collate_fn
from src.models.hybrid_transformer_cnn import HybridTransformerCNN
from src.utils import FocalLoss, train_one_epoch, evaluate, make_split


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def preprocess_splits(args, processed_dir: Path, splits_dir: Path) -> tuple[Path, Path, Path]:
    processed_dir.mkdir(parents=True, exist_ok=True)
    train_list = splits_dir / "train.txt"
    val_list = splits_dir / "val.txt"
    test_list = splits_dir / "test.txt"

    train_npz = processed_dir / "windows_train.npz"
    val_npz = processed_dir / "windows_val.npz"
    test_npz = processed_dir / "windows_test.npz"

    for list_path, out_path in [
        (train_list, train_npz),
        (val_list, val_npz),
        (test_list, test_npz),
    ]:
        preprocess_dataset(
            data_dir=args.data_dir,
            out_path=out_path,
            sample_period=args.sample_period,
            window_seconds=args.window_seconds,
            overlap=args.overlap,
            seizure_threshold=args.seizure_threshold,
            neg_to_pos=args.neg_to_pos,
            post_margin_seconds=args.post_margin_seconds,
            file_list=str(list_path),
            low_cut=args.low_cut,
            high_cut=args.high_cut,
        )

    return train_npz, val_npz, test_npz


def build_model(args) -> nn.Module:
    return HybridTransformerCNN(
        in_channels=1,
        conv_channels=tuple(args.conv_channels),
        num_heads=args.num_heads,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        n_classes=1,
    )


def main(args):
    device = (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps") if torch.backends.mps.is_available() else
        torch.device("cpu")
    )
    set_seed(args.seed)
    print(f"Using device: {device}")

    if args.splits_dir is None:
        make_split(
            data_dir=args.data_dir,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            group=args.split_group,
            seed=args.seed,
        )
        splits_dir = Path(args.data_dir).parent / "splits"
    else:
        splits_dir = Path(args.splits_dir)

    run_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / f"hybrid_{run_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    processed_dir = out_dir / "processed"
    train_npz, val_npz, test_npz = preprocess_splits(args, processed_dir, splits_dir)

    with open(out_dir / "hparams.json", "w") as fh:
        json.dump(vars(args), fh, indent=2)

    train_ds = ECGWindowDataset(train_npz)
    val_ds = ECGWindowDataset(val_npz)
    test_ds = ECGWindowDataset(test_npz)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type != "cpu",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type != "cpu",
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type != "cpu",
    )

    model = build_model(args).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    pos_samples = int(train_ds.y.sum())
    neg_samples = len(train_ds.y) - pos_samples
    if args.focal_gamma > 0:
        pos_weight = neg_samples / (pos_samples + 1e-8)
        criterion = FocalLoss(alpha=pos_weight, gamma=args.focal_gamma)
        print(f"Using Focal Loss with alpha={pos_weight:.2f}, gamma={args.focal_gamma}")
    else:
        pos_weight = neg_samples / (pos_samples + 1e-8)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    metrics_path = out_dir / "metrics.csv"
    with open(metrics_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "epoch", "train_loss", "val_loss", "auroc", "auprc", "f1",
            "precision", "recall", "accuracy", "tn", "fp", "fn", "tp",
        ])

    best_auroc = 0.0
    patience_counter = 0
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        print(f"Train Loss: {train_loss:.4f}")
        print(
            f"Val Loss: {val_metrics['loss']:.4f} | AUROC: {val_metrics['auroc']:.3f} | "
            f"AUPRC: {val_metrics['auprc']:.3f} | F1: {val_metrics['f1']:.3f}"
        )
        with open(metrics_path, "a", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                epoch, f"{train_loss:.6f}", f"{val_metrics['loss']:.6f}",
                f"{val_metrics['auroc']:.6f}", f"{val_metrics['auprc']:.6f}",
                f"{val_metrics['f1']:.6f}", f"{val_metrics['precision']:.6f}",
                f"{val_metrics['recall']:.6f}", f"{val_metrics['accuracy']:.6f}",
                val_metrics['tn'], val_metrics['fp'],
                val_metrics['fn'], val_metrics['tp'],
            ])
        scheduler.step(val_metrics['loss'])

        if val_metrics['auroc'] > best_auroc:
            best_auroc = val_metrics['auroc']
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= args.patience:
            print("Early stopping triggered")
            break

    print("\n" + "=" * 50)
    print("Evaluating on test set...")
    model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=device))
    test_metrics = evaluate(model, test_loader, criterion, device)

    print("\nTest Results:")
    print(
        f"AUROC: {test_metrics['auroc']:.3f} | AUPRC: {test_metrics['auprc']:.3f}"
    )
    print(
        f"F1: {test_metrics['f1']:.3f} | Precision: {test_metrics['precision']:.3f} | "
        f"Recall: {test_metrics['recall']:.3f}"
    )
    print(f"Accuracy: {test_metrics['accuracy']:.3f}")
    print(
        f"TP: {test_metrics['tp']} | TN: {test_metrics['tn']} | "
        f"FP: {test_metrics['fp']} | FN: {test_metrics['fn']}"
    )

    with open(metrics_path, "a", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "TEST", "-", "-",
            f"{test_metrics['auroc']:.6f}", f"{test_metrics['auprc']:.6f}",
            f"{test_metrics['f1']:.6f}", f"{test_metrics['precision']:.6f}",
            f"{test_metrics['recall']:.6f}", f"{test_metrics['accuracy']:.6f}",
            test_metrics['tn'], test_metrics['fp'],
            test_metrics['fn'], test_metrics['tp'],
        ])

    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser("Train Hybrid Transformer-CNN on raw ECG data")
    p.add_argument("--data_dir", type=str, required=True, help="Directory with raw ECG CSV files")
    p.add_argument("--splits_dir", type=str, default=None, help="Directory containing train/val/test.txt")
    p.add_argument("--out_dir", type=str, required=True, help="Output directory for run")

    p.add_argument("--sample_period", type=float, default=1/256)
    p.add_argument("--window_seconds", type=float, default=5.0)
    p.add_argument("--overlap", type=float, default=0.0)
    p.add_argument("--seizure_threshold", type=int, default=1)
    p.add_argument("--neg_to_pos", type=float, default=3.0)
    p.add_argument("--post_margin_seconds", type=float, default=0.0)
    p.add_argument("--low_cut", type=float, default=0.5)
    p.add_argument("--high_cut", type=float, default=40.0)

    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument("--split_group", choices=["patient", "session", "file"], default="patient")

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--focal_gamma", type=float, default=0.0)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--conv_channels", type=int, nargs="+", default=[32, 64])
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)

    args = p.parse_args()
    main(args)
