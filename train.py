
#train.py
from __future__ import annotations
import argparse
import datetime, json, random, os
from tqdm import tqdm
from src.models.tcn import *
from src.models.inception import *

from src.datasets import ECGWindowDataset, collate_fn
from src.utils import  train_one_epoch, evaluate, FocalLoss
from pathlib import Path
import numpy as np
import csv
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split


def main(args):
    # Paths
    data_dir = Path(args.data_dir)

    # --- reproducibility ---
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # --- run directory (self‑contained) ---
    run_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "metrics.csv"
   
    with metrics_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss", "val_loss", "auroc", "f1",
            "precision", "recall", "accuracy",
            "tn", "fp", "fn", "tp"
        ])
    with (out_dir / "hparams.json").open("w") as f_json:
        json.dump(vars(args), f_json, indent=2)

    preproc_npz = Path(args.npz_path)
    if not preproc_npz.exists():
        raise FileNotFoundError(
            f"{preproc_npz} not found.  Run preprocessing first:\n"
            f"  python -m src.preprocessing --data_dir {data_dir} "
            f"--out_path {preproc_npz} --window_seconds {args.window_seconds} ..."
        )

    ds_full = ECGWindowDataset(preproc_npz)

    if args.test_npz_path:
        ds_test = ECGWindowDataset(Path(args.test_npz_path))
    else:
        # 10 % of train become test after val split
        test_frac = 0.1
        n_test = int(len(ds_full) * test_frac)
        ds_full, ds_test = random_split(ds_full,
                                        [len(ds_full) - n_test, n_test],
                                        generator=torch.Generator().manual_seed(123))

    n_total = len(ds_full)
    n_train = int(n_total * 0.8)
    n_val = n_total - n_train
    ds_train, ds_val = random_split(ds_full, [n_train, n_val], generator=torch.Generator().manual_seed(42))

  
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    train_loader = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=args.batch_size * 2,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        ds_test,
        batch_size=args.batch_size * 2,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )


    from torch.utils.data import Subset
    if isinstance(ds_train, Subset):
        y_train = ds_train.dataset.y[ds_train.indices]
    else:
        y_train = ds_train.y
    label_counts = np.bincount(y_train, minlength=2)
    ratio = label_counts[0] / label_counts[1] if label_counts[1] > 0 else 1.0
    pos_weight = torch.tensor(ratio, dtype=torch.float32, device=device)
    print(f"[INFO] Positive samples in train: {label_counts[1]} , pos_weight={pos_weight.item():.2f}")
    model_tag = args.model
    # Step 3 – Model, optimizer, loss
    if model_tag == "inception":
        model = InceptionTimeSE(n_blocks=args.n_blocks,in_channels=1,n_classes=1,out_channels=args.out_channels,bottleneck_channels=args.bottleneck_channels,kernel_sizes=args.kernel_sizes,use_se=(not args.no_se),use_residual=True ).to(device)
       
    elif model_tag == "tcn":
        model = TCNClassifier(in_channels=1,channels=(args.out_channels, args.out_channels*2,args.out_channels*2, args.out_channels*4),).to(device)

    if args.focal_gamma and args.focal_gamma > 0:
        criterion = FocalLoss(alpha=pos_weight.item(), gamma=args.focal_gamma)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_metric = 0.0
    patience_counter = 0

  
    best_path = out_dir / f"best_model_{model_tag}.pt"
    interrupt_path = out_dir / f"interrupt_model_{model_tag}.pt"

    try:
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_metrics = evaluate(model, val_loader, criterion, device)
            print(
                f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} "
                f"AUROC={val_metrics['auroc']:.3f} F1={val_metrics['f1']:.3f}"
                f" Precision={val_metrics['precision']:.3f} Recall={val_metrics['recall']:.3f} Accuracy={val_metrics['accuracy']:.3f}"
                f" TN={val_metrics['tn']} FP={val_metrics['fp']} FN={val_metrics['fn']} TP={val_metrics['tp']}"
            )
            # append metrics to CSV
            with metrics_path.open("a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch,
                    f"{train_loss:.6f}",
                    f"{val_metrics['loss']:.6f}",
                    f"{val_metrics['auroc']:.6f}",
                    f"{val_metrics['f1']:.6f}",
                    f"{val_metrics['precision']:.6f}",
                    f"{val_metrics['recall']:.6f}",
                    f"{val_metrics['accuracy']:.6f}",
                    val_metrics['tn'],
                    val_metrics['fp'],
                    val_metrics['fn'],
                    val_metrics['tp'],
                ])

            scheduler.step(val_metrics["loss"])
        
            if val_metrics["auroc"] > best_metric + 1e-4:
                best_metric = val_metrics["auroc"]
                torch.save(model.state_dict(), best_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print("Early stopping triggered.")
                    break
    except KeyboardInterrupt:
        print("\n[KeyboardInterrupt] Training interrupted by user. Saving current weights...")
        torch.save(model.state_dict(), interrupt_path)
        print(f"Partial model saved to {interrupt_path}")
    print(f"Best AUROC = {best_metric:.3f} saved to {best_path}")

    test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"--- TEST SET ---  AUROC={test_metrics['auroc']:.3f}  "
          f"F1={test_metrics['f1']:.3f}  Precision={test_metrics['precision']:.3f} "
          f"Recall={test_metrics['recall']:.3f}  Accuracy={test_metrics['accuracy']:.3f}")
    # log to CSV
    with metrics_path.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["TEST", "-", "-", test_metrics['auroc'],
                         test_metrics['f1'], test_metrics['precision'],
                         test_metrics['recall'], test_metrics['accuracy'],
                         test_metrics['tn'], test_metrics['fp'],
                         test_metrics['fn'], test_metrics['tp']])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InceptionTime‑SE ECG seizure detection trainer")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with raw CSV files")
    parser.add_argument("--npz_path", required=True, help="Path to pre‑chunked NPZ")
    parser.add_argument("--test_npz_path", type=str, default=None,
                        help="Optional NPZ for held‑out test set.")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for artifacts")
    # parser.add_argument("--skip_preproc", action="store_true", help="Skip preprocessing if windows.npz exists")
    parser.add_argument("--window_seconds", type=float, default=15, help="Window length [s]")
    parser.add_argument("--overlap", type=float, default=0, help="Window overlap fraction [0‑1)")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--model", choices=["inception", "tcn"], default="inception",
                        help="Which backbone to train.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8, help="Early‑stopping patience")
    parser.add_argument("--seizure_threshold", type=int,default=2)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--n_blocks", type=int, default=6)
    parser.add_argument("--num_workers", type=int,default=0)
    parser.add_argument("--out_channels", type=int, default=32)
    parser.add_argument("--bottleneck_channels", type=int, default=32)
    parser.add_argument("--kernel_sizes", type=int, nargs="*", default=[10, 20, 40])
    parser.add_argument("--no_se", action="store_true", help="Disable Squeeze‑and‑Excite blocks")
    parser.add_argument("--neg_to_pos", type=float, default=4,
                    help="Keep at most this many negatives per positive "
                         "(e.g. 3 → ≤ 3× negatives).")
    parser.add_argument("--focal_gamma", type=float, default=None,
                        help="If set (>0), use Focal Loss with this γ instead of BCE.")
    args = parser.parse_args()

    main(args)
