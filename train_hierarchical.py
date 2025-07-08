import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from contextlib import nullcontext
import datetime
import csv
import json
from src.models.inception import InceptionTimeSE, HierarchicalSeizureModel
from src.utils import FocalLoss, compute_metrics,evaluate

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class WindowSequenceDataset(Dataset):
    def __init__(self, npz_path: Path, seq_len: int, stride: int = None):
        arr = np.load(npz_path)
        self.x = arr["x"]
        self.y = arr["y"]
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len
        self.sequences = []
        for i in range(0, len(self.y) - seq_len + 1, self.stride):
            self.sequences.append(list(range(i, i + seq_len)))
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq_indices = self.sequences[idx]
        x_seq = self.x[seq_indices]
        y_seq = self.y[seq_indices]
        seq_label = float(y_seq.max() > 0)
        x_seq = x_seq[:, np.newaxis, :]
        return torch.FloatTensor(x_seq), torch.FloatTensor([seq_label])

def build_model(args, device):
    window_encoder = InceptionTimeSE(
        n_blocks=args.n_blocks,
        in_channels=1,
        n_classes=1,
        out_channels=args.out_channels,
        bottleneck_channels=args.bottleneck_channels,
        kernel_sizes=args.kernel_sizes,
        use_se= args.use_se,  
    )
    
    if args.pretrained_inception:
        print(f"Loading pretrained InceptionTime from {args.pretrained_inception}")
        checkpoint = torch.load(args.pretrained_inception, map_location=device)
        window_encoder.load_state_dict(checkpoint)
        if args.freeze_encoder:
            print("Freezing window encoder weights")
            for param in window_encoder.parameters():
                param.requires_grad = False
    
    model = HierarchicalSeizureModel(
        window_encoder=window_encoder,
        hidden_size=args.hidden_size,
        seq_model_type=args.seq_model,
        num_layers=args.num_layers,
    )
    
    return model

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(loader, desc="Training")
    for x_seq, y_seq in pbar:
        x_seq = x_seq.to(device)
        y_seq = y_seq.to(device)
        optimizer.zero_grad()
        context = (torch.autocast(device_type=device.type, dtype=torch.float16)
                   if device.type != "cpu" else nullcontext())
        with context:
            logits = model(x_seq)
            loss = criterion(logits.view(-1), y_seq.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item() * x_seq.size(0)
        pbar.set_postfix({'loss': loss.item()})
    
    return running_loss / len(loader.dataset)

def evaluate2(model, loader, criterion, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    
    with torch.no_grad():
        for x_seq, y_seq in tqdm(loader, desc="Evaluating"):
            x_seq = x_seq.to(device)
            y_seq = y_seq.to(device)
            context = (torch.autocast(device_type=device.type, dtype=torch.float16)
                       if device.type != "cpu" else nullcontext())
            with context:
                logits = model(x_seq)
                loss = criterion(logits.view(-1), y_seq.view(-1))
            total_loss += loss.item() * x_seq.size(0)
            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(y_seq.cpu().numpy())
    
    all_preds = np.concatenate(all_preds).reshape(-1)
    all_labels = np.concatenate(all_labels).reshape(-1)

    metrics = compute_metrics(all_labels, all_preds)
    metrics['loss'] = total_loss / len(loader.dataset)
    return metrics

def main(args):
    device = (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps") if torch.backends.mps.is_available() else
        torch.device("cpu")
    )
    set_seed(args.seed)
    print(f"Using device: {device}")
    run_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir / f"hierarchical_{run_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # with open(out_dir / "hparams.json", "w") as f:
    #     json.dump(vars(args), f, indent=2)
    
    train_ds = WindowSequenceDataset(args.train_npz, args.seq_len, stride=args.seq_stride)
    val_ds = WindowSequenceDataset(args.val_npz, args.seq_len, stride=args.seq_len)  # No overlap for val
    test_ds = WindowSequenceDataset(args.test_npz, args.seq_len, stride=args.seq_len)  # No overlap for test
    
    print(f"Train sequences: {len(train_ds)}")
    print(f"Val sequences: {len(val_ds)}")
    print(f"Test sequences: {len(test_ds)}")
    
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type != "cpu",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type != "cpu",
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type != "cpu",
    )
    
    model = build_model(args, device).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    if args.focal_gamma > 0:
        pos_samples = sum(train_ds.y[idx_seq].max() > 0 for idx_seq in train_ds.sequences)
        total_samples = len(train_ds.sequences)
        pos_weight = (total_samples - pos_samples) / (pos_samples + 1e-8)
        criterion = FocalLoss(alpha=pos_weight, gamma=args.focal_gamma)
        print(f"Using Focal Loss with alpha={pos_weight:.2f}, gamma={args.focal_gamma}")
    else:
        criterion = nn.BCEWithLogitsLoss()

    params_to_optimize = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(params_to_optimize, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    metrics_path = out_dir / "metrics.csv"
    with open(metrics_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch', 'train_loss', 'val_loss', 'auroc', 'auprc', 'f1',
            'precision', 'recall', 'accuracy', 'tn', 'fp', 'fn', 'tp'
        ])
    
    best_auroc = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f} | AUROC: {val_metrics['auroc']:.3f} | "
              f"AUPRC: {val_metrics['auprc']:.3f} | F1: {val_metrics['f1']:.3f}")
        with open(metrics_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, f"{train_loss:.6f}", f"{val_metrics['loss']:.6f}",
                f"{val_metrics['auroc']:.6f}", f"{val_metrics['auprc']:.6f}",
                f"{val_metrics['f1']:.6f}", f"{val_metrics['precision']:.6f}",
                f"{val_metrics['recall']:.6f}", f"{val_metrics['accuracy']:.6f}",
                val_metrics['tn'], val_metrics['fp'], 
                val_metrics['fn'], val_metrics['tp']
            ])

        scheduler.step(val_metrics['loss'])

        if val_metrics['auroc'] > best_auroc:
            best_auroc = val_metrics['auroc']
            torch.save(model.state_dict(), out_dir / 'best_model.pt')
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= args.patience:
            print("Early stopping por paciencia")
            break
    
    print("\n" + "="*50)
    print("Evaluating on test set...")
    model.load_state_dict(torch.load(out_dir / 'best_model.pt', map_location=device))
    test_metrics = evaluate(model, test_loader, criterion, device)
    
    print(f"\nTest Results:")
    print(f"AUROC: {test_metrics['auroc']:.3f} | AUPRC: {test_metrics['auprc']:.3f}")
    print(f"F1: {test_metrics['f1']:.3f} | Precision: {test_metrics['precision']:.3f} | "
          f"Recall: {test_metrics['recall']:.3f}")
    print(f"Accuracy: {test_metrics['accuracy']:.3f}")
    print(f"TP: {test_metrics['tp']} | TN: {test_metrics['tn']} | "
          f"FP: {test_metrics['fp']} | FN: {test_metrics['fn']}")
    
    with open(metrics_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'TEST', '-', '-',
            f"{test_metrics['auroc']:.6f}", f"{test_metrics['auprc']:.6f}",
            f"{test_metrics['f1']:.6f}", f"{test_metrics['precision']:.6f}",
            f"{test_metrics['recall']:.6f}", f"{test_metrics['accuracy']:.6f}",
            test_metrics['tn'], test_metrics['fp'], 
            test_metrics['fn'], test_metrics['tp']
        ])
    
    print(f"\nResultados guardados en {out_dir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser("Para entrenamiento del modelo de dos etapas para deteccion de  seizure ")
    print('Solo nececitas entregar al ejecutar los 4 paths siguentes, los demas puedes personalizar opcionalmentee

    p.add_argument("--train_npz", type=Path, required=True)
    p.add_argument("--val_npz", type=Path, required=True)
    p.add_argument("--test_npz", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)

    p.add_argument("--n_blocks", type=int, default=6)
    p.add_argument("--out_channels", type=int, default=32)
    p.add_argument("--bottleneck_channels", type=int, default=32)
    p.add_argument("--kernel_sizes", type=int, nargs="+", default=[9, 19, 39])
    p.add_argument("--use_se", action="store_true", help="Enable SE blocks")
    p.add_argument("--hidden_size", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--seq_model", choices=["gru", "lstm", "transformer"], default="gru")
    
    p.add_argument("--seq_len", type=int, default=6, help="Numero de ventanas por secuencia")
    p.add_argument("--seq_stride", type=int, default=1, help="Paso avance entre secuencias")
    
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--focal_gamma", type=float, default=2.0, help="Focal loss gamma (0 to disable)")
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--num_workers", type=int, default=0) #subir a 4 por ejemplo si tienes cuda - (no macos)
    p.add_argument("--seed", type=int, default=42)
    
    p.add_argument("--pretrained_inception", type=str, default=None,
                   help="Path to pretrained InceptionTimeSE weights")
    p.add_argument("--freeze_encoder", action="store_true",
                   help="Freeze the window encoder weights")
    
    args = p.parse_args()
    main(args)
