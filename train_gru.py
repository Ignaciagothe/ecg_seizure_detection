# train_gru.py
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from src.models.transformer import TransformerSequenceModel
from pathlib import Path
import csv
from tqdm import tqdm

class EmbeddingSequenceDataset(Dataset):
    """
    Given embeddings.npy (N, D) and labels.npy (N,), chop into
    non-overlapping sequences of length seq_len.
    """

    def __init__(self, embeddings_path: Path, labels_path: Path, seq_len: int):
        self.embeddings = np.load(embeddings_path)  # shape (N_windows, D)
        self.labels = np.load(labels_path)          # shape (N_windows,)
        if self.embeddings.shape[0] != self.labels.shape[0]:
            raise ValueError(
                "Embeddings and labels must have same first dimension."
            )
        self.seq_len = seq_len
        total_windows = self.embeddings.shape[0]
        # Drop remainder so that total_windows % seq_len == 0
        usable = (total_windows // seq_len) * seq_len
        self.embeddings = self.embeddings[:usable]
        self.labels = self.labels[:usable]

        # Now reshape: (num_seqs, seq_len, D) and (num_seqs, seq_len)
        self.embeddings = self.embeddings.reshape(
            -1, seq_len, self.embeddings.shape[1]
        )
        self.labels = self.labels.reshape(-1, seq_len)

    def __len__(self):
        return self.embeddings.shape[0]  # num_seqs

    def __getitem__(self, idx):
        # return Tensors
        emb_seq = torch.from_numpy(self.embeddings[idx]).float()  # [seq_len, D]
        lbl_seq = torch.from_numpy(self.labels[idx]).float()      # [seq_len]
        return emb_seq, lbl_seq


class GRUSequenceModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        n_classes: int = 1,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, seq_len, input_dim]
        seq_out, _ = self.gru(x)  # [B, seq_len, hidden_size]
        logits = self.classifier(seq_out)  # [B, seq_len, n_classes]
        if logits.size(-1) == 1:
            return logits.squeeze(-1)  # [B, seq_len]
        else:
            return logits  # [B, seq_len, n_classes]


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for emb_seq, lbl_seq in tqdm(loader, desc="  [Train batches]"):
        emb_seq = emb_seq.to(device)       # [B, seq_len, D]
        lbl_seq = lbl_seq.to(device)       # [B, seq_len]
        optimizer.zero_grad()
        logits = model(emb_seq)            # [B, seq_len]
        # Flatten both: (B * seq_len,)
        loss = criterion(logits.view(-1), lbl_seq.view(-1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * emb_seq.size(0)
    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_labels = []
    for emb_seq, lbl_seq in tqdm(loader, desc="  [Val/Test batches]"):
        emb_seq = emb_seq.to(device)
        lbl_seq = lbl_seq.to(device)
        logits = model(emb_seq)  # [B, seq_len]
        loss = criterion(logits.view(-1), lbl_seq.view(-1))
        total_loss += loss.item() * emb_seq.size(0)
        all_logits.append(logits.cpu().numpy())
        all_labels.append(lbl_seq.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    all_logits = np.concatenate(all_logits, axis=0).reshape(-1)       # flatten to (total_windows,)
    all_labels = np.concatenate(all_labels, axis=0).reshape(-1)       # (total_windows,)
    # Compute simple metrics: accuracy, precision, recall, F1, AUROC
    # (You can reuse your existing evaluate function from src.utils, but for brevity, let's compute accuracy here.)

    preds = (all_logits > 0).astype(int)
    labels = all_labels.astype(int)

    TP = int(((preds == 1) & (labels == 1)).sum())
    TN = int(((preds == 0) & (labels == 0)).sum())
    FP = int(((preds == 1) & (labels == 0)).sum())
    FN = int(((preds == 0) & (labels == 1)).sum())

    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-12)
    precision = TP / (TP + FP + 1e-12) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN + 1e-12) if (TP + FN) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall + 1e-12) if (precision + recall) > 0 else 0.0

    # You could also compute AUROC via sklearn, if installed:
    try:
        from sklearn.metrics import roc_auc_score
        auroc = roc_auc_score(labels, all_logits)
    except ImportError:
        auroc = float("nan")

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc,
        "tn": TN,
        "fp": FP,
        "fn": FN,
        "tp": TP,
    }


def main(args):
    # --- Load train/val/test embedding files ---
    train_ds = EmbeddingSequenceDataset(
        embeddings_path=Path(args.train_emb_dir) / "embeddings.npy",
        labels_path=Path(args.train_emb_dir) / "labels.npy",
        seq_len=args.seq_len,
    )
    val_ds = EmbeddingSequenceDataset(
        embeddings_path=Path(args.val_emb_dir) / "embeddings.npy",
        labels_path=Path(args.val_emb_dir) / "labels.npy",
        seq_len=args.seq_len,
    )
    test_ds = EmbeddingSequenceDataset(
        embeddings_path=Path(args.test_emb_dir) / "embeddings.npy",
        labels_path=Path(args.test_emb_dir) / "labels.npy",
        seq_len=args.seq_len,
    )

    # --- DataLoaders ---
    # Respect requested device if available, otherwise fall back to CPU
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )

    # --- Build model ---
    example_emb, _ = train_ds[0]
    input_dim = example_emb.shape[1]  # embedding_dim
    if args.model == "gru":
        model = GRUSequenceModel(
            input_dim=input_dim,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            n_classes=1,
        ).to(device)
    else:
        model = TransformerSequenceModel(
            input_dim=input_dim,
            num_heads=args.num_heads,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            n_classes=1,
        ).to(device)

    # --- Loss, optimizer, scheduler ---
    # If your data is imbalanced per‐window, you could compute pos_weight here, but
    # for simplicity this script uses plain BCE.
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # --- Prepare logging ---
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "gru_metrics.csv"
    with metrics_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss", "val_loss", "auroc", "f1",
            "precision", "recall", "accuracy", "tn", "fp", "fn", "tp"
        ])

    best_val_auroc = 0.0
    patience_counter = 0

    # --- Training Loop ---
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        print(
            f"[Epoch {epoch:03d}] train_loss={train_loss:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  AUROC={val_metrics['auroc']:.3f}  "
            f"F1={val_metrics['f1']:.3f} Precision={val_metrics['precision']:.3f}  "
            f"Recall={val_metrics['recall']:.3f} Accuracy={val_metrics['accuracy']:.3f}  "
            f"TN={val_metrics['tn']} FP={val_metrics['fp']} FN={val_metrics['fn']} TP={val_metrics['tp']}"
        )

        # Log to CSV
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

        # Save best on AUROC
        if val_metrics["auroc"] > best_val_auroc + 1e-4:
            best_val_auroc = val_metrics["auroc"]
            torch.save(model.state_dict(), out_dir / "best_gru.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("[INFO] Early stopping triggered.")
                break

    print(f"[INFO] Best VAL AUROC = {best_val_auroc:.3f} 👍 saved to {out_dir/'best_gru.pt'}")

    # --- Final evaluation on test set ---
    print("[INFO] Evaluating on TEST set with best_gru.pt")
    model.load_state_dict(torch.load(out_dir / "best_gru.pt", map_location=device))
    test_metrics = evaluate(model, test_loader, criterion, device)
    print(
        f"--- TEST METRICS ---  AUROC={test_metrics['auroc']:.3f}  "
        f"F1={test_metrics['f1']:.3f}  Precision={test_metrics['precision']:.3f}  "
        f"Recall={test_metrics['recall']:.3f}  Accuracy={test_metrics['accuracy']:.3f}  "
        f"TN={test_metrics['tn']} FP={test_metrics['fp']} FN={test_metrics['fn']} TP={test_metrics['tp']}"
    )

    # Append TEST line to CSV
    with metrics_path.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "TEST", "-", "-", test_metrics['auroc'], test_metrics['f1'],
            test_metrics['precision'], test_metrics['recall'], test_metrics['accuracy'],
            test_metrics['tn'], test_metrics['fp'], test_metrics['fn'], test_metrics['tp']
        ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a GRU on fixed‐length embedding sequences."
    )
    parser.add_argument(
        "--train_emb_dir",
        type=Path,
        required=True,
        help="Directory containing embeddings.npy & labels.npy for TRAIN.",
    )
    parser.add_argument(
        "--val_emb_dir",
        type=Path,
        required=True,
        help="Directory containing embeddings.npy & labels.npy for VAL.",
    )
    parser.add_argument(
        "--test_emb_dir",
        type=Path,
        required=True,
        help="Directory containing embeddings.npy & labels.npy for TEST.",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=6,
        help="Number of consecutive windows per sequence (e.g. 6 windows → 30 s).",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=64,
        help="Hidden size of the GRU.",
    )
    parser.add_argument(
        "--model",
        choices=["gru", "transformer"],
        default="gru",
        help="Sequence model type",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=4,
        help="Transformer num heads (if model=transformer)",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Transformer feedforward dim (if model=transformer)",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=1,
        help="Number of layers in the sequence model.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size (number of sequences per batch).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Max number of epochs for GRU training.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for Adam optimizer.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience (on validation AUROC).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda or cpu",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="Directory where GRU checkpoints & metrics.csv will be saved.",
    )
    args = parser.parse_args()
    main(args)

