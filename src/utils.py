import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score,recall_score, precision_score, accuracy_score
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from contextlib import nullcontext
from torch import nn
from pathlib import Path
from torch.utils.data import DataLoader
from src.datasets import ECGWindowDataset, collate_fn
from src.models.inception import InceptionTimeSE 
import json
from collections import defaultdict
import random

class FocalLoss(nn.Module):
    """Binary focal loss with optional positive class weighting."""

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_factor = self.alpha * targets + (1 - targets)
        focal_weight = alpha_factor * (1 - p_t) ** self.gamma
        loss = focal_weight * bce_loss
        return loss.mean()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_prob = y_pred  
    y_hat = (y_prob >= 0.5).astype(int)
    metricas= {
        "auroc": roc_auc_score(y_true, y_prob),
        "auprc": average_precision_score(y_true, y_prob),
        "f1": f1_score(y_true, y_hat),
        'precision' : precision_score(y_true, y_hat),
        'recall' : recall_score(y_true, y_hat),
        'accuracy' : accuracy_score(y_true, y_hat),
        "tn": confusion_matrix(y_true, y_hat)[0, 0],
        "fp": confusion_matrix(y_true, y_hat)[0, 1],
        "fn": confusion_matrix(y_true, y_hat)[1, 0],
        "tp": confusion_matrix(y_true, y_hat)[1, 1],
    }
    return metricas


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    iterable = tqdm(loader, desc="train", leave=False)
    for xb, yb in iterable:
        xb = xb.to(device)
        yb = yb.float().to(device)
        optimizer.zero_grad()
        context = (torch.autocast(device_type=xb.device.type, dtype=torch.float16)
                   if xb.device.type in ("cuda", "mps") else nullcontext())
        with context:
            logits = model(xb)
            loss = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
    return running_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    y_true, y_pred = [], []
    running_loss = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.float().to(device)
            context = (torch.autocast(device_type=xb.device.type, dtype=torch.float16)
                       if xb.device.type in ("cuda", "mps") else nullcontext())
            with context:
                logits = model(xb)
                if torch.isnan(logits).any():
                    raise ValueError("NaN detected in model outputs during evaluation.Check learning rate / gradient explosion.")
                loss = criterion(logits, yb)
            running_loss += loss.item() * xb.size(0)
            y_true.append(yb.cpu().numpy())
            y_pred.append(torch.sigmoid(logits).cpu().numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = running_loss / len(loader.dataset)
    return metrics


def key_fn(fname: Path, group):
    if group == "patient":
        return fname.stem.split("_")[0]          # aaaaaaac
    if group == "session":
        return "_".join(fname.stem.split("_")[:2])  # aaaaaaac_s001
    return fname.stem                             # full filename

def make_split(data_dir="data/raw_ecg",val_ratio=0.10, test_ratio=0.10, group="patient", seed=42):

    root = Path(data_dir)
    all_csvs = sorted(root.rglob("*.csv"))
    random.seed(seed)

    bucket = defaultdict(list)
    for f in all_csvs:
        bucket[key_fn(f, group)].append(f)

    keys = list(bucket.keys())
    random.shuffle(keys)

    n = len(keys)
    n_test  = int(round(n * test_ratio))
    n_val   = int(round(n * val_ratio))

    test_keys  = set(keys[:n_test])
    val_keys   = set(keys[n_test:n_test+n_val])
    train_keys = set(keys[n_test+n_val:])

    splits = {"train": train_keys, "val": val_keys, "test": test_keys}
    out_dir = root.parent / "splits"
    out_dir.mkdir(exist_ok=True)
    for split, keyset in splits.items():
        with open(out_dir/f"{split}.txt", "w") as fh:
            for k in sorted(keyset):
                for f in bucket[k]:
                    fh.write(str(f.relative_to(data_dir)) + "\n")
        print(f"{split:>5}: {len(keyset):3d} {group}s "
              f"→ {sum(len(bucket[k]) for k in keyset):4d} files")


def compute_and_save_embeddings(
    model_path,
    npz_path,
    out_dir,
    batch_size = 256,
    device ="mps",
    hparams_path= None):
    if device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if hparams_path is None:
        hparams_path = model_path.parent / "hparams.json"
    if hparams_path.exists():
        with hparams_path.open() as f:
            hp = json.load(f)
    else:
        hp = {}

    window_encoder = InceptionTimeSE(
        n_blocks=hp.get("n_blocks", 6),
        in_channels=1,
        n_classes=1,
        out_channels=hp.get("out_channels", 32),
        bottleneck_channels=hp.get("bottleneck_channels", 32),
        kernel_sizes=hp.get("kernel_sizes", [10, 20, 40]),
        use_se=not hp.get("no_se", False),
        use_residual=True,
    ).to(device)


    checkpoint = torch.load(model_path, map_location=device)
    window_encoder.load_state_dict(checkpoint)
    window_encoder.eval()

   
    dataset = ECGWindowDataset(npz_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            signals, labels = batch
            signals = signals.to(device)     
            emb = window_encoder.get_embedding(signals)  
            all_embeddings.append(emb.cpu().numpy())
            all_labels.append(labels.numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)  
    all_labels = np.concatenate(all_labels, axis=0)         

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "embeddings.npy", all_embeddings)
    np.save(out_dir / "labels.npy", all_labels)
    print(f"[INFO] Saved embeddings → {out_dir}/embeddings.npy")
    print(f"[INFO] Saved labels     → {out_dir}/labels.npy")

