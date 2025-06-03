#src/utils.py
import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score,recall_score, precision_score, accuracy_score
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from contextlib import nullcontext
from torch import nn

# --- FocalLoss implementation ---
class FocalLoss(nn.Module):
    """
    Binary Focal Loss with logits.

    Parameters
    ----------
    alpha : float
        Balancing factor for the positive class.  Use the same value you would
        pass as `pos_weight` to BCEWithLogitsLoss.
    gamma : float
        Focusing parameter γ from Lin et al. 2017 (default recommended: 2.0).
    """
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        prob_t = torch.exp(-bce_loss)
        focal_term = (1.0 - prob_t) ** self.gamma
        loss = self.alpha * focal_term * bce_loss
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
                # gradient clipping to avoid exploding weights
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


