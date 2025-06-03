import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

def collate_fn(batch):
    """Stack numpy → torch and add the (B,1,W) channel dim."""
    xb, yb = zip(*batch)
    xb_t = torch.tensor(np.stack(xb), dtype=torch.float32).unsqueeze(1)
    yb_t = torch.tensor(np.array(yb), dtype=torch.float32)
    return xb_t, yb_t




class ECGWindowDataset(Dataset):
    """Loads pre chunked numpy arrays of windows + labels."""

    def __init__(self, npz_path: Path):
        super().__init__()
        loaded = np.load(npz_path)
        self.x = loaded["x"]  # shape (N, W)
        self.y = loaded["y"]  # shape (N,)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]
