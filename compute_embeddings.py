# compute_embeddings.py
import argparse
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from src.datasets import ECGWindowDataset, collate_fn
from src.models.inception import InceptionTimeSE  # make sure your PYTHONPATH is correct

def compute_and_save_embeddings(
    model_path: Path,
    npz_path: Path,
    out_dir: Path,
    batch_size: int = 256,
    device: str = "mps",
):
    """
    - model_path: path to best_model_inception.pt
    - npz_path: path to windows_{train,val,test}.npz
    - out_dir: where to write embeddings_{train,val,test}.npz
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    # 1) Reconstruct the exact same InceptionTimeSE architecture you used in train.py.
    window_encoder = InceptionTimeSE(
        n_blocks=6,           # or use args if you want CLI flexibility
        in_channels=1,
        n_classes=1,          # this is irrelevant for get_embedding()
        out_channels=32,
        bottleneck_channels=32,
        kernel_sizes=[10, 20, 40],
        use_se=True,
        use_residual=True,
    ).to(device)

    # 2) Load the saved weights
    checkpoint = torch.load(model_path, map_location=device)
    window_encoder.load_state_dict(checkpoint)
    window_encoder.eval()

    # 3) Instantiate dataset & dataloader
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
            # Assume collate_fn returns (signals, labels), where:
            #   signals: [batch_size, in_channels, time]
            #   labels:  [batch_size]
            signals, labels = batch
            signals = signals.to(device)        # (B, 1, window_length)
            emb = window_encoder.get_embedding(signals)  # (B, embedding_dim)
            all_embeddings.append(emb.cpu().numpy())
            all_labels.append(labels.numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)  # (N_windows, embedding_dim)
    all_labels = np.concatenate(all_labels, axis=0)          # (N_windows,)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "embeddings.npy", all_embeddings)
    np.save(out_dir / "labels.npy", all_labels)
    print(f"[INFO] Saved embeddings → {out_dir}/embeddings.npy")
    print(f"[INFO] Saved labels     → {out_dir}/labels.npy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute & save window embeddings using a pretrained InceptionTimeSE."
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        required=True,
        help="Path to best_model_inception.pt",
    )
    parser.add_argument(
        "--npz_path",
        type=Path,
        required=True,
        help="Path to windows_*.npz (train/val/test). E.g., data/processed/windows_train.npz",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="Directory where embeddings.npy & labels.npy will be saved",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="cuda or cpu",
    )
    args = parser.parse_args()

    compute_and_save_embeddings(
        args.model_path,
        args.npz_path,
        args.out_dir,
        batch_size=args.batch_size,
        device=args.device,
    )