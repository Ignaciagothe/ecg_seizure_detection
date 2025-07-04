#src/preprocessing.py
import numpy as np
import pandas as pd
import math
import json
from scipy.signal import butter, filtfilt, iirnotch
from tqdm import tqdm
from pathlib import Path

def bandpass_filter(signal, fs, lowcut=0.5, highcut=40.0, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)

def notch_filter(signal, fs, freq, q=30.0):
    b, a = iirnotch(w0=freq, Q=q, fs=fs)
    return filtfilt(b, a, signal)

def segment_trace(signal,label, window_size, stride, seizure_threshold):
    
    assert len(signal) == len(label)
    n_windows = max(1, math.floor((len(signal) - window_size) / stride) + 1)
    xs = np.empty((n_windows, window_size), dtype=np.float32)
    ys = np.empty(n_windows, dtype=np.int64)
    for i in range(n_windows):
        start = i * stride
        end = start + window_size
        window = signal[start:end]
        target = label[start:end]
        if len(window) < window_size:  # pad last window if short
            pad = window_size - len(window)
            window = np.pad(window, (0, pad), mode="constant")
            target = np.pad(target, (0, pad), mode="constant")
        xs[i] = window
        ys[i] = 1 if target.sum() > seizure_threshold else 0
    return xs, ys


def preprocess_dataset(
    data_dir,
    out_path,
    sample_period,
    window_seconds,
    overlap,
    seizure_threshold,
    neg_to_pos,
    post_margin_seconds: float = 0.0,
    file_list: str | None = None,
    low_cut: float = 0.5,
    high_cut: float = 40.0,
    notch: float | None = None
):
 
    window_size = int(window_seconds / sample_period)
    stride = int(window_size * (1 - overlap))

    if file_list is not None:
        with open(file_list) as f:
            names = [ln.strip() for ln in f if ln.strip()]
        csv_files = [Path(data_dir) / n for n in names]
    else:
        csv_files = sorted(Path(data_dir).glob("*.csv"))

    all_x, all_y = [], []
    for csv in tqdm(csv_files, desc="Pre‑processing", unit="file"):
        df = pd.read_csv(csv)
        voltage = df["Signal [mV]"].values.astype(np.float32)
        fs = int(1 / sample_period)
        if notch is not None:
            voltage = notch_filter(voltage, fs=fs, freq=notch)
        voltage = bandpass_filter(voltage, fs=fs,
                          lowcut=low_cut, highcut=high_cut)

        label = df["Seizure [bool]"].values.astype(np.int64)
        if label.size > 0 and label[0] == 1:
            label[0] = 0

        # 2) Remove post‑ictal / inter‑ictal tail to avoid confusing the model.
        if label.any():  # at least one seizure sample exists
            last_sz = np.nonzero(label)[0][-1]
            cut_idx = last_sz + int(post_margin_seconds / sample_period)
            voltage = voltage[:cut_idx]
            label   = label[:cut_idx]

        xs, ys = segment_trace(voltage, label,
                            window_size=window_size,
                            stride=stride,
                            seizure_threshold=seizure_threshold)
        
        if neg_to_pos is not None:
            pos_idx = np.where(ys == 1)[0]
            neg_idx = np.where(ys == 0)[0]
            max_neg = int(len(pos_idx) * neg_to_pos)
            if max_neg < len(neg_idx):
                neg_idx = np.random.choice(neg_idx, max_neg, replace=False)
                keep = np.concatenate([pos_idx, neg_idx])
                xs, ys = xs[keep], ys[keep]
        all_x.append(xs)
        all_y.append(ys)

    x = np.vstack(all_x)
    y = np.concatenate(all_y)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)   
    np.savez_compressed(out_path, x=x, y=y)
    print(f"Saved {x.shape[0]} windows @ {out_path}")

    with open(Path(out_path).with_suffix(".files.json"), "w") as f:
        json.dump([str(p) for p in csv_files], f, indent=2)

if __name__ == "__main__":
    import argparse, sys
    p = argparse.ArgumentParser("Pre‑compute ECG windows")
    p.add_argument("--data_dir", required=True, type=str)
    p.add_argument("--out_path", required=True, type=str)
    p.add_argument("--sample_period", default=1/256, type=float,
                   help="Sampling period in seconds (default 256 Hz).")
    p.add_argument("--window_seconds", default=5.0, type=float)
    p.add_argument("--overlap", default=0, type=float)
    p.add_argument("--seizure_threshold", default=1, type=int)
    p.add_argument("--neg_to_pos", default=3.0, type=float)
    p.add_argument("--post_margin_seconds", default=0.0, type=float,
                   help="Seconds to retain after the last seizure point before cutting (default 0).")
    p.add_argument("--file_list", type=str, default=None,
                   help="Path to txt with filenames (one per line) to include.")
    p.add_argument("--low_cut",  type=float, default=0.5)
    p.add_argument("--high_cut", type=float, default=40.0)
    p.add_argument("--notch",    type=float, default=None)

    args = p.parse_args()

    preprocess_dataset(
        data_dir=args.data_dir,
        out_path=args.out_path,
        sample_period=args.sample_period,
        window_seconds=args.window_seconds,
        overlap=args.overlap,
        seizure_threshold=args.seizure_threshold,
        neg_to_pos=args.neg_to_pos,
        post_margin_seconds=args.post_margin_seconds,
        file_list=args.file_list,
        low_cut=args.low_cut,
        high_cut=args.high_cut,
        notch=args.notch
    )



"""
python -m src.preprocessing \
  --data_dir data/raw_ecg \
  --file_list data/splits/train.txt \
  --out_path data/processed/windows_train.npz \
  --window_seconds 5 \
  --overlap 0 \
  --seizure_threshold 1 \
  --neg_to_pos 3

python -m src.preprocessing \
    --data_dir   data/raw_ecg \
    --file_list  data/splits/val.txt \
    --out_path   data/processed/windows_val.npz \
    --window_seconds    5 \
    --overlap 0 \
    --neg_to_pos 3  \
    --seizure_threshold 1

    python -m src.preprocessing \
    --data_dir   data/raw_ecg \
    --file_list  data/splits/test.txt \
    --out_path   data/processed/windows_test.npz \
    --window_seconds    5 \
    --overlap 0 \
    --neg_to_pos 3 \
    --seizure_threshold 1
"""