# ECG Seizure Detection

This repository contains a collection of scripts and models used to detect seizure events from electrocardiogram (ECG) recordings.  The project is organised as a lightweight pipeline:

1. **Pre-processing** of raw CSV traces into labelled windows.
2. **Window-level training** with an InceptionTime or TCN backbone.
3. **Sequence models** (GRU/LSTM/Transformer) that operate on sequences of window embeddings.
4. Optional **hierarchical** model which integrates the two stages.

The code is written in pure PyTorch and depends only on common scientific Python packages (see [`requirements.txt`](requirements.txt)).

## Repository layout

```
.
├── compute_embeddings.py   # Generate embeddings from a trained window model
├── train.py                # Train window-level classifier
├── train_gru.py            # Train GRU/Transformer on embedding sequences
├── train_hierarchical.py   # End-to-end hierarchical training
├── src/
│   ├── datasets.py         # Dataset classes
│   ├── preprocessing.py    # Signal filtering and window creation
│   ├── utils.py            # Training utilities and metrics
│   └── models/             # InceptionTime, TCN and sequence models
└── notebooks/              # Example notebooks and exploration scripts
```

The actual ECG data is not included in the repository.  Training scripts expect CSV files with columns similar to `Time [s]`, `Signal [mV]` and `Seizure [bool]`.

## Pre-processing

The first step converts raw traces into fixed-size windows for model training.  This is handled by [`src/preprocessing.py`](src/preprocessing.py).  Each trace is optionally denoised with a Butterworth band-pass filter and then sliced into overlapping windows.  A window is labelled as seizure when it contains at least a configurable number of positive samples.

Usage example:

```bash
python -m src.preprocessing \
    --data_dir raw_ecg \
    --out_path data/processed/windows_train.npz \
    --window_seconds 5 \
    --overlap 0.1 \
    --seizure_threshold 2
```

The resulting NPZ contains two arrays: `x` (windows) and `y` (labels).  The same script can be run to generate separate train/val/test splits.

## Window-level training

`train.py` trains a classifier on individual windows.  Two backbones are available:

- **InceptionTimeSE** – an InceptionTime network with optional squeeze‑and‑excite blocks (see [`src/models/inception.py`](src/models/inception.py)).
- **TCNClassifier** – a dilated causal temporal convolutional network ([`src/models/tcn.py`](src/models/tcn.py)).

Example command:

```bash
python train.py \
    --data_dir path/to/raw_csv \
    --npz_path data/processed/windows_train.npz \
    --out_dir runs/window_model \
    --model inception \
    --epochs 50
```

Metrics for each epoch are appended to `metrics.csv` inside the run directory together with a copy of the training hyper‑parameters.

## Generating embeddings

After training a window model you may want to compute embeddings for use by a sequence model.  `compute_embeddings.py` loads the saved `best_model_inception.pt` (or TCN equivalent) and writes `embeddings.npy` and `labels.npy`.

```bash
python compute_embeddings.py \
    --model_path runs/window_model/best_model_inception.pt \
    --npz_path data/processed/windows_train.npz \
    --out_dir data/emb/train
```

## Sequence models

`train_gru.py` trains a sequence model on the produced embeddings.  Embeddings are chunked into fixed-length sequences (e.g. 6 windows = 30 s) and fed to either a GRU or a lightweight Transformer ([`src/models/transformer.py`](src/models/transformer.py)).

```bash
python train_gru.py \
    --train_emb_dir data/emb/train \
    --val_emb_dir   data/emb/val \
    --test_emb_dir  data/emb/test \
    --seq_len 6 \
    --model gru \
    --out_dir runs/gru_model
```

The script logs training/validation metrics to `gru_metrics.csv` and saves the best model based on AUROC.

## Hierarchical model

`train_hierarchical.py` provides an end‑to‑end approach where the window encoder and sequence model are trained jointly.  It uses the `HierarchicalSeizureModel` defined in [`src/models/inception.py`](src/models/inception.py), which stacks an InceptionTime encoder with a GRU/LSTM/Transformer over the sequence of window embeddings.

Example usage:

```bash
python train_hierarchical.py \
    --train_npz data/processed/windows_train.npz \
    --val_npz   data/processed/windows_val.npz \
    --test_npz  data/processed/windows_test.npz \
    --out_dir   runs/hierarchical \
    --seq_len 6
```

## Notes on requirements

Install dependencies with

```bash
pip install -r requirements.txt
```

PyTorch should be installed separately for your platform (CUDA, CPU or MPS).  All scripts automatically select the best available device.

## Citing metrics

The training utilities compute AUROC, AUPRC, F1, precision, recall and accuracy for each epoch.  These are implemented in [`src/utils.py`](src/utils.py) and logged to CSV for later analysis.

## License

This repository is provided as an educational example.  No licence information was supplied with the original code.

python train_hierarchical.py \
    --train_npz data/processed/windows_train.npz \
    --val_npz data/processed/windows_val.npz \
    --test_npz data/processed/windows_test.npz \
    --out_dir runs/hierarchical \
    --seq_len 6 \
    --seq_stride 1 \
    --batch_size 32 \
    --epochs 50 \
    --lr 0.001 \
    --focal_gamma 2.0 \
    --hidden_size 64 \
    --num_layers 2 \
    --seq_model gru

# Example 2: Train with pretrained InceptionTime encoder
python train_hierarchical.py \
    --train_npz data/processed/windows_train.npz \
    --val_npz data/processed/windows_val.npz \
    --test_npz data/processed/windows_test.npz \
    --out_dir runs/hierarchical_pretrained \
    --pretrained_inception runs/20250601_045007/best_model_inception.pt \
    --freeze_encoder \
    --seq_len 6 \
    --seq_stride 1 \
    --batch_size 32 \
    --epochs 20 \
    --lr 0.0001 \
    --focal_gamma 2.0 \
    --hidden_size 64 \
    --num_layers 2 \
    --seq_model gru

# Example 3: Train with LSTM instead of GRU
python train_hierarchical.py \
    --train_npz data/processed/windows_train.npz \
    --val_npz data/processed/windows_val.npz \
    --test_npz data/processed/windows_test.npz \
    --out_dir runs/hierarchical_lstm \
    --seq_len 6 \
    --seq_stride 1 \
    --batch_size 32 \
    --epochs 50 \
    --lr 0.001 \
    --focal_gamma 2.0 \
    --hidden_size 64 \
    --num_layers 2 \
    --seq_model lstm

# Example 4: Train with Transformer
python train_hierarchical.py \
    --train_npz data/processed/windows_train.npz \
    --val_npz data/processed/windows_val.npz \
    --test_npz data/processed/windows_test.npz \
    --out_dir runs/hierarchical_transformer \
    --seq_len 6 \
    --seq_stride 1 \
    --batch_size 16 \
    --epochs 50 \
    --lr 0.0005 \
    --focal_gamma 2.0 \
    --hidden_size 64 \
    --num_layers 2 \
    --seq_model transformer

# Example 5: Longer sequences (12 windows = 60 seconds)
python train_hierarchical.py \
    --train_npz data/processed/windows_train.npz \
    --val_npz data/processed/windows_val.npz \
    --test_npz data/processed/windows_test.npz \
    --out_dir runs/hierarchical_long \
    --seq_len 12 \
    --seq_stride 3 \
    --batch_size 16 \
    --epochs 50 \
    --lr 0.001 \
    --focal_gamma 2.0 \
    --hidden_size 128 \
    --num_layers 2 \
    --seq_model gru