# ECG Seizure Detection

This repository contains research code for detecting epileptic seizures from single lead ECG recordings using deep learning.  The project explores window based models such as **InceptionTime** with squeeze–and–excitation as well as temporal convolutional networks (TCN).  Utilities are provided to compute window embeddings and to train sequence models (GRU or Transformer).  A hierarchical model can also be trained end‑to‑end on sequences of windows.

## Repository structure

```
├── compute_embeddings.py       # compute window embeddings using a trained model
├── train.py                    # train window classifiers (InceptionTime or TCN)
├── train_gru.py                # train GRU or Transformer on embeddings
├── train_hierarchical.py       # end-to-end training on window sequences
├── src/
│   ├── datasets.py             # ECGWindowDataset and DataLoader helpers
│   ├── preprocessing.py        # utilities to create windowed NPZ datasets
│   ├── models/                 # InceptionTime and TCN implementations
│   └── utils.py                # training loops and metrics
└── notebooks/                  # exploratory notebooks and scripts
```

## Installation

Create a Python environment (Python 3.8+ recommended) and install the dependencies:

```bash
pip install -r requirements.txt
```

PyTorch is listed in `requirements.txt` but you may want to install a version
that matches your system and CUDA setup. Visit <https://pytorch.org> for
instructions.

## Data preprocessing

Raw ECG traces (CSV files) must be converted into fixed length windows before training.
`src/preprocessing.py` scans a directory of CSV files and produces a single NPZ
containing arrays `x` (windows) and `y` (labels).  Example:

```bash
python -m src.preprocessing \
  --data_dir path/to/csvs \
  --out_path data/processed/windows_train.npz \
  --window_seconds 15 \
  --overlap 0 \
  --seizure_threshold 2
```

The script offers additional options such as `--neg_to_pos` to balance classes
and `--post_margin_seconds` to trim inter‑ictal tails. Basic band‑pass filtering is applied during preprocessing to reduce noise in the single‑lead signals.

## Training a window model

Use `train.py` to train either the InceptionTime or TCN backbone.
The script creates a timestamped directory with metrics CSV, the best model and
logged hyper‑parameters.

```bash
python train.py \
  --data_dir path/to/raw_csvs \
  --npz_path data/processed/windows_train.npz \
  --out_dir runs/ \
  --model inception \
  --epochs 50 \
  --batch_size 128
```

After training you will find `best_model_inception.pt` (or `best_model_tcn.pt`) in
the created run directory.

## Computing embeddings

With a trained window encoder you can extract embeddings for each window using
`compute_embeddings.py`:

```bash
python compute_embeddings.py \
  --model_path runs/20230101_123456/best_model_inception.pt \
  --npz_path data/processed/windows_train.npz \
  --out_dir data/embeddings/train --device cpu
```

This writes `embeddings.npy` and `labels.npy` to the output directory.
The script automatically reads `hparams.json` from the same directory as the model
checkpoint so that the architecture matches the one used during training.
Run it for your train/val/test NPZ files.

## Sequence models

`train_gru.py` trains either a GRU or a lightweight Transformer over fixed‑length sequences of embeddings.  First
prepare embeddings for each split as above, then run:

```bash
python train_gru.py \
  --train_emb_dir data/embeddings/train \
  --val_emb_dir data/embeddings/val \
  --test_emb_dir data/embeddings/test \
  --seq_len 6 \
  --out_dir runs/gru
```

A CSV with metrics and the best checkpoint `best_gru.pt` will be saved under the
given directory.

## Hierarchical model

`train_hierarchical.py` trains the window encoder and sequence model jointly on sequences of raw windows. It supports GRU, LSTM or Transformer backbones for the temporal component.

## License

This project is released under the MIT License.  See the `LICENSE` file for
details.
