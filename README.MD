# ECG Seizure Detection

This repository contains research code for detecting epileptic seizures from single lead ECG recordings using deep learning.  The project explores window based models such as **InceptionTime** with squeeze–and–excitation as well as temporal convolutional networks (TCN).  It also includes utilities to compute window embeddings and to train a GRU sequence model on those embeddings.

## Repository structure

```
├── compute_embeddings.py       # compute window embeddings using a trained model
├── train.py                    # train window classifiers (InceptionTime or TCN)
├── train_gru.py                # train GRU on sequences of embeddings
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
and `--post_margin_seconds` to trim inter‑ictal tails.

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
Run it for your train/val/test NPZ files.

## Sequence model with GRU

`train_gru.py` trains a GRU over fixed‑length sequences of embeddings.  First
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

## License

This project is released under the MIT License.  See the `LICENSE` file for
details.
