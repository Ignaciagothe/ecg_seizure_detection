# ECG Seizure Detection


This repository contains research code for detecting epileptic seizures from single lead ECG recordings using deep learning.  The project explores window based models such as **InceptionTime** with squeeze–and–excitation as well as temporal convolutional networks (TCN).  Utilities are provided to compute window embeddings and to train sequence models (GRU or Transformer).  A hierarchical model can also be trained end‑to‑end on sequences of windows.



Implementation of InceptionTime with SE is found in `src/models/inception.py`. Each `InceptionBlock` applies multiple 1-D convolutions with different kernel sizes, optional squeeze-excitation, residual connections and dropout. The final classifier performs global average pooling followed by a linear layer. The repository also includes a `HierarchicalSeizureModel` that pairs the window encoder with a GRU, LSTM or Transformer sequence module; this model can be trained end-to-end via `train_hierarchical.py`

src/models/tcn.py defines a dilated, causal TCN with several TemporalBlock layers that include residual connections and dropout, followed by global average pooling and a linear head

Training is done via train.py, which allows choosing between the InceptionTime and TCN backbones, with optional focal loss for class imbalance. The code computes pos_weight from the training labels and performs early stopping based on validation AUROC. It then saves a “best model” checkpoint and evaluates on a held‑out test set

A separate script (compute_embeddings.py) loads a pretrained InceptionTime model, extracts embeddings for each window, and writes them to disk. These embeddings can then feed into a GRU sequence model (train_gru.py) that predicts per-window seizure probability across fixed-length sequences. The GRU training loop reports AUROC, F1, precision, recall, and accuracy metrics


### Baseline Architectures (starting points for seizure detection from single-lead ECG):

- InceptionTime with SE and residual connections is well-suited for 1‑D ECG signals, capturing local patterns through multiple kernel sizes.

- The dilated causal TCN similarly handles sequential data efficiently and can model long-range dependencies through dilation.
- The pipeline optionally combines window-level embeddings with a GRU or Transformer to incorporate longer context.
- You can still train the window classifier and sequence model separately, but `train_hierarchical.py` now supports training them jointly so the encoder learns features suited to the downstream sequence model.


### Further improvements /next steps include 

- experimenting with attention-based architectures 
- self-supervised learning:

- Transformer-based sequence models have become competitive for time-series tasks and could be explored as an alternative to the GRU for capturing long-range dependencies while maintaining efficiency with lightweight architectures (e.g., temporal convolution plus attention).

- Consider augmenting training with band-pass filtering, STFT-based spectrogram inputs, or other feature representations if the raw single-lead signal is noisy.

- When computing embeddings, the script reads the saved hyperparameters from `hparams.json` so the architecture matches the training configuration.

### Alternative approaches:

If computational resources allow, self-supervised pretraining on large amounts of unlabeled ECG:
- Contrastive learning 
- Masked reconstruction 
This may yield better representations for seizure detection.

- A 2‑D representation (such as spectrogram or wavelet transform) combined with lightweight 2‑D CNN 
- ConvLSTM architectures 



### Repository structure

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

### Installation

Create a Python environment (Python 3.8+ recommended) and install the dependencies:

```bash
pip install -r requirements.txt
```

PyTorch is listed in `requirements.txt` but you may want to install a version
that matches your system and CUDA setup. Visit <https://pytorch.org> for
instructions.

### Data preprocessing

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

### Training a window model

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

### Computing embeddings

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

```bash
python train_hierarchical.py \
  --train_npz data/processed/windows_train.npz \
  --val_npz data/processed/windows_val.npz \
  --test_npz data/processed/windows_test.npz \
  --seq_len 6 \
  --out_dir runs/hier
```


