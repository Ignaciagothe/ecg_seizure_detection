# ECG Seizure Detection

This repository contains code for detecting seizure events from electrocardiogram (ECG) recordings. The project is organised around three main steps:

1. **Pre-processing** of raw traces into labelled windows.
2. **Model training** using an InceptionTime-based encoder and a sequence model.
3. **Evaluation** on held‑out sequences.

The code is written in PyTorch and relies only on common scientific Python packages (see [`requirements.txt`](requirements.txt)).

## Repository layout

```
.
├── src/                # Datasets, preprocessing utilities and models
│   ├── datasets.py     # Dataset classes for windowed data
│   ├── preprocessing.py# Signal filtering and window extraction
│   ├── utils.py        # Training utilities and metrics
│   └── models/         # InceptionTime, TCN and transformer modules
├── train_hierarchical.py  # End‑to‑end training script
├── notebooks/          # Exploration notebooks and plotting helpers
└── requirements.txt
```

Raw ECG CSV files are not included. Scripts expect columns similar to `Time [s]`, `Signal [mV]` and `Seizure [bool]`.

## Installation

Install dependencies with

```bash
pip install -r requirements.txt
```

Install PyTorch separately for your platform (CUDA, CPU or MPS). All scripts automatically select the best available device.

## Pre-processing

`src/preprocessing.py` converts raw traces into fixed length windows for model training. Signals are band‑pass filtered and then segmented. Each window is labelled positive if it contains a configurable number of seizure samples.

Example command:

```bash
python -m src.preprocessing \
    --data_dir data/raw_ecg \
    --file_list data/splits/train.txt \
    --out_path data/processed/windows_train.npz \
    --window_seconds 5 \
    --overlap 0 \
    --seizure_threshold 1
```

The output NPZ file stores arrays `x` (windows) and `y` (labels). Repeat with different `file_list` arguments to generate validation and test sets.

## Training

`train_hierarchical.py` trains a hierarchical model that combines an InceptionTime encoder with a sequence model (GRU, LSTM or Transformer). The script loads window data directly from the NPZ files and optimises a binary classification loss over sequences of windows.

Basic usage:

```bash
python train_hierarchical.py \
    --train_npz data/processed/windows_train.npz \
    --val_npz data/processed/windows_val.npz \
    --test_npz data/processed/windows_test.npz \
    --out_dir runs/hierarchical \
    --seq_len 6
```

Metrics such as AUROC, AUPRC, F1, precision, recall and accuracy are written to `metrics.csv` in the output directory. The best model (based on validation AUROC) is saved as `best_model.pt`.

The script also supports loading a pretrained InceptionTime encoder with `--pretrained_inception` and optionally freezing its weights with `--freeze_encoder`.

## Utilities

- `src/utils.py` implements training loops, metrics computation and helper functions for dataset splitting.
- `src/models/` contains the neural network modules used by the project, including the hierarchical architecture.

## License

This repository is provided for educational purposes. No licence information was supplied with the original code.
