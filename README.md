# ECG Seizure Detection

This repository provides PyTorch implementations to detect seizure events in ECG traces. It covers data preparation, model training and evaluation.

## Repository overview

- `src/` – datasets, preprocessing and models
- `train_hierarchical.py` – training based on preprocessed windows
- `train_hybrid_pipeline.py` – end‑to‑end pipeline from raw files
- `notebooks/` – exploration and plotting helpers
- `requirements.txt` – list of Python packages

Raw ECG files are not included. Each CSV should contain at least `Time [s]`, `Signal [mV]` and `Seizure [bool]` columns.

## Installation

Install the required packages and a compatible PyTorch build:

```bash
pip install -r requirements.txt
```

## Data preparation

Run the preprocessing script to convert raw traces into fixed length windows. The command below generates an NPZ archive containing `x` (windows) and `y` (labels) arrays.

```bash
python -m src.preprocessing \
    --data_dir data/raw_ecg \
    --file_list data/splits/train.txt \
    --out_path data/processed/windows_train.npz \
    --window_seconds 5
```

Create separate NPZ files for training, validation and testing.

## Training options

### Hierarchical model

Use the hierarchical script to train a sequence model on preprocessed windows:

```bash
python train_hierarchical.py \
    --train_npz data/processed/windows_train.npz \
    --val_npz data/processed/windows_val.npz \
    --test_npz data/processed/windows_test.npz \
    --out_dir runs/hierarchical
```

Metrics are saved to `metrics.csv` in the chosen `out_dir`. The best weights are stored as `best_model.pt`.

### Hybrid Transformer-CNN pipeline

Alternatively run the hybrid pipeline on raw ECG files. The script will optionally create train/val/test splits if they are not provided.

```bash
python train_hybrid_pipeline.py \
    --data_dir data/raw_ecg \
    --out_dir runs/hybrid
```

## License

This project is provided for educational purposes; no license was supplied with the original code.
