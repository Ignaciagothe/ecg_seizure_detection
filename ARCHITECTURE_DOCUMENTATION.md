# ECG Seizure Detection - Comprehensive Repository Documentation

## Table of Contents
1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Data Pipeline](#data-pipeline)
4. [Model Architectures](#model-architectures)
5. [Training Scripts](#training-scripts)
6. [Algorithms and Techniques](#algorithms-and-techniques)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Usage Examples](#usage-examples)
9. [Technical Implementation Details](#technical-implementation-details)

## Overview

This repository implements deep learning solutions for epileptic seizure detection from single-lead ECG signals using PyTorch. The project provides two main approaches:

1. **Hierarchical Model**: Uses preprocessed windows with CNN feature extraction followed by sequential modeling
2. **Hybrid Transformer-CNN Pipeline**: End-to-end processing from raw ECG files with CNN + Transformer architecture

### Key Features
- Multi-modal deep learning approaches (CNN, RNN, Transformer)
- Comprehensive preprocessing pipeline with filtering and segmentation
- Attention mechanisms for temporal modeling
- Focal loss for class imbalance handling
- Comprehensive evaluation metrics (AUROC, AUPRC, F1, etc.)

## Repository Structure

```
ecg_seizure_detection/
├── src/                          # Core source code
│   ├── models/                   # Neural network architectures
│   │   ├── inception.py         # InceptionTime + Hierarchical models
│   │   ├── hybrid_transformer_cnn.py # Hybrid Transformer-CNN
│   │   ├── tcn.py              # Temporal Convolutional Network
│   │   └── transformer.py       # Pure Transformer model
│   ├── datasets.py              # Data loading utilities
│   ├── preprocessing.py         # Signal processing pipeline
│   └── utils.py                 # Training, evaluation, and utility functions
├── train_hierarchical.py        # Training script for hierarchical models
├── train_hybrid_pipeline.py     # Training script for hybrid pipeline
├── notebooks/                   # Jupyter notebooks for analysis
│   ├── exploration_signals_plots.py
│   └── training_analysis.ipynb
├── README.md                    # Basic usage instructions
└── requirements.txt             # Python dependencies
```

## Data Pipeline

### Input Data Format
The repository expects CSV files with the following columns:
- `Time [s]`: Timestamp in seconds
- `Signal [mV]`: ECG voltage measurements in millivolts  
- `Seizure [bool]`: Binary seizure labels (0=normal, 1=seizure)

### Preprocessing Pipeline (`src/preprocessing.py`)

#### 1. Signal Filtering
```python
def bandpass_filter(signal, fs, lowcut=0.5, highcut=40.0, order=5)
def notch_filter(signal, fs, freq, q=30.0)
```
- **Bandpass Filter**: Removes baseline drift (< 0.5 Hz) and high-frequency noise (> 40 Hz)
- **Notch Filter**: Optional 50/60 Hz powerline interference removal
- **Implementation**: Uses Butterworth filters with zero-phase filtering (filtfilt)

#### 2. Segmentation
```python
def segment_trace(signal, label, window_size, stride, seizure_threshold)
```
- **Window Size**: Typically 5 seconds (1280 samples at 256 Hz)
- **Overlap**: Configurable overlap between windows (default 0%)
- **Label Assignment**: Window labeled as seizure if > threshold seizure samples
- **Padding**: Last window padded if insufficient samples

#### 3. Class Balancing
- **Negative-to-Positive Ratio**: Configurable (default 3:1)
- **Post-ictal Margin**: Optional removal of samples after seizure events
- **Random Undersampling**: Reduces majority class samples

### Data Splits
The system supports three splitting strategies:
- **File-based**: Split by complete files
- **Patient-based**: Split by patient IDs  
- **Session-based**: Split by recording sessions

## Model Architectures

### 1. InceptionTime-Based Models (`src/models/inception.py`)

#### InceptionTimeSE
```python
class InceptionTimeSE(nn.Module):
    def __init__(self, n_blocks=6, in_channels=1, n_classes=1, 
                 out_channels=32, bottleneck_channels=32,
                 kernel_sizes=[9, 19, 39], use_se=False, 
                 use_residual=True, dropout=0.1)
```

**Architecture Components:**
- **Inception Blocks**: Multi-scale temporal feature extraction
- **Kernel Sizes**: [9, 19, 39] samples capture different temporal scales
- **Squeeze-Excite (SE)**: Optional channel attention mechanism
- **Residual Connections**: Skip connections for gradient flow
- **Bottleneck Layers**: Dimensional reduction for efficiency

**Forward Pass:**
1. Multi-scale convolutions with different kernel sizes
2. Concatenation of all temporal scales
3. Batch normalization and dropout
4. Optional SE attention
5. Residual connection
6. Global average pooling and classification

#### HierarchicalSeizureModel
```python
class HierarchicalSeizureModel(nn.Module):
    def __init__(self, window_encoder: InceptionTimeSE,
                 hidden_size=64, seq_model_type='gru',
                 num_layers=1, n_classes=1, dropout=0.1)
```

**Two-Stage Architecture:**
1. **Window Encoder**: InceptionTimeSE extracts features from individual windows
2. **Sequence Model**: GRU/LSTM/Transformer models temporal dependencies
3. **Attention Pooling**: Weighted aggregation of sequence outputs
4. **Classification**: Final seizure prediction

**Mathematical Formulation:**
```
h_t = WindowEncoder(x_t)                    # Window features
s_1, ..., s_T = SequenceModel(h_1, ..., h_T) # Sequential modeling  
c, α = AttentionPool(s_1, ..., s_T)        # Attention pooling
y = Classifier(c)                           # Final prediction
```

### 2. Hybrid Transformer-CNN (`src/models/hybrid_transformer_cnn.py`)

#### Architecture Overview
```python
class HybridTransformerCNN(nn.Module):
    def __init__(self, in_channels=1, conv_channels=(32, 64),
                 num_heads=4, hidden_dim=128, num_layers=2,
                 n_classes=1, dropout=0.1)
```

**Pipeline Components:**
1. **CNN Feature Extractor**: Local pattern recognition
2. **Positional Encoding**: Temporal position information
3. **Attention Denoiser**: Multi-head attention for noise reduction
4. **Transformer Encoder**: Long-range dependency modeling
5. **Global Pooling**: Temporal aggregation
6. **Classification Head**: Final prediction

#### Positional Encoding
```python
class PositionalEncoding(nn.Module):
    def __init__(self, dim, dropout=0.1, max_len=10000)
```
- **Sinusoidal Encoding**: Standard transformer positional encoding
- **Dynamic Extension**: Handles sequences longer than max_len
- **Formulation**: PE(pos,2i) = sin(pos/10000^(2i/d)), PE(pos,2i+1) = cos(pos/10000^(2i/d))

#### Attention Denoiser
```python
class AttentionDenoiser(nn.Module)
```
- **Multi-Head Attention**: Learns to focus on relevant temporal regions
- **Noise Reduction**: Filters out artifacts and noise patterns
- **Residual Connections**: Preserves important signal components

### 3. Temporal Convolutional Network (`src/models/tcn.py`)

#### TCNClassifier Architecture
```python
class TCNClassifier(nn.Module):
    def __init__(self, in_channels=1, n_classes=1,
                 channels=(16, 32, 32, 64), kernel_size=5,
                 dropout=0.1)
```

**Key Features:**
- **Dilated Convolutions**: Exponentially increasing receptive field
- **Causal Convolutions**: No future information leakage
- **Residual Connections**: Gradient flow optimization
- **Temporal Block Structure**: Repeated conv-norm-relu-dropout blocks

#### Temporal Block
```python
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout)
```
- **Dilation Pattern**: 2^i for block i (1, 2, 4, 8, ...)
- **Receptive Field**: Grows exponentially with depth
- **Chomp Operation**: Removes future-looking padded values

### 4. Pure Transformer (`src/models/transformer.py`)

#### TransformerSequenceModel
```python
class TransformerSequenceModel(nn.Module):
    def __init__(self, input_dim, num_heads=4, hidden_dim=128,
                 num_layers=2, n_classes=1, dropout=0.1)
```

**Architecture:**
- **Multi-Head Self-Attention**: Captures temporal dependencies
- **Feed-Forward Networks**: Non-linear transformations
- **Layer Normalization**: Stabilizes training
- **Positional Encoding**: Inherent from input embeddings

## Training Scripts

### 1. Hierarchical Training (`train_hierarchical.py`)

#### Key Features
- **Two-Stage Training**: Separate window and sequence modeling
- **Preprocessing Required**: Uses pre-segmented `.npz` files
- **Memory Efficient**: Processes fixed-size windows
- **Model Variants**: Supports GRU, LSTM, Transformer sequence models

#### Usage Example
```bash
python train_hierarchical.py \
    --train_npz data/processed/windows_train.npz \
    --val_npz   data/processed/windows_val.npz \
    --test_npz  data/processed/windows_test.npz \
    --out_dir   runs/hierarchical \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-3
```

#### Model Configuration
```python
def build_model(args, device):
    window_encoder = InceptionTimeSE(
        n_blocks=args.n_blocks,
        in_channels=1,
        n_classes=1,
        out_channels=args.out_channels,
        use_se=args.use_se,
        dropout=args.dropout
    )
    
    model = HierarchicalSeizureModel(
        window_encoder=window_encoder,
        hidden_size=args.hidden_size,
        seq_model_type=args.seq_model_type,
        num_layers=args.num_layers,
        dropout=args.dropout
    )
    return model
```

### 2. Hybrid Pipeline Training (`train_hybrid_pipeline.py`)

#### Key Features
- **End-to-End Processing**: Direct training from raw CSV files
- **Automatic Preprocessing**: Handles segmentation during training
- **Memory Intensive**: Processes variable-length sequences
- **Transformer-CNN Architecture**: Combined local and global modeling

#### Usage Example
```bash
python train_hybrid_pipeline.py \
    --data_dir data/raw_ecg \
    --out_dir  runs/hybrid \
    --epochs 50 \
    --batch_size 16 \
    --lr 1e-3 \
    --conv_channels 32 64 \
    --num_heads 4 \
    --num_layers 2
```

#### Preprocessing Integration
```python
def preprocess_splits(args, processed_dir, splits_dir):
    """Automatically preprocess raw data if needed"""
    # Creates train/val/test splits
    # Applies filtering and segmentation
    # Saves processed data for training
```

## Algorithms and Techniques

### 1. Loss Functions

#### Focal Loss
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        self.alpha = alpha    # Class weighting
        self.gamma = gamma    # Focusing parameter
```

**Mathematical Formulation:**
```
FL(p_t) = -α_t(1-p_t)^γ log(p_t)
where p_t = p if y=1 else 1-p
```

**Benefits:**
- **Class Imbalance**: α parameter handles class imbalance
- **Hard Example Mining**: γ parameter focuses on difficult examples
- **Gradient Modulation**: Reduces loss for well-classified examples

#### Binary Cross-Entropy with Class Weights
```python
pos_weight = neg_samples / (pos_samples + 1e-8)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

### 2. Optimization Strategies

#### Learning Rate Scheduling
```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
)
```
- **Adaptive Reduction**: Reduces LR when validation loss plateaus
- **Factor**: Multiplier for LR reduction (0.5 = halve learning rate)
- **Patience**: Number of epochs to wait before reduction

#### Early Stopping
```python
if val_metrics['auroc'] > best_auroc:
    best_auroc = val_metrics['auroc']
    torch.save(model.state_dict(), out_dir / 'best_model.pt')
    patience_counter = 0
else:
    patience_counter += 1
    if patience_counter >= args.patience:
        break
```

### 3. Attention Mechanisms

#### Attention Pooling
```python
class AttentionPool(nn.Module):
    def forward(self, x):
        scores = self.attn(torch.tanh(x)).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return context, weights
```

**Mathematical Formulation:**
```
e_i = f_att(tanh(h_i))    # Attention scores
α_i = softmax(e_i)        # Attention weights  
c = Σ α_i * h_i           # Weighted context
```

#### Multi-Head Self-Attention (Transformer)
- **Query, Key, Value**: Q, K, V = h*W_q, h*W_k, h*W_v
- **Attention Weights**: A = softmax(QK^T/√d_k)
- **Output**: O = AV

### 4. Regularization Techniques

#### Dropout
- **Model Regularization**: Prevents overfitting
- **Attention Dropout**: Applied to attention weights
- **Feature Dropout**: Applied to intermediate representations

#### Batch Normalization
- **Training Stabilization**: Normalizes layer inputs
- **Gradient Flow**: Improves gradient propagation
- **Regularization Effect**: Provides implicit regularization

## Evaluation Metrics

### Core Metrics (`src/utils.py`)

#### Binary Classification Metrics
```python
def compute_metrics(y_true, y_pred):
    y_hat = (y_pred >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_hat, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        "auroc": roc_auc_score(y_true, y_pred),
        "auprc": average_precision_score(y_true, y_pred),
        "f1": f1_score(y_true, y_hat, zero_division=0),
        "precision": precision_score(y_true, y_hat, zero_division=0),
        "recall": recall_score(y_true, y_hat, zero_division=0),
        "accuracy": accuracy_score(y_true, y_hat),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp
    }
    return metrics
```

#### Metric Interpretations

1. **AUROC (Area Under ROC Curve)**
   - Range: [0, 1], higher is better
   - Measures ranking quality across all thresholds
   - Robust to class imbalance when used with other metrics

2. **AUPRC (Area Under Precision-Recall Curve)**
   - Range: [0, 1], higher is better
   - More sensitive to class imbalance than AUROC
   - Critical for medical applications (seizure detection)

3. **F1-Score**
   - Harmonic mean of precision and recall
   - Balances false positives and false negatives
   - Good single metric for imbalanced data

4. **Precision (Positive Predictive Value)**
   - TP / (TP + FP)
   - Proportion of predicted seizures that are actual seizures
   - Critical for reducing false alarms

5. **Recall (Sensitivity)**
   - TP / (TP + FN)  
   - Proportion of actual seizures that are detected
   - Critical for patient safety

6. **Accuracy**
   - (TP + TN) / (TP + TN + FP + FN)
   - Overall correctness
   - Can be misleading with class imbalance

### Evaluation Pipeline
```python
def evaluate(model, loader, criterion, device):
    model.eval()
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device))
            y_true.append(yb.cpu().numpy())
            y_pred.append(torch.sigmoid(logits).cpu().numpy())
    
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    metrics = compute_metrics(y_true, y_pred)
    return metrics
```

## Usage Examples

### 1. Complete Preprocessing Pipeline
```bash
# Split data into train/val/test
python -c "from src.utils import make_split; make_split('data/raw_ecg')"

# Preprocess training data
python -m src.preprocessing \
    --data_dir data/raw_ecg \
    --file_list data/splits/train.txt \
    --out_path data/processed/windows_train.npz \
    --window_seconds 5 \
    --overlap 0 \
    --seizure_threshold 1 \
    --neg_to_pos 3.0 \
    --low_cut 0.5 \
    --high_cut 40.0

# Repeat for validation and test sets
```

### 2. Hierarchical Model Training
```bash
python train_hierarchical.py \
    --train_npz data/processed/windows_train.npz \
    --val_npz data/processed/windows_val.npz \
    --test_npz data/processed/windows_test.npz \
    --out_dir runs/hierarchical \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-3 \
    --n_blocks 6 \
    --out_channels 32 \
    --hidden_size 64 \
    --seq_model_type gru \
    --use_se \
    --focal_gamma 2.0 \
    --patience 10
```

### 3. Hybrid Pipeline Training
```bash
python train_hybrid_pipeline.py \
    --data_dir data/raw_ecg \
    --out_dir runs/hybrid \
    --epochs 50 \
    --batch_size 16 \
    --lr 1e-3 \
    --window_seconds 5.0 \
    --overlap 0.0 \
    --conv_channels 32 64 \
    --num_heads 4 \
    --hidden_dim 128 \
    --num_layers 2 \
    --dropout 0.1 \
    --focal_gamma 2.0
```

### 4. Model Evaluation and Inference
```python
# Load trained model
import torch
from src.models.inception import InceptionTimeSE, HierarchicalSeizureModel

model = HierarchicalSeizureModel(...)
model.load_state_dict(torch.load('runs/hierarchical/best_model.pt'))
model.eval()

# Make predictions
with torch.no_grad():
    predictions = torch.sigmoid(model(batch))
    seizure_probability = predictions.cpu().numpy()
```

## Technical Implementation Details

### 1. Memory Management
- **Gradient Accumulation**: For large batch sizes
- **Mixed Precision**: FP16 training support
- **DataLoader Optimization**: Multi-worker loading, pin memory

### 2. Device Compatibility
```python
device = torch.device("cuda" if torch.cuda.is_available() 
                     else "mps" if torch.backends.mps.is_available() 
                     else "cpu")
```

### 3. Reproducibility
```python
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### 4. Experiment Tracking
- **Metrics Logging**: CSV files with epoch-by-epoch metrics
- **Model Checkpointing**: Best model saved based on validation AUROC
- **Hyperparameter Logging**: JSON files with configuration

### 5. Error Handling and Validation
- **NaN Detection**: Checks for NaN in model outputs
- **Gradient Explosion**: Learning rate scheduling and clipping
- **Data Validation**: Shape and type checking

## Performance Considerations

### 1. Computational Complexity
- **Hierarchical Model**: O(W×T×F) where W=windows, T=time, F=features
- **Transformer Model**: O(T²×d) for self-attention with sequence length T
- **Memory Usage**: Transformer more memory-intensive than hierarchical

### 2. Training Time
- **Hierarchical**: Faster training due to fixed window sizes
- **Hybrid Pipeline**: Slower due to variable-length sequences
- **Preprocessing**: Upfront cost for hierarchical, online for hybrid

### 3. Inference Speed
- **Model Size**: InceptionTime < TCN < Transformer
- **Latency**: Hierarchical allows streaming inference
- **Throughput**: Batch processing optimized for all models

This comprehensive documentation covers all aspects of the ECG seizure detection repository, from high-level architecture to implementation details. The repository provides a complete framework for seizure detection research with multiple model architectures and training strategies.