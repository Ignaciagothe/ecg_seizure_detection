# ECG Seizure Detection Repository - Executive Summary

## Project Overview

This repository implements state-of-the-art deep learning models for **epileptic seizure detection from single-lead ECG signals**. The system provides two complementary approaches for processing ECG data and detecting seizure events with high accuracy.

## Key Algorithms and Models

### 1. **Hierarchical Model Architecture**
- **InceptionTime CNN**: Multi-scale temporal feature extraction using inception blocks with kernel sizes [9, 19, 39]
- **Sequential Modeling**: GRU/LSTM/Transformer processes sequences of window embeddings
- **Attention Pooling**: Weighted aggregation of temporal features for final classification
- **Two-Stage Training**: Separate window encoding and sequence modeling phases

### 2. **Hybrid Transformer-CNN Pipeline** 
- **CNN Feature Extractor**: Local pattern recognition with convolutional layers
- **Positional Encoding**: Sinusoidal temporal position embeddings
- **Multi-Head Self-Attention**: Captures long-range dependencies in ECG signals
- **Transformer Encoder**: Multiple layers of attention + feed-forward networks
- **End-to-End Processing**: Direct training from raw CSV files to seizure predictions

### 3. **Additional Model Variants**
- **Temporal Convolutional Network (TCN)**: Dilated causal convolutions for sequential modeling
- **Pure Transformer**: Direct sequence-to-sequence attention modeling

## Signal Processing Pipeline

### Preprocessing Algorithms
1. **Bandpass Filtering**: Butterworth filter (0.5-40 Hz) removes baseline drift and noise
2. **Optional Notch Filter**: Removes 50/60 Hz powerline interference
3. **Segmentation**: Sliding window extraction (typically 5-second windows)
4. **Label Assignment**: Window-level seizure labeling based on sample thresholds
5. **Class Balancing**: Negative-to-positive ratio control (default 3:1)

### Data Splits
- **File-based**: Complete ECG recordings in train/val/test
- **Patient-based**: Patient-stratified splits prevent data leakage
- **Session-based**: Recording session-level splitting

## Training Algorithms

### Loss Functions
- **Focal Loss**: Addresses class imbalance with focusing parameter γ and weighting α
  - Formula: `FL(p_t) = -α_t(1-p_t)^γ log(p_t)`
- **Weighted Binary Cross-Entropy**: Class-balanced loss with positive class weighting

### Optimization Strategies
- **Adam Optimizer**: Adaptive learning rate optimization
- **ReduceLROnPlateau**: Learning rate scheduling based on validation metrics
- **Early Stopping**: Prevents overfitting using validation AUROC monitoring
- **Gradient Clipping**: Prevents gradient explosion in RNN/Transformer models

### Regularization Techniques
- **Dropout**: Applied to embeddings, attention weights, and classifier layers
- **Batch Normalization**: Stabilizes training in CNN layers
- **Squeeze-and-Excitation**: Channel attention mechanism for feature selection

## Evaluation Metrics

The system provides comprehensive evaluation using medical-grade metrics:

1. **AUROC (Area Under ROC Curve)**: Ranking quality across all thresholds
2. **AUPRC (Area Under Precision-Recall Curve)**: Critical for imbalanced medical data
3. **F1-Score**: Harmonic mean of precision and recall
4. **Precision**: Reduces false alarm rate (important for clinical deployment)
5. **Recall/Sensitivity**: Ensures seizure detection (critical for patient safety)
6. **Confusion Matrix**: Detailed breakdown of TP, TN, FP, FN

## Technical Innovations

### Attention Mechanisms
- **Hierarchical Attention**: Weighted pooling of sequence embeddings
- **Multi-Head Self-Attention**: Parallel attention heads capture different temporal patterns
- **Attention Denoising**: Learned noise reduction using attention weights

### Memory Efficiency
- **Window-Based Processing**: Fixed-size windows reduce memory requirements
- **Gradient Accumulation**: Enables large effective batch sizes
- **Mixed Precision Training**: FP16 support for GPU acceleration

### Model Interpretability
- **Attention Weights**: Visualize which time regions the model focuses on
- **Feature Embeddings**: Analyze learned representations for different seizure types
- **Layer-wise Analysis**: Understanding hierarchical feature learning

## Usage Scenarios

### Scenario 1: Research and Experimentation
- Use **hierarchical model** with preprocessed windows
- Experiment with different sequence models (GRU/LSTM/Transformer)
- Analyze attention patterns and feature representations

### Scenario 2: Production Deployment  
- Use **hybrid pipeline** for end-to-end processing
- Direct deployment from raw ECG CSV files
- Minimal preprocessing requirements

### Scenario 3: Real-time Streaming
- Deploy **hierarchical model** with sliding window inference
- Process 5-second windows in real-time
- Low-latency seizure detection

## Performance Characteristics

### Model Comparison
| Model | Training Speed | Memory Usage | Inference Speed | Accuracy |
|-------|---------------|--------------|-----------------|----------|
| Hierarchical | Fast | Low | Fast | High |
| Hybrid Pipeline | Medium | High | Medium | High |
| TCN | Fast | Low | Very Fast | Medium |
| Pure Transformer | Slow | Very High | Medium | High |

### Computational Requirements
- **GPU Memory**: 4-16 GB depending on model and batch size
- **Training Time**: 2-8 hours on modern GPUs
- **Inference Speed**: 1-10ms per window on GPU

## Repository Architecture

```
📦 ECG Seizure Detection
├── 🧠 Models (src/models/)
│   ├── InceptionTime + Hierarchical
│   ├── Hybrid Transformer-CNN  
│   ├── Temporal Convolutional Network
│   └── Pure Transformer
├── 🔧 Preprocessing (src/)
│   ├── Signal filtering & segmentation
│   ├── Data splits & balancing
│   └── Feature extraction
├── 🚀 Training Scripts
│   ├── Hierarchical training pipeline
│   └── End-to-end hybrid pipeline
├── 📊 Evaluation & Utils
│   ├── Comprehensive metrics
│   ├── Model checkpointing
│   └── Experiment tracking
└── 📓 Analysis Notebooks
    ├── Signal exploration
    └── Training analysis
```

## Scientific Foundation

The repository implements methods based on cutting-edge research:
- **InceptionTime**: State-of-the-art time series classification
- **Transformer Architecture**: Attention mechanisms for temporal modeling
- **Focal Loss**: Advanced handling of class imbalance
- **Multi-scale Feature Learning**: Capture patterns at different temporal scales
- **Medical Signal Processing**: Domain-specific preprocessing for ECG analysis

## Clinical Relevance

### Medical Applications
- **Epilepsy Monitoring**: Continuous seizure detection in clinical settings
- **Home Monitoring**: Wearable device integration for ambulatory patients
- **Emergency Response**: Automated seizure alerts for caregivers
- **Research Studies**: Large-scale seizure pattern analysis

### Validation Considerations
- **Cross-Patient Generalization**: Models tested across different patients
- **Temporal Robustness**: Performance across different recording sessions
- **Artifact Handling**: Robust to movement artifacts and noise
- **Clinical Validation**: Metrics align with medical evaluation standards

This repository provides a complete, production-ready framework for ECG-based seizure detection with multiple architectural options, comprehensive evaluation, and clinical applicability.