#!/usr/bin/env python3
"""
Model Architecture Visualization and Testing Script
Demonstrates the architectures and provides basic functionality tests
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from models.inception import InceptionTimeSE, HierarchicalSeizureModel
from models.hybrid_transformer_cnn import HybridTransformerCNN
from models.tcn import TCNClassifier
from models.transformer import TransformerSequenceModel

def test_inception_model():
    """Test InceptionTimeSE model"""
    print("\n🧠 Testing InceptionTimeSE Model")
    print("=" * 50)
    
    model = InceptionTimeSE(
        n_blocks=6,
        in_channels=1,
        n_classes=1,
        out_channels=32,
        use_se=True,
        dropout=0.1
    )
    
    # Test input: batch_size=4, channels=1, time_steps=1280 (5 seconds at 256 Hz)
    batch_size, channels, time_steps = 4, 1, 1280
    x = torch.randn(batch_size, channels, time_steps)
    
    print(f"Input shape: {x.shape}")
    
    with torch.no_grad():
        output = model(x)
        embedding = model.get_embedding(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("✅ InceptionTimeSE test passed!")
    
    return model

def test_hierarchical_model():
    """Test Hierarchical Seizure Model"""
    print("\n🏗️  Testing Hierarchical Seizure Model")
    print("=" * 50)
    
    # Create window encoder
    window_encoder = InceptionTimeSE(
        n_blocks=4,
        in_channels=1,
        n_classes=1,
        out_channels=32,
        use_se=True
    )
    
    # Create hierarchical model
    model = HierarchicalSeizureModel(
        window_encoder=window_encoder,
        hidden_size=64,
        seq_model_type='gru',
        num_layers=2,
        dropout=0.1
    )
    
    # Test input: batch_size=2, sequence_length=10, channels=1, time_steps=1280
    batch_size, seq_len, channels, time_steps = 2, 10, 1, 1280
    x = torch.randn(batch_size, seq_len, channels, time_steps)
    
    print(f"Input shape: {x.shape}")
    
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("✅ Hierarchical model test passed!")
    
    return model

def test_hybrid_transformer_model():
    """Test Hybrid Transformer-CNN Model"""
    print("\n🔄 Testing Hybrid Transformer-CNN Model")
    print("=" * 50)
    
    model = HybridTransformerCNN(
        in_channels=1,
        conv_channels=(32, 64),
        num_heads=4,
        hidden_dim=128,
        num_layers=2,
        n_classes=1,
        dropout=0.1
    )
    
    # Test input: batch_size=4, channels=1, time_steps=1280
    batch_size, channels, time_steps = 4, 1, 1280
    x = torch.randn(batch_size, channels, time_steps)
    
    print(f"Input shape: {x.shape}")
    
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("✅ Hybrid Transformer-CNN test passed!")
    
    return model

def test_tcn_model():
    """Test Temporal Convolutional Network Model"""
    print("\n⚡ Testing Temporal Convolutional Network")
    print("=" * 50)
    
    model = TCNClassifier(
        in_channels=1,
        n_classes=1,
        channels=(16, 32, 32, 64),
        kernel_size=5,
        dropout=0.1
    )
    
    # Test input: batch_size=4, channels=1, time_steps=1280
    batch_size, channels, time_steps = 4, 1, 1280
    x = torch.randn(batch_size, channels, time_steps)
    
    print(f"Input shape: {x.shape}")
    
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("✅ TCN model test passed!")
    
    return model

def test_transformer_model():
    """Test Pure Transformer Model"""
    print("\n🤖 Testing Pure Transformer Model")
    print("=" * 50)
    
    # For transformer, we need embeddings as input (not raw signal)
    input_dim = 64  # Feature dimension
    model = TransformerSequenceModel(
        input_dim=input_dim,
        num_heads=4,
        hidden_dim=128,
        num_layers=2,
        n_classes=1,
        dropout=0.1
    )
    
    # Test input: batch_size=4, sequence_length=100, feature_dim=64
    batch_size, seq_len, feature_dim = 4, 100, input_dim
    x = torch.randn(batch_size, seq_len, feature_dim)
    
    print(f"Input shape: {x.shape}")
    
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("✅ Transformer model test passed!")
    
    return model

def print_architecture_summary():
    """Print comprehensive architecture summary"""
    print("\n" + "="*80)
    print("🏛️  ECG SEIZURE DETECTION ARCHITECTURE SUMMARY")
    print("="*80)
    
    print("""
📊 DATA FLOW OVERVIEW:
    Raw ECG CSV Files
           ↓
    Signal Preprocessing (Filtering, Segmentation)
           ↓
    Window Extraction (5-second windows)
           ↓
    Model Training/Inference
           ↓
    Seizure Probability Output

🧠 MODEL ARCHITECTURES:

1. HIERARCHICAL MODEL (Two-Stage)
   ┌─────────────────┐    ┌──────────────────┐    ┌─────────────┐
   │  Window Inputs  │ -> │  InceptionTime   │ -> │ Embeddings  │
   │  (B,S,C,T)     │    │  CNN Encoder     │    │  (B,S,E)    │
   └─────────────────┘    └──────────────────┘    └─────────────┘
                                                         ↓
   ┌─────────────────┐    ┌──────────────────┐    ┌─────────────┐
   │  Seizure Prob   │ <- │  Attention Pool  │ <- │ GRU/LSTM/   │
   │     (B,1)       │    │     + MLP        │    │ Transformer │
   └─────────────────┘    └──────────────────┘    └─────────────┘

2. HYBRID TRANSFORMER-CNN (End-to-End)
   ┌─────────────────┐    ┌──────────────────┐    ┌─────────────┐
   │   Raw Signal    │ -> │   CNN Feature    │ -> │  Features   │
   │    (B,C,T)      │    │   Extractor      │    │   (B,T',E)  │
   └─────────────────┘    └──────────────────┘    └─────────────┘
                                                         ↓
   ┌─────────────────┐    ┌──────────────────┐    ┌─────────────┐
   │  Seizure Prob   │ <- │  Global Pool +   │ <- │ Transformer │
   │     (B,1)       │    │     MLP          │    │  Encoder    │
   └─────────────────┘    └──────────────────┘    └─────────────┘

🎯 KEY INNOVATIONS:
• Multi-scale temporal feature extraction (InceptionTime)
• Attention-based sequence modeling
• Focal loss for class imbalance
• End-to-end and hierarchical training options
• Medical-grade evaluation metrics

📈 PERFORMANCE CHARACTERISTICS:
┌───────────────────┬─────────────┬──────────────┬─────────────────┐
│     Model Type    │ Parameters  │ Memory Usage │ Training Speed  │
├───────────────────┼─────────────┼──────────────┼─────────────────┤
│ InceptionTime     │   ~100K     │     Low      │     Fast        │
│ Hierarchical      │   ~200K     │     Low      │     Fast        │
│ Hybrid Trans-CNN  │   ~500K     │    Medium    │    Medium       │
│ TCN               │   ~150K     │     Low      │   Very Fast     │
│ Pure Transformer  │   ~300K     │     High     │     Slow        │
└───────────────────┴─────────────┴──────────────┴─────────────────┘
    """)

def main():
    """Main testing function"""
    print("🧪 ECG Seizure Detection - Model Architecture Testing")
    print("="*80)
    
    # Test all models
    models = {}
    
    try:
        models['inception'] = test_inception_model()
        models['hierarchical'] = test_hierarchical_model()
        models['hybrid'] = test_hybrid_transformer_model()
        models['tcn'] = test_tcn_model()
        models['transformer'] = test_transformer_model()
        
        print_architecture_summary()
        
        print(f"\n✅ All {len(models)} models tested successfully!")
        print("\n📝 Model parameter counts:")
        for name, model in models.items():
            param_count = sum(p.numel() for p in model.parameters())
            print(f"   {name:>12}: {param_count:>8,} parameters")
            
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        
    print("\n" + "="*80)
    print("🎉 Testing completed! Check ARCHITECTURE_DOCUMENTATION.md for details.")
    print("="*80)

if __name__ == "__main__":
    main()