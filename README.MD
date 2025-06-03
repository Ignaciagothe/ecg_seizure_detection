

# ECG Seizure Detection Project Structure

This project is structured to facilitate the development and testing of ECG seizure detection models using PyTorch. The following directory structure outlines the organization of the codebase:

```tree
    ecg_seizure/
    │
    ├── data/                  # (ignored by git) raw & processed.npz
    │
    ├── src/          
    │   ├── __init__.py
    │   ├── config.py           # dataclass + Hydra/YAML loader optional
    │   ├── preprocessing.py    # segment_trace(), preprocess_dataset()
    │   ├── datasets.py         # ECGWindowDataset, collate_fn
    │   ├── models/
    │   │   ├── __init__.py
    │   │   ├── inception_se.py  
    │   │   └── patchtst.py      
    │   └── engine.py           # train_one_epoch(), evaluate()
    │
    ├── train.py                # 20-line CLI entry point
    ├── requirements.txt
    └── tests/
        └── test_segment.py
    ```




"""
1.
    cd /Users/ignaciagothe/Ignacia\ macbook/Proyecto\ Ecg\ ipre/ecg_seizure

    python train.py \
    --data_dir data/raw \
    --npz_path data/processed/windows_train.npz \
    --test_npz_path data/processed/windows_test.npz \
    --out_dir runs \
    --model inception \
    --epochs 50 \
    --batch_size 128 \
    --lr 0.0005 \
    --n_blocks 6 \
    --out_channels 32 \
    --bottleneck_channels 32 \
    --kernel_sizes 10 20 40 \
    --focal_gamma 2.0


2.
    # TRAIN split
    python compute_embeddings.py \
    --model_path runs/20250601_104512/best_model_inception.pt \
    --npz_path data/processed/windows_train.npz \
    --out_dir runs/20250601_104512/embeddings_train \
    --batch_size 256 \
    --device cuda

    # VAL split
    python compute_embeddings.py \
    --model_path runs/20250601_104512/best_model_inception.pt \
    --npz_path data/processed/windows_val.npz \
    --out_dir runs/20250601_104512/embeddings_val \
    --batch_size 256 \
    --device cuda

    # TEST split
    python compute_embeddings.py \
    --model_path runs/20250601_104512/best_model_inception.pt \
    --npz_path data/processed/windows_test.npz \
    --out_dir runs/20250601_104512/embeddings_test \
    --batch_size 256 \
    --device cuda


3. 
python train_gru.py \
  --train_emb_dir runs/20250601_104512/embeddings_train \
  --val_emb_dir   runs/20250601_104512/embeddings_val \
  --test_emb_dir  runs/20250601_104512/embeddings_test \
  --seq_len 6 \
  --hidden_size 64 \
  --num_layers 1 \
  --batch_size 32 \
  --epochs 20 \
  --lr 0.001 \
  --patience 5 \
  --device cuda \
  --out_dir runs/20250601_104512/gru_run


4. for inferenece

        import torch
        import numpy as np
        from pathlib import Path
        from train_gru import EmbeddingSequenceDataset, GRUSequenceModel

 # 1) Load saved embedding (.npy)

        test_embs = np.load("runs/.../embeddings_test/embeddings.npy")  # shape (N_test_windows, D)
        test_labels = np.load("runs/.../embeddings_test/labels.npy")    # shape (N_test_windows,)

        seq_len = 6  # must match what you trained with
        D = test_embs.shape[1]
        usable = (test_embs.shape[0] // seq_len) * seq_len
        test_embs = test_embs[:usable].reshape(-1, seq_len, D)  # (N_seqs, seq_len, D)
        test_labels = test_labels[:usable].reshape(-1, seq_len)  # (N_seqs, seq_len)

 # 2) Build model, load weights
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = GRUSequenceModel(input_dim=D, hidden_size=64, num_layers=1).to(device)
        model.load_state_dict(torch.load("runs/.../best_gru.pt", map_location=device))
        model.eval()

 # 3) Iterate & predict
 
        all_logits = []
        with torch.no_grad():
            for seq in torch.from_numpy(test_embs).float().to(device):
                seq = seq.unsqueeze(0)  # [1, seq_len, D]
                logits = model(seq)     # [1, seq_len]
                all_logits.append(logits.cpu().numpy())
        all_logits = np.concatenate(all_logits, axis=0).reshape(-1)  # (usable,)
        # all_labels = test_labels.reshape(-1)  # (usable,)
        preds = (all_logits > 0).astype(int)
        # …compute metrics on preds vs. all_labels…
"""

"""
Example workflow after reorganising the project structure
---------------------------------------------------------

1) Generate file lists for an 80/20 split and store them in *data/splits/*

    ls data/raw | sort | head -n -10 > data/splits/train_list.txt
    ls data/raw | sort | tail -n 10  > data/splits/test_list.txt



2) Pre‑process each split to NPZ files in *data/processed/*

    python -m src.preprocessing \
        --data_dir data/raw \
        --file_list data/splits/train_list.txt \
        --out_path data/processed/windows_train.npz \
        --window_seconds 5 --overlap 0.5 --neg_to_pos 3



    python -m src.preprocessing \
        --data_dir data/raw \
        --file_list data/splits/test_list.txt \
        --out_path data/processed/windows_test.npz \
        --window_seconds 5 --overlap 0.5 --neg_to_pos 3

OR

2)  
python -m src.preprocessing \
    --data_dir data/raw \
    --out_path data/processed/windows_15s.npz \
    --window_seconds 15 \
    --overlap 0.5 \
    --neg_to_pos 3

3) Train with separate test set, storing artefacts in *runs/*

    python train.py \
        --data_dir data/raw \
        --npz_path data/processed/windows_train.npz \
        --test_npz_path data/processed/windows_test.npz \
        --out_dir runs \
        --model inception \
        --epochs 40 \
        --batch_size 128 \
        --lr 0.001

---
        

python train.py \
  --data_dir data/raw \
  --npz_path data/processed/windows_train.npz \
  --test_npz_path data/processed/windows_test.npz \
  --out_dir runs \
  --model inception \
  --epochs 50 \
  --batch_size 128 \
  --lr 0.0005 \
  --n_blocks 6 \
  --out_channels 32 \
  --bottleneck_channels 32 \
  --kernel_sizes 10 20 40 

"""
