# MuonSSM Language Pre-training

This guide walks you through training a MuonSSM model specifically MuonLonghorn - Longhorn with Momentum and Newton-Schulz orthogonalization from scratch.

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Understanding the Model Configuration](#understanding-the-model-configuration)
5. [Dataset Preparation](#dataset-preparation)
6. [Training the Model](#training-the-model)
7. [Monitoring Training](#monitoring-training)
8. [Troubleshooting](#troubleshooting)

---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/NVlabs/GatedDeltaNet.git
cd GatedDeltaNet
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Using conda
conda create -n muonssm python=3.11
conda activate muonssm

# Or using venv
python -m venv muonssm_env
source muonssm_env/bin/activate  # Linux/Mac
# muonssm_env\Scripts\activate  # Windows
```

### Step 3: Install PyTorch with CUDA

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 4: Install Dependencies

```bash
# Core dependencies
pip install lightning pytorch-lightning transformers einops xformers

# Install Flash Attention
pip install flash-attn --no-build-isolation

# Install Mamba SSM (required for MuonLonghorn kernels)
pip install mamba-ssm

# Install causal-conv1d
pip install causal-conv1d

# Weights & Biases for logging
pip install wandb

# Additional utilities
pip install sentencepiece protobuf
```

### Step 5: Install Apex (Optional but Recommended)

```bash
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
cd ..
```

---

## Understanding the Model Configuration

### Model Architecture Selection in `lit_gpt/config.py`

The model configurations are defined in `lit_gpt/config.py`. Here's how to set up a MuonLonghorn model:

#### Key Configuration Parameters

```python
# File: lit_gpt/config.py

@dataclass
class Config:
    # Basic model parameters
    name: str = "lit-GPT"              # Model name
    block_size: int = 4096             # Maximum sequence length
    vocab_size: int = 50254            # Vocabulary size
    n_layer: int = 16                  # Number of transformer layers
    n_head: int = 32                   # Number of attention heads
    n_embd: int = 4096                 # Embedding dimension
    
    # Model type selection (set ONE of these > 0)
    gated_delta_per_layer: int = -1    # Use GatedDeltaNet
    muon_longhorn_per_layer: int = -1  # Use MuonLonghorn (our target!)
    longhorn_per_layer: int = -1       # Use base Longhorn (no momentum)
    
    # Other important settings
    local_window: int = -1             # Local attention window (-1 = global)
    mamba_init: bool = False           # Use Mamba-style initialization
```

#### Pre-defined Model Configurations

The file contains several pre-defined configurations. Here are the MuonLonghorn ones:

```python
# ~120M parameter MuonLonghorn model
dict(
    org="NVIDIA",
    name="GatedDeltaNet_170M",
    block_size=4096, 
    vocab_size=32000,
    muon_longhorn_per_layer=1,  # Enable MuonLonghorn for ALL layers
    n_layer=10,
    n_head=12,
    n_embd=672,
    intermediate_size=2304,
    local_window=2048,
    mamba_init=True,
),

```

### Model Block Selection in `lit_gpt/model.py`

The `Block` class in `lit_gpt/model.py` handles the selection of attention mechanism:

```python
# File: lit_gpt/model.py (simplified view)

class Block(nn.Module):
    def __init__(self, config: Config, layer_idx: int) -> None:
        super().__init__()
        self.norm_1 = config.norm_class(config.n_embd, eps=config.norm_eps)
        
        # Model type selection based on config
        self.use_gated_deltanet = layer_idx % config.gated_delta_per_layer == 0 if config.gated_delta_per_layer > 0 else False
        self.use_muon_longhorn = layer_idx % config.muon_longhorn_per_layer == 0 if config.muon_longhorn_per_layer > 0 else False
        self.use_longhorn = layer_idx % config.longhorn_per_layer == 0 if config.longhorn_per_layer > 0 else False
        
        # Initialize the appropriate attention mechanism
        if self.use_gated_deltanet:
            self.attn = GatedDeltaNet(hidden_size=config.n_embd)
        elif self.use_muon_longhorn:
            # MuonLonghorn with Newton-Schulz orthogonalization
            self.attn = MuonLonghorn(
                d_model=config.n_embd,
                layer_idx=layer_idx,
                use_newton_schulz=True,  # Enable Newton-Schulz
                use_fast_path=False,
                beta=0.9,                # Momentum decay
                alpha=1.0,               # Velocity scale
            )
        elif self.use_longhorn:
            # Base Longhorn (no momentum)
            self.attn = MuonLonghorn(
                d_model=config.n_embd,
                layer_idx=layer_idx,
                use_newton_schulz=False,
                use_fast_path=False,
                beta=0.0,                # No momentum
                alpha=1.0,
            )
        else:
            # Standard self-attention
            self.attn = CausalSelfAttention(config, n_embd=config.n_embd, layer_idx=layer_idx)
```

### MuonLonghorn Parameters (in `lit_gpt/longhorn.py`)

Key parameters for MuonLonghorn:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_model` | - | Model dimension (same as `n_embd`) |
| `d_state` | 16 | SSM state dimension |
| `d_conv` | 4 | Convolution kernel size |
| `expand` | 2 | Expansion factor for inner dimension |
| `beta` | 0.9 | Momentum decay factor (0 = no momentum) |
| `alpha` | 1.0 | Velocity scale factor |
| `use_newton_schulz` | True | Enable Newton-Schulz orthogonalization |
| `ns_steps` | 1 | Number of Newton-Schulz iterations |
| `ns_mode` | 'compile' | 'compile' or 'triton' for NS implementation |

---

## Dataset Preparation

### Option 1: Download FineWeb-Edu Dataset (Recommended)

The FineWeb-Edu dataset is a high-quality educational web content dataset.

```bash
# Create data directory
mkdir -p data/fineweb_edu

# Install datasets library
pip install datasets

# Create a download script (save as download_fineweb.py)
cat > download_fineweb.py << 'EOF'
from datasets import load_dataset
import os

# Download FineWeb-Edu (10B token split)
dataset = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    name="sample-10BT",
    split="train",
    cache_dir="./data/fineweb_edu_cache"
)

# Save to disk
dataset.save_to_disk("./data/fineweb_edu_raw")
print(f"Downloaded {len(dataset)} samples")
EOF

python download_fineweb.py
```

### Option 2: Use SlimPajama Dataset

```bash
# Download SlimPajama (smaller sample for testing)
cat > download_slimpajama.py << 'EOF'
from datasets import load_dataset

# Download SlimPajama sample
dataset = load_dataset(
    "cerebras/SlimPajama-627B",
    split="train[:1%]",  # 1% sample for testing
    cache_dir="./data/slimpajama_cache"
)

dataset.save_to_disk("./data/slimpajama_raw")
print(f"Downloaded {len(dataset)} samples")
EOF

python download_slimpajama.py
```

### Step 2: Tokenize and Prepare Data

The training script expects data in a specific packed format. Create the preparation script:

```bash
cat > prepare_data.py << 'EOF'
"""
Prepare dataset for GatedDeltaNet training.
Tokenizes and packs data into binary format.
"""
import os
import numpy as np
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoTokenizer
from tqdm import tqdm

# Configuration
DATA_DIR = "./data/fineweb_edu_raw"  # Change if using different dataset
OUTPUT_DIR = "./data/fineweb_edu_10BT_split"
BLOCK_SIZE = 4097  # 4096 + 1 for next token prediction
TOKENIZER_NAME = "TinyLlama/TinyLlama_v1.1"

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

# Load dataset
print("Loading dataset...")
dataset = load_from_disk(DATA_DIR)

# Create output directories
train_dir = Path(OUTPUT_DIR) / "train"
val_dir = Path(OUTPUT_DIR) / "validation"
train_dir.mkdir(parents=True, exist_ok=True)
val_dir.mkdir(parents=True, exist_ok=True)

# Split dataset (95% train, 5% validation)
split_idx = int(len(dataset) * 0.95)
train_data = dataset.select(range(split_idx))
val_data = dataset.select(range(split_idx, len(dataset)))

def tokenize_and_pack(data, output_dir, prefix, chunk_size=100000):
    """Tokenize data and save as packed binary files."""
    all_tokens = []
    file_idx = 0
    
    for i, example in tqdm(enumerate(data), total=len(data), desc=f"Processing {prefix}"):
        # Get text content
        text = example.get("text", example.get("content", ""))
        if not text:
            continue
            
        # Tokenize
        tokens = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(tokens)
        
        # Save when we have enough tokens
        while len(all_tokens) >= chunk_size * BLOCK_SIZE:
            # Extract chunk
            chunk_tokens = all_tokens[:chunk_size * BLOCK_SIZE]
            all_tokens = all_tokens[chunk_size * BLOCK_SIZE:]
            
            # Reshape and save
            chunk_array = np.array(chunk_tokens, dtype=np.uint16).reshape(-1, BLOCK_SIZE)
            output_path = output_dir / f"{prefix}_{file_idx:05d}.bin"
            chunk_array.tofile(output_path)
            print(f"Saved {output_path} with shape {chunk_array.shape}")
            file_idx += 1
    
    # Save remaining tokens
    if len(all_tokens) >= BLOCK_SIZE:
        n_complete = (len(all_tokens) // BLOCK_SIZE) * BLOCK_SIZE
        chunk_tokens = all_tokens[:n_complete]
        chunk_array = np.array(chunk_tokens, dtype=np.uint16).reshape(-1, BLOCK_SIZE)
        output_path = output_dir / f"{prefix}_{file_idx:05d}.bin"
        chunk_array.tofile(output_path)
        print(f"Saved {output_path} with shape {chunk_array.shape}")

# Process train and validation data
print("\nProcessing training data...")
tokenize_and_pack(train_data, train_dir, "train_slim")

print("\nProcessing validation data...")
tokenize_and_pack(val_data, val_dir, "validation", chunk_size=10000)

print("\nData preparation complete!")
print(f"Training data: {train_dir}")
print(f"Validation data: {val_dir}")
EOF

python prepare_data.py
```

### Directory Structure After Preparation

```
data/
└── fineweb_edu_10BT_split/
    ├── train/
    │   ├── train_slim_00000.bin
    │   ├── train_slim_00001.bin
    │   └── ...
    └── validation/
        ├── validation_00000.bin
        └── ...
```

---

## Training the Model

### Step 1: Create Training Script

Create a shell script for training:

```bash
cat > run_training.sh << 'EOF'
#!/bin/bash

# ============================================
# MuonLonghorn Training Script
# ============================================

# Paths - MODIFY THESE FOR YOUR SETUP
PROJECT_ROOT="/path/to/GatedDeltaNet"  # Change this!
TRAIN_DATA="${PROJECT_ROOT}/data/fineweb_edu_10BT_split/train"
VALIDATION_DATA="${PROJECT_ROOT}/data/fineweb_edu_10BT_split/validation"
SAVE_DIR="${PROJECT_ROOT}/muon_longhorn_results"

# Model Configuration
MODEL='GatedDeltaNet_120M'        # Model name from config.py
# MODEL='GatedDeltaNet_340M'      # Larger model
# MODEL='GatedDeltaNet_H1_0.4B'   # Even larger model

# Training Configuration
CONFIG='tsz512x4k_10B'            # Training token config
# Options: tsz512x4k_10B, tsz512x4k_15B, tsz512x4k_20B, etc.
# Format: tsz{batch_size}x{seq_len}_{total_tokens}

NAME="muonlonghorn_${MODEL}"      # Experiment name
LR=1e-3                           # Learning rate
MICRO_BATCH_SIZE=2                # Batch size per GPU (reduce if OOM)
EVAL_ITERS=25                     # Validation iterations

# Triton Cache (for faster kernel compilation)
export TRITON_CACHE_DIR="${SAVE_DIR}/triton_cache/"
mkdir -p ${TRITON_CACHE_DIR}

# CUDA Libraries (may need adjustment for your setup)
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Run Training
python ${PROJECT_ROOT}/pretrain.py \
    --train_data_dir ${TRAIN_DATA} \
    --val_data_dir ${VALIDATION_DATA} \
    --output_root ${SAVE_DIR} \
    --exp_name ${NAME} \
    --model_name ${MODEL} \
    --train_config ${CONFIG} \
    --eval_iters ${EVAL_ITERS} \
    --learning_rate ${LR} \
    --micro_batch_size ${MICRO_BATCH_SIZE} \
    --interactive_job

# Add --debug to disable wandb logging
# Add --resume to resume from checkpoint
EOF

chmod +x run_training.sh
```

### Step 2: Run Training

```bash
# Make sure you're in the GatedDeltaNet directory
cd /path/to/GatedDeltaNet

# Run training
./run_training.sh
```

### Training Arguments Explained

| Argument | Description | Example |
|----------|-------------|---------|
| `--train_data_dir` | Path to training data | `./data/train` |
| `--val_data_dir` | Path to validation data | `./data/validation` |
| `--output_root` | Where to save checkpoints | `./results` |
| `--exp_name` | Experiment name (for logging) | `muon_exp1` |
| `--model_name` | Model config name from config.py | `GatedDeltaNet_120M` |
| `--train_config` | Training config (tokens/batch) | `tsz512x4k_10B` |
| `--learning_rate` | Learning rate | `1e-3` |
| `--micro_batch_size` | Batch size per GPU | `2` |
| `--eval_iters` | Validation iterations | `25` |
| `--interactive_job` | For single-node training | - |
| `--debug` | Disable wandb logging | - |
| `--resume` | Resume from checkpoint | - |

### Training Configuration Naming Convention

The `train_config` follows this pattern: `tsz{batch}x{seq}_{tokens}`

| Config | Global Batch | Seq Length | Total Tokens |
|--------|--------------|------------|--------------|
| `tsz512x4k_10B` | 512 | 4096 | 10 Billion |
| `tsz512x4k_15B` | 512 | 4096 | 15 Billion |
| `tsz512x4k_20B` | 512 | 4096 | 20 Billion |
| `tsz1024x4k_20B` | 1024 | 4096 | 20 Billion |

---

## Monitoring Training

### Using Weights & Biases (Recommended)

1. Sign up at [wandb.ai](https://wandb.ai)
2. Login: `wandb login`
3. Training metrics will be automatically logged

### Console Output

During training, you'll see output like:

```
iter 100 step 25: loss 8.2341, iter time: 245.32ms remaining time: 2.5 hours
  total training throughput 12.5K tokens/s per GPU
  peak memory allocate 18.5 GB
```

### Checkpoints

Checkpoints are saved to `{output_root}/outputs/{exp_name}/`:
- `latest-model-ckpt.pth` - Latest checkpoint (for resuming)
- `final-model-ckpt.pth` - Final model after training

---

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)

```bash
# Reduce micro_batch_size
--micro_batch_size 1

# Or use gradient accumulation (automatic based on global batch size)
```

#### 2. CUDA/Triton Errors

```bash
# Clear Triton cache
rm -rf ~/.triton/cache/*
rm -rf ${SAVE_DIR}/triton_cache/*

# Set LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

#### 3. Import Errors

```bash
# Make sure all dependencies are installed
pip install mamba-ssm causal-conv1d flash-attn --no-build-isolation

# Install from source if pip fails
git clone https://github.com/Dao-AILab/causal-conv1d.git
cd causal-conv1d && pip install . && cd ..
```

#### 4. Data Loading Issues

- Ensure data files are named correctly (`train_slim_*.bin`, `validation_*.bin`)
- Check file permissions
- Verify data format matches expected dtype (uint16)

### Performance Tips

1. **Use BFloat16**: Automatically enabled via `precision="bf16-mixed"`
2. **Enable Flash Attention**: Automatically used when available
3. **Triton Caching**: Set `TRITON_CACHE_DIR` to avoid recompilation
4. **Multi-GPU**: Script automatically uses all available GPUs via FSDP

---

## Quick Start Summary

```bash
# 1. Clone and setup
git clone https://github.com/NVlabs/GatedDeltaNet.git
cd GatedDeltaNet
pip install torch lightning transformers mamba-ssm causal-conv1d flash-attn einops xformers wandb

# 2. Prepare data (assumes you have tokenized .bin files)
mkdir -p data/my_dataset/{train,validation}
# Copy your train_slim_*.bin files to data/my_dataset/train/
# Copy your validation_*.bin files to data/my_dataset/validation/

# 3. Train
python pretrain.py \
    --train_data_dir ./data/my_dataset/train \
    --val_data_dir ./data/my_dataset/validation \
    --output_root ./results \
    --exp_name my_muonlonghorn \
    --model_name GatedDeltaNet_120M \
    --train_config tsz512x4k_10B \
    --learning_rate 1e-3 \
    --micro_batch_size 2 \
    --interactive_job \
    --debug  # Remove this to enable wandb logging
```

---

## References

- [Gated Delta Networks Paper](https://arxiv.org/abs/2412.06464)
- [MuonLonghorn Implementation](lit_gpt/longhorn.py)
- [Original Samba Repository](https://github.com/microsoft/Samba)
- [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention)

---