# Vision Transformer (ViT) From Scratch

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen)](https://python.org)
[![Paper](https://img.shields.io/badge/Paper-An%20Image%20is%20Worth%2016×16%20Words-blue)](https://arxiv.org/abs/2010.11929)
[![Dataset](https://img.shields.io/badge/Dataset-MNIST-orange)](http://yann.lecun.com/exdb/mnist/)

A pure PyTorch implementation of the **Vision Transformer (ViT)** built entirely from scratch — no `timm`, no HuggingFace, no high-level wrappers. Every component is implemented manually and the full model is trained on **MNIST** for image classification.

Based on the paper: **"An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale"** (Dosovitskiy et al., ICLR 2021).

---

## Architecture Overview

```
Input Image (1×28×28)
        │
        ▼
┌─────────────────────┐
│   Patch Embedding   │  Conv2d(1, 32, kernel=7, stride=7) → flatten → (16, 32)
└────────┬────────────┘
         │
         ▼
  Prepend [CLS] token  →  (17, 32)
         │
         ▼
  + Learnable Positional Embedding  →  (17, 32)
         │
         ▼
┌─────────────────────────────────┐
│  Transformer Encoder × 4 layers │
│                                 │
│  ┌─────────────────────────┐    │
│  │   LayerNorm             │    │
│  │   Multi-Head Attention  │    │  4 heads, head_dim = 8
│  │   Residual connection   │    │
│  ├─────────────────────────┤    │
│  │   LayerNorm             │    │
│  │   MLP (GELU, ×4 hidden) │    │  hidden_dim = 128
│  │   Residual connection   │    │
│  └─────────────────────────┘    │
└────────────────┬────────────────┘
                 │
         Extract [CLS] token  →  (32,)
                 │
                 ▼
┌──────────────────────┐
│  MLP Classification  │  LayerNorm → Linear(32, 10)
│  Head                │
└──────────┬───────────┘
           │
           ▼
    Class Logits (10)
```

---

## Model Configuration

| Hyperparameter | Value |
|---|---|
| Image size | 28 × 28 (MNIST) |
| Patch size | 7 × 7 |
| Number of patches | 16 (4 × 4 grid) |
| Token dimension (`d_model`) | 32 |
| Number of encoder layers | 4 |
| Number of attention heads | 4 |
| MLP hidden dimension | 128 (4 × token_dim) |
| Number of classes | 10 |
| Positional embedding | Learnable |
| Classification token | Learnable `[CLS]` token |

---

## Components

### `PatchEmbedding.py` — Image to Patch Tokens

Splits the input image into fixed-size patches and projects each patch into a `token_dim`-dimensional embedding vector. This is implemented using a single `nn.Conv2d` layer where `kernel_size = stride = patch_size`, which achieves the same effect as slicing and linearly projecting non-overlapping patches:

```
Input:  (B, 1, 28, 28)
Conv2d: kernel=7, stride=7, out_channels=32
Output: (B, 32, 4, 4)  →  flatten(2)  →  transpose(1, 2)  →  (B, 16, 32)
```

---

### `MultiHeadAttention.py` — Scaled Dot-Product Multi-Head Attention

Standard multi-head self-attention used inside each Transformer encoder block:

1. Project input into Q, K, V via separate linear layers
2. Reshape into `(B, num_heads, seq_len, head_dim)`
3. Compute `softmax(QKᵀ / sqrt(head_dim)) · V`
4. Concatenate heads and project through `fc_out`

Optional mask support for padding.

---

### `PostionalEncoding.py` — Sinusoidal Positional Encoding (reference)

A sinusoidal positional encoding module. Note: the main ViT model uses **learnable** positional embeddings (`nn.Parameter`) rather than fixed sinusoidal ones, which is the approach used in the original ViT paper. This module is kept as a reference implementation.

---

### `TransformerEncoder.py` — Transformer Encoder Block

A single encoder block following the **Pre-LN** (Pre-Layer Normalization) variant:

```
x = x + MultiHeadAttention(LayerNorm(x))
x = x + MLP(LayerNorm(x))
```

The MLP uses `GELU` activation with a hidden dimension of `4 × token_dim`. Pre-LN places normalization before each sub-layer, which tends to train more stably than the Post-LN variant used in the original Transformer paper.

---

### `Classification.py` — MLP Classification Head

Takes the `[CLS]` token output (index 0 of the sequence dimension) and maps it to class logits:

```
x → LayerNorm → Linear(token_dim, num_classes) → logits
```

---

### `VisionTransformer.py` — Full ViT Model

The top-level module that assembles all components:

1. **Patch Embedding** — convert image to patch token sequence
2. **[CLS] token** — prepend a learnable classification token
3. **Positional Embedding** — add learnable position information to all tokens (patches + CLS)
4. **Transformer Encoder** — 4 stacked encoder blocks via `nn.Sequential`
5. **CLS extraction** — take the first token `x[:, 0]`
6. **MLP Head** — classify into 10 digit classes

---

### `ViT_Experiment.ipynb` — Training & Visualization

A Jupyter notebook covering:
- Loading and preprocessing MNIST
- Instantiating and training the ViT model
- Tracking training and validation accuracy
- Visualizing results

---

## Project Structure

```
VisionTransformer/
├── PatchEmbedding.py       # Conv2d-based patch tokenization
├── MultiHeadAttention.py   # Scaled dot-product multi-head attention
├── PostionalEncoding.py    # Sinusoidal positional encoding (reference)
├── TransformerEncoder.py   # Pre-LN encoder block (Attention + MLP + residuals)
├── Classification.py       # MLP head: LayerNorm + Linear
├── VisionTransformer.py    # Full ViT: patch embed + CLS + pos embed + encoder + head
├── ViT_Experiment.ipynb    # Training loop and evaluation on MNIST
└── README.md
```

---

## Usage

```bash
pip install torch torchvision jupyter
```

```python
import torch
from VisionTransformer import VisionTransformer

model = VisionTransformer()

# Input: batch of grayscale 28×28 images
x = torch.randn(64, 1, 28, 28)

logits = model(x)
print(logits.shape)  # (64, 10)

# Get predicted class
preds = logits.argmax(dim=-1)
print(preds.shape)   # (64,)
```

For the full training experiment, open and run `ViT_Experiment.ipynb`.

---

## Design Decisions

| Decision | Detail |
|---|---|
| Patch embedding method | `nn.Conv2d` with `kernel_size = stride = patch_size` (equivalent to linear projection of patches) |
| Positional embedding | Learnable `nn.Parameter` (ViT paper default, not sinusoidal) |
| Normalization order | Pre-LN (normalize before attention/MLP, not after) |
| MLP activation | GELU (standard in ViT) |
| Classification | `[CLS]` token extracted at sequence position 0 |
| Dataset | MNIST — chosen for fast iteration and CPU-friendly training |

---

## Reference

> Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N. (2020).  
> **An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale.** *ICLR 2021.*  
> https://arxiv.org/abs/2010.11929
