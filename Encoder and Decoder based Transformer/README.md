# Encoder–Decoder Transformer From Scratch

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen)](https://python.org)
[![Paper](https://img.shields.io/badge/Paper-Attention%20Is%20All%20You%20Need-blue)](https://arxiv.org/abs/1706.03762)

A complete, from-scratch PyTorch implementation of the original Transformer architecture introduced in **"Attention Is All You Need"** (Vaswani et al., NeurIPS 2017). Every component — from sinusoidal positional encoding to the full encoder–decoder stack — is implemented manually without relying on `nn.Transformer` or any high-level abstraction.

---

## Architecture Overview

```
Input Tokens                          Target Tokens
     │                                      │
  Embedding + sqrt(d_model)            Embedding + sqrt(d_model)
     │                                      │
  Positional Encoding               Positional Encoding
     │                                      │
┌────▼────────────────────┐    ┌─────▼──────────────────────────┐
│    Encoder (×N layers)  │    │      Decoder (×N layers)        │
│                         │    │                                  │
│  ┌─────────────────┐    │    │  ┌──────────────────────────┐   │
│  │ Multi-Head      │    │    │  │ Masked Multi-Head        │   │
│  │ Self-Attention  │    │    │  │ Self-Attention (causal)  │   │
│  └────────┬────────┘    │    │  └─────────────┬────────────┘   │
│  Add & LayerNorm        │    │  Add & LayerNorm               │
│  ┌─────────────────┐    │    │  ┌──────────────────────────┐   │
│  │ Feed-Forward    │    │    │  │ Cross-Attention          │◄──┼── encoder output
│  │ Network         │    │    │  │ (Q from decoder,         │   │
│  └────────┬────────┘    │    │  │  K/V from encoder)       │   │
│  Add & LayerNorm        │    │  └─────────────┬────────────┘   │
└────────────┼────────────┘    │  Add & LayerNorm               │
             │                 │  ┌──────────────────────────┐   │
             └────────────────►│  │ Feed-Forward Network     │   │
                               │  └─────────────┬────────────┘   │
                               │  Add & LayerNorm               │
                               └──────────────────┬─────────────┘
                                                  │
                                          Linear Projection
                                                  │
                                            Logits (vocab)
```

---

## Components

### `PostionalEncoding.py` — Sinusoidal Positional Encoding

Implements the fixed sinusoidal encoding from the paper. Even indices use `sin`, odd indices use `cos`, with frequencies defined by:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

The encoding is precomputed up to `max_seq_len=5000` and stored as a non-trainable buffer. Token embeddings are scaled by `sqrt(d_model)` before the positional encoding is added, matching the paper exactly.

---

### `FeedForward.py` — Position-wise Feed-Forward Network

A two-layer MLP applied independently to each position:

```
FFN(x) = ReLU(x W₁ + b₁) W₂ + b₂
```

The inner dimension `d_ff` defaults to `4 × d_model` (e.g., 2048 when `d_model=512`), consistent with the original paper.

---

### `multi_head_attention.py` — Multi-Head Self-Attention

Implements scaled dot-product attention with multiple parallel heads:

1. Project input into Q, K, V via separate `nn.Linear` layers
2. Split into `num_heads` heads along the embedding dimension
3. Compute scaled dot-product attention: `softmax(QKᵀ / sqrt(head_dim)) · V`
4. Concatenate heads and project through `fc_out`

Supports an optional boolean padding mask (`True` = keep, `False` = mask).

---

### `MaskedMultiHeadAttention.py` — Causal (Masked) Self-Attention

Extends multi-head attention with an autoregressive causal mask. A precomputed upper-triangular boolean mask (registered as a buffer) blocks each position from attending to future positions. Both the causal mask and an optional padding mask are applied simultaneously to the attention scores before softmax.

---

### `CrossMultiHeadAttention.py` — Cross-Attention

Implements decoder-to-encoder attention. Queries come from the decoder's current hidden state, while Keys and Values come from the encoder output:

- `Q` ← decoder hidden state (`tgt_len`)
- `K`, `V` ← encoder output (`src_len`)

Supports a source-side padding mask to prevent attention over encoder padding tokens.

---

### `Encoder.py` — Transformer Encoder

**`TransformerEncoderBlock`**: A single encoder layer consisting of:
- Multi-head self-attention → residual + LayerNorm
- Feed-forward network → residual + LayerNorm
- Dropout applied to both sub-layers

**`TransformerEncoder`**: Stacks `num_layers` encoder blocks followed by a final `LayerNorm`.

---

### `Decoder.py` — Transformer Decoder

**`DecoderLayer`**: A single decoder layer consisting of:
1. Masked multi-head self-attention (causal + padding mask) → residual + LayerNorm
2. Cross-attention over encoder output (source padding mask) → residual + LayerNorm
3. Feed-forward network → residual + LayerNorm

**`TransformerDecoder`**: Stacks `num_layers` decoder blocks followed by a final `LayerNorm`.

---

### `Transformer.py` — Full Encoder–Decoder Transformer

The top-level module that wires everything together:

- Separate source and target token embeddings (`nn.Embedding` with `padding_idx=0`)
- Shared sinusoidal positional encoding applied to both
- Stacked encoder and decoder
- Final linear projection to vocabulary logits
- **Weight tying**: the target embedding matrix is shared with the output projection (`out_proj.weight = tgt_embedding.weight`), as recommended in the paper

---

## Project Structure

```
Encoder and Decoder based Transformer/
├── PostionalEncoding.py          # Sinusoidal positional encoding
├── FeedForward.py                # Position-wise FFN (ReLU, d_ff = 4×d_model)
├── multi_head_attention.py       # Multi-head self-attention
├── MaskedMultiHeadAttention.py   # Causal + padding masked self-attention
├── CrossMultiHeadAttention.py    # Decoder-to-encoder cross-attention
├── Encoder.py                    # Encoder block + stacked encoder
├── Decoder.py                    # Decoder layer + stacked decoder
├── Transformer.py                # Full seq2seq Transformer
└── README.md
```

---

## Usage

```bash
pip install torch
```

```python
import torch
from Transformer import Transformer

model = Transformer(
    src_vocab_size=10000,
    tgt_vocab_size=10000,
    embed_size=512,
    num_heads=8,
    num_encoder_layer=6,
    num_decoder_layer=6,
    d_ff=2048,
    dropout=0.1,
    max_seq_len=200
)

# src/tgt: (batch_size, seq_len) integer token ids
src = torch.randint(1, 10000, (32, 50))
tgt = torch.randint(1, 10000, (32, 50))

# Optional: boolean padding masks (True = real token, False = pad)
src_mask = (src != 0)  # (batch_size, src_len)
tgt_mask = (tgt != 0)  # (batch_size, tgt_len)

logits = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
print(logits.shape)  # (32, 50, 10000)
```

---

## Design Decisions

| Decision | Detail |
|---|---|
| Weight tying | Target embedding weights shared with output projection |
| Embedding scaling | Input embeddings multiplied by `sqrt(d_model)` before positional encoding |
| Causal mask storage | Precomputed and stored as a `register_buffer` for efficiency |
| Padding convention | `True` = valid token, `False` = padding (masked out) |
| Default `d_ff` | `4 × embed_size` when not explicitly provided |

---

## Reference

> Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017).  
> **Attention Is All You Need.** *Advances in Neural Information Processing Systems (NeurIPS).*  
> https://arxiv.org/abs/1706.03762
