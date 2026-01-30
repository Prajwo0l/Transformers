# Transformer From Scratch – “Attention Is All You Need” Implementation

## 📌 Overview

This project is a from-scratch PyTorch implementation of the Transformer architecture proposed in the paper “Attention Is All You Need” (Vaswani et al., 2017).
The goal of this project is to deeply understand and replicate the original encoder–decoder Transformer model without relying on high-level frameworks such as HuggingFace Transformers.

The implementation includes all core components of the Transformer, including positional encoding, multi-head self-attention, encoder and decoder layers, and feed-forward networks.

## 🧠 Key Concepts Implemented

This repository implements the following components from the original paper:

1. Positional Encoding

Sinusoidal positional encoding as described in the paper

Added to token embeddings to inject sequence order information

2. Multi-Head Attention

Linear projections for Query, Key, and Value

Head splitting and parallel attention computation

Scaled dot-product attention

Head concatenation and final linear projection (fc_out)

3. Transformer Encoder

Stacked encoder layers (configurable number of layers)

Self-attention + residual connections + layer normalization

Position-wise feed-forward networks

4. Transformer Decoder

Masked self-attention for autoregressive decoding

Cross-attention (decoder queries, encoder keys/values)

Feed-forward layers with residual connections and normalization

5. Complete Encoder–Decoder Transformer

End-to-end Transformer architecture replicating the original paper

Modular and reusable PyTorch classes

## 🏗️ Project Structure
```text
├── FeedForward.py
├── MultiHeadAttention.py
├── PositionalEncoding.py
├── EncoderLayer.py
├── DecoderLayer.py
├── Transformer.py
├── train.py (optional)
└── README.md


## ⚙️ Technologies Used

Python 3.x

PyTorch

NumPy

Math (for sinusoidal positional encoding)

```python

##  🚀 How to Run
Install dependencies
pip install torch

## Example Usage
```python
from Transformer import Transformer
import torch

model = Transformer(
    d_model=512,
    num_heads=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    d_ff=2048
)

src = torch.randint(0, 1000, (32, 50))   # batch_size=32, seq_len=50
tgt = torch.randint(0, 1000, (32, 50))

out = model(src, tgt)
print(out.shape)


## 📖 Learning Goals of This Project

Understand the mathematical and architectural foundations of Transformers

Implement attention mechanisms manually instead of using high-level libraries

Learn tensor shape manipulation, masking, and multi-head attention internals

Build a reusable Transformer architecture for future research and experiments

## 📚 Reference Paper

Vaswani et al., Attention Is All You Need, NeurIPS 2017
https://arxiv.org/abs/1706.03762

## 🧩 Future Improvements

Add training loop for machine translation tasks

Implement learned and rotary positional embeddings

Add visualization for attention weights

Optimize with PyTorch Lightning or Accelerate

Implement GPT-style decoder-only Transformer

Benchmark against HuggingFace Transformer outputs


## ⭐ Acknowledgements

This project was built for educational purposes to deeply understand the Transformer architecture and its internal workings.
