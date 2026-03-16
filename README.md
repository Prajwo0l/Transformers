# Transformers From Scratch

A collection of Transformer architecture implementations built entirely from scratch using PyTorch — no high-level libraries like HuggingFace or `timm`. Each sub-project is a self-contained, modular implementation aimed at deeply understanding the internals of Transformer-based models.

---

## Repository Structure

```
Transformers/
├── Encoder and Decoder based Transformer/   # Full seq2seq Transformer (Vaswani et al., 2017)
│   ├── PostionalEncoding.py
│   ├── FeedForward.py
│   ├── multi_head_attention.py
│   ├── MaskedMultiHeadAttention.py
│   ├── CrossMultiHeadAttention.py
│   ├── Encoder.py
│   ├── Decoder.py
│   ├── Transformer.py
│   └── README.md
│
├── VisionTransformer/                       # ViT (Dosovitskiy et al., 2020)
│   ├── PatchEmbedding.py
│   ├── MultiHeadAttention.py
│   ├── PostionalEncoding.py
│   ├── TransformerEncoder.py
│   ├── Classification.py
│   ├── VisionTransformer.py
│   ├── ViT_Experiment.ipynb
│   └── README.md
│
└── README.md                                # This file
```

---

## Projects

### 1. Encoder–Decoder Transformer
> Paper: [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., NeurIPS 2017

A complete PyTorch implementation of the original sequence-to-sequence Transformer. Covers sinusoidal positional encoding, multi-head self-attention, masked self-attention, cross-attention, stacked encoder/decoder layers, and a full end-to-end Transformer class with embedding weight tying.

→ See [`Encoder and Decoder based Transformer/README.md`](./Encoder%20and%20Decoder%20based%20Transformer/README.md)

---

### 2. Vision Transformer (ViT)
> Paper: [An Image is Worth 16×16 Words](https://arxiv.org/abs/2010.11929) — Dosovitskiy et al., ICLR 2021

A pure PyTorch implementation of the Vision Transformer applied to image classification. Covers patch embedding via convolution, learnable class token and positional embeddings, stacked Transformer encoder blocks with Pre-LN and GELU, and an MLP classification head. Trained and evaluated on MNIST.

→ See [`VisionTransformer/README.md`](./VisionTransformer/README.md)

---

## Motivation

Both projects were built with the same philosophy: implement everything manually, understand every tensor operation, and avoid abstracting away the core mechanics behind library calls. The goal is not just working code — it's a working mental model of how Transformers operate at every level.

---

## Requirements

```bash
pip install torch numpy jupyter
```

Python 3.8+ and PyTorch 2.0+ recommended.

---

## References

- Vaswani et al. — *Attention Is All You Need* (2017) — https://arxiv.org/abs/1706.03762
- Dosovitskiy et al. — *An Image is Worth 16×16 Words* (2020) — https://arxiv.org/abs/2010.11929
