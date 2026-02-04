# Vision Transformer (ViT) from Scratch in PyTorch

<img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">  
<img src="https://img.shields.io/badge/Python-3.8%2B-brightgreen" alt="Python">  
<img src="https://img.shields.io/badge/License-MIT-blue" alt="License">

Implementation of the **Vision Transformer (ViT)** model **from scratch** using only PyTorch — no high-level libraries like `timm` or `transformers`.  
Tested and trained on **MNIST** for simplicity and quick experimentation.

Based on the paper:  
**"An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale"** (Dosovitskiy et al., 2020)

---

## 📌 Project Goals

- Understand how Vision Transformers work internally
- Implement every major component manually
- Keep the code modular, readable and educational
- Train a small ViT that actually learns on MNIST

---

## ✨ Features

- Pure PyTorch implementation
- Modular components (Patch Embedding, Multi-Head Attention, Transformer Encoder, MLP Head)
- Learnable positional embeddings + class token
- Ready-to-run Jupyter notebook for training & visualization
- Small model size → fast training even on CPU / free Colab

---

## 📂 Project Structure

```text
vit-from-scratch/
├── Classification.py           # MLP head after [CLS] token
├── MultiHeadAttention.py       # Scaled dot-product multi-head attention
├── PatchEmbedding.py           # Image → patches → linear projection
├── PositionalEncoding.py       # (optional) sinusoidal version — not used in main model
├── TransformerEncoder.py       # One transformer block (LN → Attention → LN → MLP)
├── VisionTransformer.py        # Full model: patching + cls token + pos embed + N×encoder + head
├── ViT_Experiment.ipynb        # Training script + visualization (MNIST)
└── README.md