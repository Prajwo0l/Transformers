import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads, dropout=0.1):
        super().__init__()
        assert embed_size % num_heads == 0

        self.embed_size = embed_size
        self.num_heads  = num_heads
        self.head_dim   = embed_size // num_heads

        self.q_proj = nn.Linear(embed_size, embed_size)
        self.k_proj = nn.Linear(embed_size, embed_size)
        self.v_proj = nn.Linear(embed_size, embed_size)
        self.out    = nn.Linear(embed_size, embed_size)

        self.dropout = nn.Dropout(dropout)

        # Causal (future) mask – upper triangle
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(5000, 5000), diagonal=1).bool()
        )

    def forward(self, x, padding_mask=None):
        """
        x: (B, T, C)
        padding_mask: (B, T) boolean, True = keep (real token), False = mask (padding)
        """
        B, T, C = x.shape

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape to (B, nh, T, hd)
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * (self.head_dim ** -0.5)

        # 1. Apply causal mask (no looking at future tokens)
        causal_m = self.causal_mask[:T, :T]                  # (T, T)
        scores = scores.masked_fill(causal_m, float('-inf'))

        # 2. Apply padding mask (mask out padding positions in keys)
        if padding_mask is not None:
            # padding_mask: (B, T) → expand to (B, 1, 1, T) for broadcasting over heads & queries
            # We want to mask where padding_mask == False
            scores = scores.masked_fill(
                ~padding_mask.unsqueeze(1).unsqueeze(2),   # (B, 1, 1, T)
                float('-inf')
            )

        # Softmax + dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        out = self.out(out)
        return out