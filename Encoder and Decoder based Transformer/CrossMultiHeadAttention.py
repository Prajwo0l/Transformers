import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossMultiHeadAttention(nn.Module):
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

    def forward(self, query_input, key_value_input=None, padding_mask=None):
        """
        query_input:     (B, tgt_len, embed)   ← from decoder
        key_value_input: (B, src_len, embed)   ← from encoder
        padding_mask:    (B, src_len) boolean, True = valid token, False = pad
        """
        if key_value_input is None:
            key_value_input = query_input

        B, Tq, C = query_input.shape
        _, Tk, _ = key_value_input.shape

        Q = self.q_proj(query_input)
        K = self.k_proj(key_value_input)
        V = self.v_proj(key_value_input)

        # Reshape → (B, nh, T, hd)
        Q = Q.view(B, Tq, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, Tk, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, Tk, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * (self.head_dim ** -0.5)

        # Apply padding mask on the key dimension (src side)
        if padding_mask is not None:
            # padding_mask (B, src_len) → expand to (B, 1, 1, src_len)
            # We mask where padding_mask == False
            scores = scores.masked_fill(
                ~padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(B, Tq, C)

        out = self.out(out)
        return out