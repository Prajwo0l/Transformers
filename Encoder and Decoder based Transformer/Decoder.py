import torch
import torch as nn 
import torch.nn.functional as F

from FeedForward import FeedForward
from MaskedMultiHeadAttention import MaskedMultiHeadAttention
from CrossMultiHeadAttention import CrossMultiHeadAttention
from Encoder import *

class DecoderLayer(nn.Module):
    def __init__(self, embed_size, num_heads, d_ff=None, dropout=0.1):
        super().__init__()

        if d_ff is None:
            d_ff = 4 * embed_size  # common default: 4×d_model

        # 1. Masked (causal) self-attention
        self.self_attn = MaskedMultiHeadAttention(embed_size, num_heads)

        # 2. Cross-attention (looks at encoder output)
        self.cross_attn = CrossMultiHeadAttention(embed_size, num_heads)

        # 3. Feed-forward
        self.ffn = FeedForward(embed_size, d_ff)

        # Layer norms
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_output, tgt_padding_mask=None, src_padding_mask=None):
        # 1. Masked self-attention (causal + padding)
        self_attn_out = self.self_attn(x, padding_mask=tgt_padding_mask)
        x = self.norm1(x + self.dropout1(self_attn_out))

        # 2. Cross-attention (padding on encoder side)
        cross_out = self.cross_attn(x, encoder_output, padding_mask=src_padding_mask)
        x = self.norm2(x + self.dropout2(cross_out))

        # 3. Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_out))

        return x
    
class TransformerDecoder(nn.Module):
    def __init__(self,embed_size,num_heads,num_layers,d_ff=None,dropout=0.1):
        super().__init__()
        self.layers=nn.ModuleList([
            DecoderLayer(embed_size,num_heads,d_ff or 4*embed_size,dropout)
            for _ in range(num_layers)
        ])
        self.norm=nn.Layer(embed_size)

    def forward(self,x,encoder_output,tgt_mask=None,src_mask=None):
        for layer in self.layers:
            x=layer(x,encoder_output,tgt_padding_mask=tgt_mask,src_padding_mask=src_mask)
        x=self.norm(x)
        return x
    