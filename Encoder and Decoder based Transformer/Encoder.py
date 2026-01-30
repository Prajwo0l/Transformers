#Testing with Dummy Data
import torch 
import torch as nn
import torch.nn.functional as F

from FeedForward import FeedForward
from multi_head_attention import MultiHeadAttention
from PostionalEncoding import PositionalEncoding

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_size, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.pos_enc = PositionalEncoding(embed_size)   
        self.mha = MultiHeadAttention(embed_size, num_heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.ffn = FeedForward(embed_size, d_ff)
        self.norm2 = nn.LayerNorm(embed_size)

        self.dropout1 = nn.Dropout(dropout)   
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x,mask=None):
        x = self.pos_enc(x)

        attn_out = self.mha(x,mask=mask)
        x = self.norm1(x + self.dropout1(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))

        return x
    
class TransformerEncoder(nn.Module):
    def __init__(self,embed_size,num_heads,num_layers,d_ff=None,dropout=0.1):
        super().__init__()
        self.layers=nn.ModuleList([
            TransformerEncoderBlock(embed_size,num_heads,d_ff or 4*embed_size,dropout)
            for _ in range(num_layers)
        ])
        self.norm=nn.LayerNorm(embed_size)

    def forward(self,x,mask=None):
        for layer in self.layers:
            x=layer(x,mask=mask)
        x=self.norm(x)
        return x
