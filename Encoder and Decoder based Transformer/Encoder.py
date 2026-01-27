#Testing with Dummy Data
import torch 
import torch as nn
import torch.nn.functional as F

from FeedForward import FeedForward
from multi_head_attention import MultiHeadAttention
from PostionalEncoding import PostionalEncoding

class EncoderLayer(nn.Module):
    def __init__(self,embedded_size,num_heads,dff=None,dropout=0.1):
        super().__init__()

        if dff is None:
            dff=4*embedded_size
        
        self.self_attn=MultiHeadAttention(embedded_size,num_heads,dropout=dropout)
        self.norm1=nn.LayerNorm(embedded_size)
        self.dropout1=nn.Dropout(dropout)

        self.ffn=FeedForward(embedded_size,dff)
        self.norm2=nn.LayerNorm(embedded_size)
        self.dropout2=nn.Dropout(dropout)

    def forward(self,x):
        # Multihead self attention
        attn_out=self.self_attn(x)
        x=x+self.dropout1(attn_out)
        x=self.norm1(x)

        ffn_out=self.ffn(x)
        encoder_output=x+self.dropout2(ffn_out)
        encoder_output=self.norm2(x)

        return encoder_output