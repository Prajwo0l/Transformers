import torch
import torch as nn 
import torch.nn.functional as F

from FeedForward import FeedForward
from MaskedMultiHeadAttention import MaskedMultiHeadAttention
from CrossMultiHeadAttention import CrossMultiHeadAttention
from Encoder import *

class DecoderLayer(nn.Module):
    def __init__(self,embedded_size,num_heads,dff=None,dropout=0.1):
        super().__init__()

        if dff is None:
            dff=4*embedded_size
        self.masked_attn=MaskedMultiHeadAttention(embedded_size,num_heads,dropout=0.2)
        self.norm1=nn.LayerNorm(embedded_size)
        self.dropout1=nn.Dropout(dropout)

        self.cross_attn=CrossMultiHeadAttention(embedded_size,num_heads,dropout)
        self.norm2=nn.LayerNorm(embedded_size)
        self.dropout2=nn.Dropout(dropout)



        self.ffn=FeedForward(embedded_size,dff)
        self.norm2=nn.LayerNorm(embedded_size)
        self.dropout3=nn.Dropout(dropout)

    def forward(self,x,encoder_output,self_mask=None,cross_mask=None):
        attn_out=self.masked_attn(x,x,mask=self_mask)
        x=self.norm1(x+self.dropout1(attn_out))

        cross_out=self.cross_attn(x,encoder_output,mask=cross_mask)
        x=self.norm2(x+self.dropout2(cross_out))


        ffn_out=self.ffn(x)
        x=self.norm3(x+self.dropout3(ffn_out))

        return x
