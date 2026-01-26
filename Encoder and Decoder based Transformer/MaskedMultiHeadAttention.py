import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedMultiHeadAttention(nn.Module):
    def __init__(self,embedded_size,num_heads,dropout=0.1,context_length=512):
        super().__init__()
        assert embedded_size%num_heads==0

        self.embedded_size=embedded_size
        self.num_heads=num_heads
        self.head_size=embedded_size//num_heads

        self.query=nn.Linear(embedded_size,embedded_size)
        self.key=nn.Linear(embedded_size,embedded_size)
        self.value=nn.Linear(embedded_size,embedded_size)
        self.out_proj=nn.Linear(embedded_size,embedded_size)

        self.dropout=nn.Dropout(dropout)

        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))


    def forward(self,x):
        batch_size,seq_len,_=x.size()

        Q=self.query(x)
        K=self.key(x)
        V=self.value(x)

        #Reshape and transpose forr multi head attention
        Q=Q.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        K=K.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        V=V.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)

        scores=torch.matmul(Q,K.transpose(-2,-1))/(self.head_dim**0.5)
        #Apply causal mask:set future postions to -inf before softmax
        mask=self.mask[:seq_len,:seq_len].bool()
        scores=scores.masked_fill(mask,float("-inf"))

        #Apply softmax to get attention weights
        scores=F.softmax(scores,dim=-1)
        scores=self.dropout(scores)

        #Compute context vectors
        context_vec=torch.matmul(scores,V)

        #Concatenate head and project back
        context_vec=context_vec.transpose(1,2).contiguous().view(batch_size,seq_len,self.embedded_size)
        output=self.out_proj(context_vec)

        return output