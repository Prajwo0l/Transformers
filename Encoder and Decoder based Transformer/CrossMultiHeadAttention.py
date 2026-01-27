import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossMultiHeadAttention(nn.Module):
    def __init__(self,embedded_size,num_heads):
        super().__init__()
        self.num_heads=num_heads
        self.embedded_size=embedded_size
        self.head_dim=embedded_size//num_heads
        assert embedded_size%num_heads==0

        self.query=nn.Linear(embedded_size,embedded_size)
        self.key=nn.Linear(embedded_size,embedded_size)
        self.value=nn.Linear(embedded_size,embedded_size)
        
        self.fc_out=nn.Linear(embedded_size,embedded_size)

    def forward(self,x_query,x_kv=None,mask=None):
        if x_kv is None:
            x_kv=x_query
        batch_size,seq_length,embedded_size=x_query.shape
        encoder_len=x_kv.shape[1]

        Q=self.query(x_query)
        K=self.key(x_kv)
        V=self.value(x_kv)

        Q=Q.view(batch_size,seq_length,self.num_heads,self.head_dim).transpose(1,2)
        K=K.view(batch_size,encoder_len,self.num_heads,self.head_dim).transpose(1,2)
        V=V.view(batch_size,encoder_len,self.num_heads,self.head_dim).transpose(1,2)

        scores=torch.matmul(Q,K.transpose(-2,-1))/(self.head_dim**0.5)
        attention=F.softmax(scores,dim=-1)
        output=torch.matmul(attention, V)

        output=output.transpose(1,2).contiguous().view(batch_size,seq_length,self.embedded_size)

        output=self.fc_out(output)
        return output