import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
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

    def forward(self,x,mask=None):
        batch_size,seq_length,embedded_size=x.shape

        Q=self.query(x)
        K=self.key(x)
        V=self.value(x)

        Q=Q.view(batch_size,seq_length,self.num_heads,self.head_dim).transpose(1,2)
        K=K.view(batch_size,seq_length,self.num_heads,self.head_dim).transpose(1,2)
        V=V.view(batch_size,seq_length,self.num_heads,self.head_dim).transpose(1,2)

        scores=torch.matmul(Q,K.transpose(-2,-1))/torch.sqrt(torch.tensor(self.head_dim,dtype=torch.float32))
        attention=F.softmax(scores,dim=-1)
        output=torch.matmul(attention,scores)

        output=output.transpose(1,2).contiguous().view(batch_size,seq_length,self.embedded_size)

        output=self.fc_out(output)
        return output