import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
  def __init__(self,embedded_size,num_heads,dropout=0.1):
    super().__init__()
    self.embedded_size=embedded_size
    self.num_heads=num_heads

    self.head_dim=embedded_size//num_heads
    assert embedded_size%num_heads==0
    self.dropout = nn.Dropout(dropout)
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

    scores=torch.matmul(Q,K.transpose(-2,-1))/(self.head_dim**0.5)
    if mask is not None:
      scores = scores.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float('-inf'))
    attention=F.softmax(scores,dim=-1)
    out=torch.matmul(attention,V)

    out=out.transpose(1,2).contiguous().view(batch_size,seq_length,embedded_size)

    out=self.fc_out(out)
    return out