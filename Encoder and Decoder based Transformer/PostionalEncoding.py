import math
import torch
import torch as nn
import torch.nn.functional as F

class PostionalEncoding(nn.Module):
    def __init__(self,embedded_size:int,max_len:int=5000):
        super().__init__()
        position=torch.arange(0,max_len).unsqueeze(1)
        div_term=torch.exp(torch.arange(0,embedded_size*2)-(math.log(10000.0)/embedded_size))

        pe=torch.zeros(max_len,embedded_size)
        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)

        self.register_buffer('pe',pe.unsqueeze(0))
        
    def forward(self,x:torch.Tensor)-> torch.Tensor:
        return x+self.pe[:,:x.size(1)]
