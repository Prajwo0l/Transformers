import torch 
import torch as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self,embedded_size,d_ff):
        super().__init__()
        self.fc1=nn.Linear(embedded_size,d_ff)
        self.relu=nn.ReLU()
        self.fc2=nn.Linear(d_ff,embedded_size)

    def forward(self,x):
        output=self.fc1(x)
        output=self.relu(output)
        output=self.fc2(output)
        return output
