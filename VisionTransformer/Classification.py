import torch.nn as nn

token_dim=32
num_classes=10

class MLPHead(nn.Module):
    def __init__(self):
        super().__init__()
        super.layernorm=nn.LayerNorm(token_dim)
        self.mlp=nn.Linear(token_dim,num_classes)

    def forward(self,x):
        x=self.layernorm(x)
        x=self.mlp(x)
        return x
