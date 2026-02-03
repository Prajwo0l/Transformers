import torch.nn as nn
from MultiHeadAttention import MultiHeadAttention

img_size=28
num_channels=1
patch_size=7
num_patches=(img_size//patch_size)**2
token_dim=32
num_heads=4
num_layers=4 #number of transformer layers
mlp_hidden_dim=4*token_dim
num_classes=10
learning_rate=3e-4
epochs=5
batch_size=64

class TransformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layernorm1=nn.LayerNorm(token_dim)
        self.layernorm2=nn.LayerNorm(token_dim)
        self.multihead_attention=MultiHeadAttention(token_dim,num_heads,batch_first=True)
        self.mlp=nn.Sequential(
            nn.Linear(token_dim,mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim,token_dim)
        )
    
    def forward(self,x):
        residual1=x
        x=self.layernorm1(x)
        x=self.multihead_attention(x,x,x)[0]
        x=x+residual1

        residual2=x
        x=self.layernorm2(x)
        x=self.mlp(x)
        x=x+residual2
        return x