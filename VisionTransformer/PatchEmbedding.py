import torch
import torch.nn as nn
import functional as F


img_size=28
patch_size=7
num_channels=1
token_dim=32
patch_size=(img_size//patch_size)**2

class PatchEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed=nn.Conv2d(num_channels,token_dim,kernel_size=patch_size,stride=patch_size)

    def forward(self,x):
        x=self.patch_embed(x)
        x=x.flatten(2)
        x=x.transpose(1,2)
        return x
    