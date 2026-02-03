import torch
import torch.nn as nn

from PatchEmbedding import PatchEmbedding
from TransformerEncoder import TransformerEncoder
from Classification import MLPHead

token_dim=32
patch_size=7
img_size=28
num_patches=(img_size//patch_size)**2
num_layers=4
class VisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embedding=PatchEmbedding()
        self.cls_token=nn.Parameter(torch.randn(1,1,token_dim))
        self.postion_embedding=nn.Parameter(torch.randn(1,num_patches+1,token_dim))
        self.transformer_encoder=nn.Sequential(*[TransformerEncoder()for _ in range(num_layers)])
        self.mlp_head=MLPHead()


    def forward(self,x):
        x=self.patch_embedding(x)
        num_image_in_batch=x.shape[0]
        cls_token=self.cls_token.expand(num_image_in_batch,-1,-1)
        x=torch.cat((cls_token,x),dim=1)
        x=x+self.postion_embedding
        x=self.transformer_encoder(x)
        x=x[:,0]
        x=self.mlp_head(x)
        return x