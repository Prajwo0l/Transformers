#Testing with Dummy Data
import torch 
import torch as nn
import torch.nn.functional as F

from FeedForward import FeedForward
from multi_head_attention import MultiHeadAttention
from PostionalEncoding import PostionalEncoding

batch_size=2
seq_len=5
embedded_size=8
num_heads=2

#Dummy Embeddings
x=torch.rand(batch_size,seq_len,embedded_size)

#Step 1 : Postional Encoding
pos_enc=PostionalEncoding(embedded_size)
x=pos_enc(x)
#step 2 : After the Positional Encoding 
mha=MultiHeadAttention(embedded_size,num_heads)
out=mha(x)
print("Output after multi-head attention : \n",out.shape)
#Step3 : Residual connection with the output
residual=x+out
#Step4:Layer Normalization of the output from residual connection

norm=nn.LayerNorm(embedded_size)
out_norm=norm(residual)

print("Shape after Add& Norm",out_norm.shape)


#Step 5 : After the output is normalized  we proceed in FFN
batch_size,seq_len,embedded_size=out_norm.shape
dff=4*embedded_size
ffn=FeedForward(embedded_size,dff)
ffn_out=ffn(out_norm)
print("Output from FFN",out_norm.shape)


#Step6:
residual2=out_norm+ffn_out
norm2=nn.Layernorm(embedded_size)
final_output=norm2(residual2)

