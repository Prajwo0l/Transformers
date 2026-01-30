import torch
import torch.nn as nn
import functional as F
import math

from Encoder import TransformerEncoder
from Decoder import TransformerDecoder
from PostionalEncoding import PositionalEncoding

class Transformer(nn.Module):
    def __init__(self,src_vocab_size,
                 tgt_vocab_size,
                 embed_size=512,
                 num_heads=8,
                 num_encoder_layer=6,
                 num_decoder_layer=6,
                 d_ff=2048,
                 dropout=0.1,
                 max_seq_len=200):
        super().__init__()
        self.embed_size=embed_size

        self.src_embedding=nn.Embedding(src_vocab_size,embed_size,padding_idx=0)
        self.tgt_embedding=nn.Embedding(tgt_vocab_size,embed_size,padding_idx=0)

        self.pos_encoding=PositionalEncoding(embed_size,max_seq_len=max_seq_len)

        self.encoder=TransformerEncoder(
            embed_size=embed_size,
            num_heads=num_heads,
            num_layers=num_encoder_layer,
            d_ff=d_ff,
            dropout=dropout
        )
        self.decoder=TransformerDecoder(
            embed_size=embed_size,
            num_heads=num_heads,
            num_layers=num_decoder_layer,
            d_ff=d_ff,
            dropout=dropout
        )
        self.out_proj=nn.Linear(embed_size,tgt_vocab_size)
        self.tgt_embedding.weight=self.out_proj.weight

    def forward(self,src,tgt,src_mask=None,tgt_mask=None):
        src_embed=self.src_embedding(src)*math.sqrt(self.embed_size)
        src_embed=self.pos_encoding(src_embed)

        tgt_embed=self.tgt_embedding(tgt)* math.sqrt(self.embed_size)
        tgt_embed=self.pos_encoding(tgt_embed)

        enc_out=self.encoder(src_embed,mask=src_mask)
        dec_out=self.decoder(tgt_embed,
                             enc_out,
                             tgt_mask=tgt_mask,
                             src_mask=src_mask)
        logits=self.out_proj(dec_out)
        return logits
    