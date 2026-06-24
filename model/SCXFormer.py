import torch 
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, CrossEncLayer
from layers.SelfAttention_Family import DimensionalInteractionBlock
from layers.Embed import DataEmbedding_inverted, DSW_embedding
from einops import rearrange, repeat
import numpy as np

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        # B: batch_size;    D: d_model;
        # L: seq_len;       P: pred_len;
        # V: number of variate (tokens), can also includes covariates
        # N: Seg_num
        # K: Groups

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm
        self.var_num = configs.enc_in

        self.seg_num = configs.seg_num

        # DIP embedding (segmented)
        # Input:  [B, L, V]
        # Output: [B, V, N, D]
        self.dip_enc_embedding = DSW_embedding(
            configs.seq_len // configs.seg_num,
            configs.d_model,
            configs.dropout
        )

        # Encoder
        # Output shape after each block: [B, V, D]
        self.encoder = Encoder(
            [
                CrossEncLayer(
                    DimensionalInteractionBlock(
                        configs.seg_num,
                        configs.factor,
                        self.var_num,
                        configs.d_model,
                        configs.n_heads,
                        configs.d_ff,
                        configs.dropout
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                )
                for _ in range(configs.e_layers)
            ],
            
        )
        self.dim_norm=nn.LayerNorm(configs.d_model)

        # Prediction head
        # Input:  [B, V, D]
        # Output: [B, V, P]
        self.projection = nn.Linear(
            configs.seg_num * configs.d_model,
            configs.pred_len,
            bias=True
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        # x_enc: [B, L, V]
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x_enc = x_enc / stdev

        # DIP embedding
        # [B, L, V] -> [B, V, N, D]
        dip_out = self.dip_enc_embedding(x_enc)
        
        # encoder
        # Output: [B, V, D], [B, V, N, D]
        dim_out, attns = self.encoder(dip_out,attn_mask=None)
        dim_out = self.dim_norm(dim_out)
        
        # flatten segment & feature
        # project back to D
        dim_out = rearrange(dim_out, 'B V N D -> B V (N D)')   # [B, V, N*D]

        # Prediction
        # [B, V, N*D] -> [B, V, P] -> [B, P, V]
        dec_out = self.projection(dim_out).transpose(1, 2)

        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1)

        return dec_out, attns

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, attns = self.forecast(
            x_enc, x_mark_enc, x_dec, x_mark_dec
        )
        return dec_out[:, -self.pred_len:, :], attns