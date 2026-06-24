import torch 
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, CrossEncLayer
from layers.SelfAttention_Family import DualRouteInteractionBlock_
from layers.Embed import DataEmbedding_inverted, DSW_embedding, DataEmbedding
from einops import rearrange, repeat
import numpy as np

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        # B: batch_size;    D: d_model;
        # L: seq_len;       P: pred_len;
        # V: number of variate (tokens)

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm
        self.var_num = configs.enc_in

        # embedding (segmented)
        # Input:  [B, L, V]
        # Output: [B, V, D]
        self.embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, dropout=configs.dropout)

        # Encoder
        # Output shape after each block: [B, V, D]
        self.encoder = Encoder(
            [
                CrossEncLayer(
                    DualRouteInteractionBlock_(
                        configs.factor,
                        self.var_num,
                        configs.d_model,
                        configs.n_heads,
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
        self.dim_norm = nn.LayerNorm(configs.d_model)
        
        self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)  

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc: [B, L, V]
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x_enc = x_enc / stdev

        # embedding
        # [B, L, V] -> [B, V, D]
        x_enc = self.embedding(x_enc, x_mark_enc)          # [B, V, D]
        
        # Dual-route encoder
        # Output: [B, V, D], [B, V, D]
        output, attns = self.encoder(x_enc, attn_mask=None)
        output = self.dim_norm(output)

        # Prediction
        # [B, V, D] -> [B, P, V]
        dec_out = self.projection(output).transpose(1, 2)
        dec_out = dec_out[:, :, :self.var_num]

        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1)

        return dec_out, attns

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, attns = self.forecast(
            x_enc, x_mark_enc, x_dec, x_mark_dec
        )
        return dec_out[:, -self.pred_len:, :], attns