import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    PatchTST-style channel-independent patch Transformer for long-term forecasting.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm
        self.output_attention = configs.output_attention

        patch_len = getattr(configs, 'patch_len', 16)
        stride = getattr(configs, 'stride', 8)
        self.patch_len = min(patch_len, self.seq_len)
        self.stride = max(1, min(stride, self.patch_len))

        if self.seq_len <= self.patch_len:
            self.padding = 0
        else:
            remainder = (self.seq_len - self.patch_len) % self.stride
            self.padding = (self.stride - remainder) % self.stride

        self.n_patches = (self.seq_len + self.padding - self.patch_len) // self.stride + 1

        self.patch_embedding = nn.Linear(self.patch_len, configs.d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, self.n_patches, configs.d_model))
        fc_dropout = configs.fc_dropout if getattr(configs, 'fc_dropout', None) is not None else configs.dropout
        head_dropout = configs.head_dropout if getattr(configs, 'head_dropout', None) is not None else configs.dropout
        self.dropout = nn.Dropout(fc_dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=configs.d_model,
            nhead=configs.n_heads,
            dim_feedforward=configs.d_ff,
            dropout=configs.dropout,
            activation=configs.activation,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=configs.e_layers,
            norm=nn.LayerNorm(configs.d_model),
        )
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(head_dropout),
            nn.Linear(self.n_patches * configs.d_model, configs.pred_len),
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        batch_size, _, n_vars = x_enc.shape
        x = x_enc.permute(0, 2, 1)
        if self.padding > 0:
            x = F.pad(x, (0, self.padding), mode='replicate')

        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride).contiguous()
        patches = patches.view(batch_size * n_vars, self.n_patches, self.patch_len)

        enc_out = self.patch_embedding(patches)
        enc_out = self.dropout(enc_out + self.position_embedding)
        enc_out = self.encoder(enc_out)
        dec_out = self.head(enc_out)
        dec_out = dec_out.view(batch_size, n_vars, self.pred_len).transpose(1, 2)

        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1)

        return dec_out, None

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :], attns
