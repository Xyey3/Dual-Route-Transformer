import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
except Exception:
    Mamba = None


class TemporalConvMixer(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(TemporalConvMixer, self).__init__()
        self.depthwise = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            groups=d_model,
        )
        self.pointwise = nn.Linear(d_model, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y = self.depthwise(x.transpose(1, 2)).transpose(1, 2)
        y = self.pointwise(self.activation(y))
        return self.dropout(y)


class MambaBlock(nn.Module):
    def __init__(self, configs):
        super(MambaBlock, self).__init__()
        self.norm1 = nn.LayerNorm(configs.d_model)
        if Mamba is None:
            self.mixer = TemporalConvMixer(configs.d_model, configs.dropout)
        else:
            self.mixer = Mamba(
                d_model=configs.d_model,
                d_state=getattr(configs, 'd_state', None) or getattr(configs, 'mamba_d_state', 16),
                d_conv=getattr(configs, 'mamba_d_conv', 4),
                expand=getattr(configs, 'mamba_expand', 2),
            )
        self.norm2 = nn.LayerNorm(configs.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_ff),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_ff, configs.d_model),
        )
        self.dropout = nn.Dropout(configs.dropout)

    def forward(self, x):
        x = x + self.dropout(self.mixer(self.norm1(x)))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class Model(nn.Module):
    """
    S-Mamba baseline with temporal selective-state mixing over multivariate tokens.
    Falls back to a depthwise temporal mixer when mamba_ssm is unavailable.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm

        self.value_embedding = nn.Linear(configs.enc_in, configs.d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, configs.seq_len, configs.d_model))
        self.blocks = nn.ModuleList([MambaBlock(configs) for _ in range(configs.e_layers)])
        self.norm = nn.LayerNorm(configs.d_model)
        self.temporal_projection = nn.Linear(configs.seq_len, configs.pred_len)
        self.output_projection = nn.Linear(configs.d_model, configs.c_out)
        self.dropout = nn.Dropout(configs.dropout)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        x = self.value_embedding(x_enc)
        x = self.dropout(x + self.position_embedding[:, :x.shape[1], :])

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = self.temporal_projection(x.transpose(1, 2)).transpose(1, 2)
        dec_out = self.output_projection(x)

        if self.use_norm and dec_out.shape[-1] == x_enc.shape[-1]:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1)

        return dec_out, None

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :], attns
