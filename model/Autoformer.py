import torch
import torch.nn as nn


class MovingAverage(nn.Module):
    def __init__(self, kernel_size):
        super(MovingAverage, self).__init__()
        self.kernel_size = max(1, kernel_size)
        self.avg = nn.AvgPool1d(kernel_size=self.kernel_size, stride=1, padding=0)

    def forward(self, x):
        if self.kernel_size == 1:
            return x

        left = (self.kernel_size - 1) // 2
        right = self.kernel_size - 1 - left
        front = x[:, 0:1, :].repeat(1, left, 1)
        end = x[:, -1:, :].repeat(1, right, 1)
        x = torch.cat([front, x, end], dim=1)
        return self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAverage(kernel_size)

    def forward(self, x):
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, max_len, dropout):
        super(DataEmbedding, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.value_embedding(x) + self.position_embedding[:, :x.shape[1], :])


class AutoformerEncoderLayer(nn.Module):
    def __init__(self, configs):
        super(AutoformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(
            configs.d_model,
            configs.n_heads,
            dropout=configs.dropout,
            batch_first=True,
        )
        self.decomp1 = SeriesDecomp(configs.moving_avg)
        self.decomp2 = SeriesDecomp(configs.moving_avg)
        self.norm1 = nn.LayerNorm(configs.d_model)
        self.norm2 = nn.LayerNorm(configs.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_ff),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_ff, configs.d_model),
        )
        self.dropout = nn.Dropout(configs.dropout)

    def forward(self, x):
        y, attn = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
        x, _ = self.decomp1(x + self.dropout(y))
        y = self.ffn(self.norm2(x))
        x, _ = self.decomp2(x + self.dropout(y))
        return x, attn


class AutoformerDecoderLayer(nn.Module):
    def __init__(self, configs):
        super(AutoformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(
            configs.d_model,
            configs.n_heads,
            dropout=configs.dropout,
            batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
            configs.d_model,
            configs.n_heads,
            dropout=configs.dropout,
            batch_first=True,
        )
        self.decomp1 = SeriesDecomp(configs.moving_avg)
        self.decomp2 = SeriesDecomp(configs.moving_avg)
        self.decomp3 = SeriesDecomp(configs.moving_avg)
        self.norm1 = nn.LayerNorm(configs.d_model)
        self.norm2 = nn.LayerNorm(configs.d_model)
        self.norm3 = nn.LayerNorm(configs.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_ff),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_ff, configs.d_model),
        )
        self.dropout = nn.Dropout(configs.dropout)

    def forward(self, x, cross):
        y, self_attn = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
        x, _ = self.decomp1(x + self.dropout(y))

        y, cross_attn = self.cross_attn(self.norm2(x), cross, cross, need_weights=False)
        x, _ = self.decomp2(x + self.dropout(y))

        y = self.ffn(self.norm3(x))
        x, _ = self.decomp3(x + self.dropout(y))
        return x, (self_attn, cross_attn)


class Model(nn.Module):
    """
    Autoformer-style decomposition Transformer baseline.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm

        max_len = max(configs.seq_len, configs.label_len + configs.pred_len) + 8
        self.decomp = SeriesDecomp(configs.moving_avg)
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, max_len, configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, max_len, configs.dropout)
        self.encoder = nn.ModuleList([AutoformerEncoderLayer(configs) for _ in range(configs.e_layers)])
        self.decoder = nn.ModuleList([AutoformerDecoderLayer(configs) for _ in range(configs.d_layers)])
        self.encoder_norm = nn.LayerNorm(configs.d_model)
        self.decoder_norm = nn.LayerNorm(configs.d_model)
        self.projection = nn.Linear(configs.d_model, configs.c_out)
        self.trend_projection = nn.Linear(configs.dec_in, configs.c_out)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        seasonal_init, trend_init = self.decomp(x_enc)
        mean = x_enc.mean(dim=1, keepdim=True).repeat(1, self.pred_len, 1)
        zeros = torch.zeros(
            x_enc.shape[0],
            self.pred_len,
            x_enc.shape[2],
            device=x_enc.device,
            dtype=x_enc.dtype,
        )
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)

        enc_out = self.enc_embedding(x_enc)
        attns = []
        for layer in self.encoder:
            enc_out, attn = layer(enc_out)
            attns.append(attn)
        enc_out = self.encoder_norm(enc_out)

        dec_out = self.dec_embedding(seasonal_init)
        for layer in self.decoder:
            dec_out, attn = layer(dec_out, enc_out)
            attns.append(attn)
        dec_out = self.decoder_norm(dec_out)

        seasonal_part = self.projection(dec_out[:, -self.pred_len:, :])
        trend_part = self.trend_projection(trend_init[:, -self.pred_len:, :])
        dec_out = seasonal_part + trend_part

        if self.use_norm and dec_out.shape[-1] == x_enc.shape[-1]:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1)

        return dec_out, attns

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :], attns
