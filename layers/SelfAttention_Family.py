import torch
import torch.nn as nn
from torch import einsum
import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
from reformer_pytorch import LSHSelfAttention
from einops import rearrange,repeat
from layers.mamba import *

# Code implementation from https://github.com/thuml/Flowformer
class FlowAttention(nn.Module):
    def __init__(self, attention_dropout=0.1):
        super(FlowAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)

    def kernel_method(self, x):
        return torch.sigmoid(x)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        # kernel
        queries = self.kernel_method(queries)
        keys = self.kernel_method(keys)
        # incoming and outgoing
        normalizer_row = 1.0 / (torch.einsum("nhld,nhd->nhl", queries + 1e-6, keys.sum(dim=2) + 1e-6))
        normalizer_col = 1.0 / (torch.einsum("nhsd,nhd->nhs", keys + 1e-6, queries.sum(dim=2) + 1e-6))
        # reweighting
        normalizer_row_refine = (
            torch.einsum("nhld,nhd->nhl", queries + 1e-6, (keys * normalizer_col[:, :, :, None]).sum(dim=2) + 1e-6))
        normalizer_col_refine = (
            torch.einsum("nhsd,nhd->nhs", keys + 1e-6, (queries * normalizer_row[:, :, :, None]).sum(dim=2) + 1e-6))
        # competition and allocation
        normalizer_row_refine = torch.sigmoid(
            normalizer_row_refine * (float(queries.shape[2]) / float(keys.shape[2])))
        normalizer_col_refine = torch.softmax(normalizer_col_refine, dim=-1) * keys.shape[2]  # B h L vis
        # multiply
        kv = keys.transpose(-2, -1) @ (values * normalizer_col_refine[:, :, :, None])
        x = (((queries @ kv) * normalizer_row[:, :, :, None]) * normalizer_row_refine[:, :, :, None]).transpose(1,
                                                                                                                2).contiguous()
        return x, None


# Code implementation from https://github.com/shreyansh26/FlashAttention-PyTorch
class FlashAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FlashAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def flash_attention_forward(self, Q, K, V, mask=None):
        BLOCK_SIZE = 32
        NEG_INF = -1e10  # -infinity
        EPSILON = 1e-10
        # mask = torch.randint(0, 2, (128, 8)).to(device='cuda')
        O = torch.zeros_like(Q, requires_grad=True)
        l = torch.zeros(Q.shape[:-1])[..., None]
        m = torch.ones(Q.shape[:-1])[..., None] * NEG_INF

        O = O.to(device='cuda')
        l = l.to(device='cuda')
        m = m.to(device='cuda')

        Q_BLOCK_SIZE = min(BLOCK_SIZE, Q.shape[-1])
        KV_BLOCK_SIZE = BLOCK_SIZE

        Q_BLOCKS = torch.split(Q, Q_BLOCK_SIZE, dim=2)
        K_BLOCKS = torch.split(K, KV_BLOCK_SIZE, dim=2)
        V_BLOCKS = torch.split(V, KV_BLOCK_SIZE, dim=2)
        if mask is not None:
            mask_BLOCKS = list(torch.split(mask, KV_BLOCK_SIZE, dim=1))

        Tr = len(Q_BLOCKS)
        Tc = len(K_BLOCKS)

        O_BLOCKS = list(torch.split(O, Q_BLOCK_SIZE, dim=2))
        l_BLOCKS = list(torch.split(l, Q_BLOCK_SIZE, dim=2))
        m_BLOCKS = list(torch.split(m, Q_BLOCK_SIZE, dim=2))

        for j in range(Tc):
            Kj = K_BLOCKS[j]
            Vj = V_BLOCKS[j]
            if mask is not None:
                maskj = mask_BLOCKS[j]

            for i in range(Tr):
                Qi = Q_BLOCKS[i]
                Oi = O_BLOCKS[i]
                li = l_BLOCKS[i]
                mi = m_BLOCKS[i]

                scale = 1 / np.sqrt(Q.shape[-1])
                Qi_scaled = Qi * scale

                S_ij = torch.einsum('... i d, ... j d -> ... i j', Qi_scaled, Kj)
                if mask is not None:
                    # Masking
                    maskj_temp = rearrange(maskj, 'b j -> b 1 1 j')
                    S_ij = torch.where(maskj_temp > 0, S_ij, NEG_INF)

                m_block_ij, _ = torch.max(S_ij, dim=-1, keepdims=True)
                P_ij = torch.exp(S_ij - m_block_ij)
                if mask is not None:
                    # Masking
                    P_ij = torch.where(maskj_temp > 0, P_ij, 0.)

                l_block_ij = torch.sum(P_ij, dim=-1, keepdims=True) + EPSILON

                P_ij_Vj = torch.einsum('... i j, ... j d -> ... i d', P_ij, Vj)

                mi_new = torch.maximum(m_block_ij, mi)
                li_new = torch.exp(mi - mi_new) * li + torch.exp(m_block_ij - mi_new) * l_block_ij

                O_BLOCKS[i] = (li / li_new) * torch.exp(mi - mi_new) * Oi + (
                        torch.exp(m_block_ij - mi_new) / li_new) * P_ij_Vj
                l_BLOCKS[i] = li_new
                m_BLOCKS[i] = mi_new

        O = torch.cat(O_BLOCKS, dim=2)
        l = torch.cat(l_BLOCKS, dim=2)
        m = torch.cat(m_BLOCKS, dim=2)
        return O, l, m

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        res = \
        self.flash_attention_forward(queries.permute(0, 2, 1, 3), keys.permute(0, 2, 1, 3), values.permute(0, 2, 1, 3),
                                     attn_mask)[0]
        return res.permute(0, 2, 1, 3).contiguous(), None


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


# Code implementation from https://github.com/zhouhaoyi/Informer2020
class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        # real U = U_part(factor*ln(L_k))*L_q
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(
            L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(
            Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H,
                                                L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            # requires that L_Q == L_V, i.e. for self-attention only
            assert (L_Q == L_V)
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) /
                     L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[
                                                  None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * \
                 np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * \
            np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(
            queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(
            context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class ReformerLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, causal=False, bucket_size=4, n_hashes=4):
        super().__init__()
        self.bucket_size = bucket_size
        self.attn = LSHSelfAttention(
            dim=d_model,
            heads=n_heads,
            bucket_size=bucket_size,
            n_hashes=n_hashes,
            causal=causal
        )

    def fit_length(self, queries):
        # inside reformer: assert N % (bucket_size * 2) == 0
        B, N, C = queries.shape
        if N % (self.bucket_size * 2) == 0:
            return queries
        else:
            # fill the time series
            fill_len = (self.bucket_size * 2) - (N % (self.bucket_size * 2))
            return torch.cat([queries, torch.zeros([B, fill_len, C]).to(queries.device)], dim=1)

    def forward(self, queries, keys, values, attn_mask, tau, delta):
        # in Reformer: defalut queries=keys
        B, N, C = queries.shape
        queries = self.attn(self.fit_length(queries))[:, :N, :]
        return queries, None

class TwoStageAttentionLayer(nn.Module):
    '''
    The Two Stage Attention (TSA) Layer
    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    '''
    def __init__(self, seg_num, factor, var_num, d_model, n_heads, d_ff=None, dropout=0.1):
        super(TwoStageAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.time_attention = AttentionLayer(FullAttention(False, attention_dropout=dropout), d_model, n_heads)
        self.dim_sender = AttentionLayer(FullAttention(False, attention_dropout=dropout), d_model, n_heads)
        self.dim_receiver = AttentionLayer(FullAttention(False, attention_dropout=dropout), d_model, n_heads)
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))
        
        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.MLP2 = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.linear_pred = nn.Linear(d_model, 56)

    def forward(self, queries):
        # Cross Time Stage: Directly apply MSA to each dimension
        batch = queries.shape[0]

        time_in = rearrange(queries, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')

        time_enc, time_attn = self.time_attention(
            time_in,
            time_in,
            time_in,
            None,
        )
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)

        # Cross Dimension Stage: use a small set of learnable vectors to aggregate and distribute messages to build the D-to-D connection
        dim_send = rearrange(dim_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b = batch)

        batch_router = repeat(self.router, 'seg_num factor d_model -> (repeat seg_num) factor d_model', repeat = batch)

        dim_buffer, sender_attn = self.dim_sender(batch_router, dim_send, dim_send, None)
        dim_receive, receiver_attn = self.dim_receiver(dim_send, dim_buffer, dim_buffer, None)

        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)
        
        final_out = rearrange(dim_enc, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b = batch)

        return final_out, (time_attn, sender_attn, receiver_attn)

    # def forward(self, queries):
    #     '''
    #     queries: [batch, D, L, d_model]
    #     '''
    #     batch = queries.shape[0]

    #     # ==============================
    #     # 1️⃣ Cross-Dimension Stage
    #     # ==============================
    #     dim_in = rearrange(queries, 'b ts_d seg_num d_model -> (b seg_num) ts_d d_model', b=batch)

    #     # broadcast learnable router
    #     batch_router = repeat(self.router, 'seg_num factor d_model -> (repeat seg_num) factor d_model', repeat=batch)

    #     # sender & receiver attention
    #     dim_buffer, sender_attn = self.dim_sender(batch_router, dim_in, dim_in, None)
    #     dim_receive, receiver_attn = self.dim_receiver(dim_in, dim_buffer, dim_buffer, None)

    #     dim_enc = dim_in + self.dropout(dim_receive)
    #     dim_enc = self.norm1(dim_enc)
    #     dim_enc = dim_enc + self.dropout(self.MLP1(dim_enc))
    #     dim_enc = self.norm2(dim_enc)

    #     # reshape back
    #     time_in = rearrange(dim_enc, '(b seg_num) ts_d d_model -> (b ts_d) seg_num d_model', b=batch)

    #     # ==============================
    #     # 2️⃣ Cross-Time Stage
    #     # ==============================
    #     time_enc, time_attn = self.time_attention(time_in, time_in, time_in, None)

    #     time_out = time_in + self.dropout(time_enc)
    #     time_out = self.norm3(time_out)
    #     time_out = time_out + self.dropout(self.MLP2(time_out))
    #     time_out = self.norm4(time_out)

    #     # ==============================
    #     # Output reshape
    #     # ==============================
    #     final_out = rearrange(time_out, '(b ts_d) seg_num d_model -> b ts_d seg_num d_model', b=batch)

    #     return final_out, (sender_attn, receiver_attn, time_attn)

# class DynamicRoutingLayer(nn.Module):
#     def __init__(self, seg_num, var_num, d_model, update_rate=0.3):
#         super(DynamicRoutingLayer, self).__init__()
#         self.d_model = d_model
#         self.seg_num = seg_num
#         self.update_rate = update_rate  # Controls the update step size
#         self.router = nn.Parameter(torch.randn(seg_num, var_num, d_model))

#     def forward(self, inputs):
#         """
#         Args:
#             inputs: Tensor of shape [(batch, seg_num), var_num, d_model]
#         Returns:
#             Updated router after dynamic routing.
#         """
#         # Compute similarity matrix to generate relationships between dimensions
#         similarity = torch.bmm(inputs, inputs.transpose(1, 2))  # [(batch * seg_num), var_num, var_num]

#         # Compute routing weights using softmax
#         routing_weights = nn.functional.softmax(similarity, dim=-1)  # [(batch * seg_num), var_num, var_num]

#         # Expand router to match batch size and segments
#         batch_router = repeat(self.router, 's v d -> (b s) v d', b=inputs.shape[0] // self.seg_num)

#         # Update router using routing weights
#         updated_router = torch.bmm(routing_weights, batch_router)  # [(batch * seg_num), var_num, d_model]
#         updated_router=nn.functional.softmax(updated_router, dim=1)

#         return (1 - self.update_rate) * batch_router + batch_router * updated_router
      
class DynamicRoutingLayer(nn.Module):
    def __init__(self, factor, var_num, d_model):
        super(DynamicRoutingLayer, self).__init__()
        self.router = nn.Parameter(torch.randn(factor, d_model))
        self.d_conv = 3
        self.conv1d = nn.Conv1d(
            in_channels=var_num,
            out_channels=factor,
            kernel_size=1,     
        )
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()
        
    def forward(self, inputs):
        """
        Args:
            inputs: Tensor of shape [(batch, seg_num), var_num, d_model]
        Returns:
            Updated router after dynamic routing.
        """
        inputs = self.conv1d(inputs)                   # [(b, s), factor, d_model]   
        inputs = self.act(self.norm(inputs))
        return F.normalize(inputs * self.router, dim=-1)
      
class DynamicTwoStageAttentionLayer(nn.Module):
    '''
    The Two Stage Attention (TSA) Layer
    input/output shape: [batch_size, Variable_dim(V), Seg_num(L), d_model]
    '''
    def __init__(self, factor, var_num, d_model, n_heads, d_ff=None, dropout=0.1):
        super(DynamicTwoStageAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.time_attention = AttentionLayer(FullAttention(False, attention_dropout=dropout), d_model, n_heads)
        self.dim_sender = AttentionLayer(FullAttention(False, attention_dropout=dropout), d_model, n_heads)
        self.dim_receiver = AttentionLayer(FullAttention(False, attention_dropout=dropout), d_model, n_heads)
        # self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))

        # dynamic_routing
        self.dynamic_routing = DynamicRoutingLayer(factor, var_num ,d_model)

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.MLP2 = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.linear_pred = nn.Linear(d_model, 56)

    def forward(self, queries):
        # Cross Time Stage: Directly apply MSA to each dimension
        batch = queries.shape[0]

        time_in = rearrange(queries, 'b v seg_num d_model -> (b v) seg_num d_model')

        time_enc, time_attn = self.time_attention( 
            time_in,
            time_in,
            time_in,
            None,
        )
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)

        # Cross Dimension Stage: use a small set of learnable vectors to aggregate and distribute messages to build the D-to-D connection
        dim_send = rearrange(dim_in, '(b v) seg_num d_model -> (b seg_num) v d_model', b = batch)
        batch_router = self.dynamic_routing(dim_send)

        dim_buffer, sender_attn = self.dim_sender(batch_router, dim_send, dim_send, None)
        dim_receive, receiver_attn = self.dim_receiver(dim_send, dim_buffer, dim_buffer, None)

        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)
        
        final_out = rearrange(dim_enc, '(b seg_num) v d_model -> b v seg_num d_model', b = batch)

        return final_out, (time_attn, sender_attn, receiver_attn)
    
class SCX_Block(nn.Module): # Sparse Cluster Extractor 
    def __init__(self,seg_num, var_num, heads, d_model, attn_dropout, groups=10, return_full_attn=False):
        super().__init__()
        self.num_per_group = max(math.ceil(var_num / groups), 1)
        self.gather_layer = nn.Conv1d(groups * self.num_per_group, groups, groups=groups, kernel_size=1)
        self.dropout = nn.Dropout(attn_dropout)
        self.seg_num = seg_num
        self.q, self.k, self.v = nn.Linear(d_model, d_model), nn.Linear(d_model, d_model), nn.Linear(d_model, d_model)
        self.out = nn.Sequential(nn.Linear(d_model, d_model), nn.Dropout(attn_dropout))
        self.cluster = nn.Parameter(torch.randn(seg_num, groups, d_model))
        self.groups, self.heads, self.scale = groups, heads, d_model / heads
        self.return_full_attn = return_full_attn

    def forward(self, x):
        # x: [(b seg_num), var_num, d_model]
        bs, var_num, d_model = x.shape
        b=bs//self.seg_num
        h, d_h, K = self.heads, d_model // self.heads, self.num_per_group
        x = torch.log(F.relu(x) + 1.0)
        cluster = repeat(self.cluster, 'seg_num groups d_model -> (repeat seg_num) groups d_model', repeat = b)

        q = self.q(cluster).reshape(bs, self.groups, h, d_h).permute(0, 2, 1, 3)  # [bs, h, groups, var_num]
        k = self.k(x).reshape(bs, var_num, h, d_h).permute(0, 2, 1, 3)
        v = self.v(x).reshape(bs, var_num, h, d_h).permute(0, 2, 1, 3)

        scores = torch.matmul(q, k.transpose(-1, -2)) * (self.scale ** -0.5)     # [bs, h, groups, var_num]
        topk_scores, idx = torch.topk(scores, k=K, dim=-1)                       # [bs, h, groups, K]
        attn_k = F.softmax(topk_scores, dim=-1)
        attn_k = self.dropout(attn_k)

        idx_expanded = idx.unsqueeze(-1).expand(-1, -1, -1, -1, d_h)             # [bs, h, groups, K, d_h]
        xx_ = torch.take_along_dim(v.unsqueeze(2), idx_expanded, dim=3)          # [bs, h, groups, K, d_h]
        x = self.gather_layer(xx_.reshape(bs * h, self.groups * K, d_h))          # [bs*h, groups, d_h]
        x = x.reshape(bs, h, -1, d_h)
        # x = torch.exp((x - x.min()) / (x.max() - x.min() + 1e-12))
        x_min = x.min(dim=-1, keepdim=True)[0]  # shape: [bs, h, g, d_h]
        x_max = x.max(dim=-1, keepdim=True)[0]  # shape: [bs, h, g, d_h]
        # 避免除以0
        denom = (x_max - x_min).clamp(min=1e-6)
        x = (x - x_min) / denom
        x = torch.exp(x)
        
        out = self.out(rearrange(x, 'bs h g d_h -> bs g (h d_h)', h=h))

        if self.return_full_attn:
            full_attn = F.softmax(scores, dim=-1)
            full_attn = self.dropout(full_attn)
            return out, full_attn
        else:
            return out, (attn_k, idx)
    
class DualRouteInteractionBlock(nn.Module):
    '''
    Dual-Route Interaction (DRI) Block
    Input/Output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    - Route 1: Temporal Path (within each variable)
    - Route 2: Dimensional Path with Sparse Routing (across variables)
    '''
    def __init__(self, seg_num, factor, var_num, d_model, n_heads, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model

        # ----- Temporal (Time-axis) Attention -----
        self.temporal_attn = AttentionLayer(
            FullAttention(False, attention_dropout=dropout),
            d_model,
            n_heads
        )

        # ----- Dimension-axis Routing (Sender: SCX, Receiver: Attention) -----
        self.dim_router_sender = SCX_Block(
            seg_num, var_num, n_heads, d_model, dropout, groups=factor
        )
        self.dim_router_receiver = AttentionLayer(
            FullAttention(False, attention_dropout=dropout),
            d_model,
            n_heads
        )

        self.dropout = nn.Dropout(dropout)

        # LayerNorms
        self.norm_temporal_1 = nn.LayerNorm(d_model)
        self.norm_temporal_2 = nn.LayerNorm(d_model)
        self.norm_dim_1 = nn.LayerNorm(d_model)
        self.norm_dim_2 = nn.LayerNorm(d_model)

        # Feed-forward Networks
        self.ffn_temporal = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.ffn_dimensional = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

        # Final projection example (unchanged)
        self.linear_pred = nn.Linear(d_model, 56)

    def forward(self, queries):
        batch = queries.shape[0]

        # ========== Stage 1: Temporal Path (Intra-variable attention) ==========
        temporal_input = rearrange(
            queries,
            'b v seg_num d_model -> (b v) seg_num d_model'
        )

        temporal_context, temporal_attn_map = self.temporal_attn(
            temporal_input, temporal_input, temporal_input, None
        )

        temporal_res = temporal_input + self.dropout(temporal_context)
        temporal_norm = self.norm_temporal_1(temporal_res)

        temporal_ffn_out = temporal_norm + self.dropout(self.ffn_temporal(temporal_norm))
        temporal_out = self.norm_temporal_2(temporal_ffn_out)

        # ========== Stage 2: Dimension Path (Inter-variable routing) ==========
        # reshape to merge batch & seg, keep variables as sequence
        dim_tokens_in = rearrange(
            temporal_out,
            '(b v) seg_num d_model -> (b seg_num) v d_model',
            b=batch
        )

        # ---- Sender (SCX sparse cluster routing) ----
        route_tokens, route_select_attn = self.dim_router_sender(dim_tokens_in)

        # ---- Receiver (message distribution back to variables) ----
        dim_message, route_distribute_attn = self.dim_router_receiver(
            dim_tokens_in, route_tokens, route_tokens, None
        )

        dim_res = dim_tokens_in + self.dropout(dim_message)
        dim_norm = self.norm_dim_1(dim_res)

        dim_ffn_out = dim_norm + self.dropout(self.ffn_dimensional(dim_norm))
        dim_out = self.norm_dim_2(dim_ffn_out)

        # ========== Restore original shape (B, D, L, d_model) ==========
        output_tensor = rearrange(
            dim_out,
            '(b seg_num) v d_model -> b v seg_num d_model',
            b=batch
        )

        return output_tensor, (
            temporal_attn_map,
            route_select_attn,
            route_distribute_attn
        )