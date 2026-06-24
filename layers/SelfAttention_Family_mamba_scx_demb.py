import torch
import torch.nn as nn
from torch import einsum
import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
from reformer_pytorch import LSHSelfAttention
from einops import rearrange,repeat
import math
import torch.nn.functional as F
from mamba_ssm import Mamba

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

# ==========================================
# TIP
# ==========================================
class TemporalRoute(nn.Module):
    """
    Mamba Interaction Path
    Input / Output: [B, V, D]
    """
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.mamba = Mamba(
            d_model=d_model, 
            d_state=16, 
            d_conv=4, 
            expand=2
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: [B, V, D]
        res = x
        x = self.mamba(x) 
        x = self.dropout(x)
        return self.norm(x + res), None
    
# ==========================================
# DIP
# ==========================================
class SCX_Block(nn.Module): 
    """Sparse Cluster Extractor
    Input shape:  [B, V, D]
    Output shape: [B, K, D]
    """
    def __init__(self, var_num, heads, d_model, attn_dropout, groups=10, return_full_attn=False):
        super().__init__()
        self.V = var_num          # number of variables
        self.K = groups           # number of clusters
        self.H = heads            # number of heads
        self.D = d_model
        self.Dh = d_model // heads

        # Tokens per cluster (P)
        self.P = max(math.ceil(var_num / groups), 1)

        # Learnable cluster centers: [1, K, D] (Broadcasting to Batch)
        self.cluster = nn.Parameter(torch.randn(1, self.K, self.D))

        # ===== Projections =====
        self.q = nn.Linear(self.D, self.D)
        self.kv = nn.Linear(self.D, 2 * self.D)
        
        # ===== Sparse aggregation =====
        self.gather_layer = nn.Conv1d(
            in_channels=self.K * self.P,
            out_channels=self.K,
            groups=self.K,
            kernel_size=1
        )

        # ===== Output =====
        self.out = nn.Sequential(nn.Linear(self.D, self.D), nn.Dropout(attn_dropout))
        self.dropout = nn.Dropout(attn_dropout)
        self.scale = self.Dh ** -0.5
        self.return_full_attn = return_full_attn
        
        self.post_norm = nn.LayerNorm(self.Dh)
        
    def forward(self, x):
        """
        x: [B, V, D]
        """
        B, V, D = x.shape

        # Pos-enhancement
        x = torch.log(F.relu(x) + 1.0)

        # Repeat cluster centers
        cluster = repeat(self.cluster, '1 K D -> B K D', B=B)

        # QKV projection
        q = self.q(cluster).reshape(B, self.K, self.H, self.Dh).permute(0, 2, 1, 3) # [B, H, K, Dh]

        kv = self.kv(x)
        k, v = kv.chunk(2, dim=-1)
        k = k.reshape(B, V, self.H, self.Dh).permute(0, 2, 1, 3) # [B, H, V, Dh]
        v = v.reshape(B, V, self.H, self.Dh).permute(0, 2, 1, 3) # [B, H, V, Dh]

        # Sparse attention
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale # [B, H, K, V]

        # Top-P token selection per cluster
        topk_scores, idx = torch.topk(scores, k=self.P, dim=-1) # [B, H, K, P]
        
        # Gather selected tokens
        idx_exp = idx.unsqueeze(-1).expand(-1, -1, -1, -1, self.Dh)
        xx = torch.take_along_dim(v.unsqueeze(2), idx_exp, dim=3) # [B, H, K, P, Dh]

        # Aggregate P tokens -> 1 cluster token
        x_gathered = self.gather_layer(
            xx.reshape(B * self.H, self.K * self.P, self.Dh)
        ).reshape(B, self.H, self.K, self.Dh)

        # Output projection
        x_norm = self.post_norm(x_gathered)
        out = self.out(rearrange(x_norm, 'B H K Dh -> B K (H Dh)'))

        if self.return_full_attn:
            full_attn = self.dropout(F.softmax(scores, dim=-1))
            return out, full_attn
        else:
            return out, idx

# class DimensionalRoute(nn.Module):
#     """
#     Dimensional Interaction Path (DIP)
#     Input / Output: [B, V, D]
#     """
#     def __init__(self, factor, var_num, d_model, n_heads, dropout=0.1):
#         super().__init__()
#         self.var_num = var_num
#         self.d_model = d_model
#         self.factor = factor
        
#         self.dim_router_sender = SCX_Block(
#             var_num, n_heads, d_model, dropout, groups=factor
#         )

#         self.dropout = nn.Dropout(dropout)
#         self.norm_1 = nn.LayerNorm(d_model)
#         self.norm_2 = nn.LayerNorm(d_model)

#         self.ffn = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         # x: [B, V, D]
#         # SCX Routing
#         route_tokens, route_select_attn = self.dim_router_sender(x)
#         route_tokens = self.dropout(route_tokens)  # [B, K, D]
        
#         # Global Route Distribution
#         route_global = route_tokens.mean(dim=1, keepdim=True)  # [B, 1, D]
#         dim_res = x + route_global # [B, V, D]

#         # Gated refinement
#         dim_norm = self.norm_1(dim_res)        # [B, V, D]
#         gate = self.ffn(dim_norm)              # [B, V, D]
#         dim_out = self.norm_2(dim_norm * gate) # [B, V, D]

#         return dim_out, route_select_attn
class DimensionalRoute(nn.Module):
    """
    Semantic Structural Interaction Path
    Input / Output: [B, V, D]
    """
    def __init__(self, factor, var_num, d_model, n_heads, dropout=0.1):
        super().__init__()

        self.var_num = var_num
        self.d_model = d_model
        self.factor = factor

        # Mamba semantic propagation
        # [B, V, D] -> [B, V, D]
        self.mamba_fwd = Mamba(
            d_model=d_model,
            d_state=16,
            d_conv=4,
            expand=2
        )

        self.mamba_bwd = Mamba(
            d_model=d_model,
            d_state=16,
            d_conv=4,
            expand=2
        )

        # SCX semantic extractor
        # [B, V, D] -> [B, K, D]
        self.dim_router_sender = SCX_Block(
            var_num, n_heads, d_model, dropout,
            groups=factor, return_full_attn=True
        )

        # Refinement
        # [B, V, D] -> [B, V, D]
        self.dropout = nn.Dropout(dropout)

        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x):
        # x: [B, V, D]
        residual = x

        # Bi-Mamba semantic propagation
        # [B, V, D] -> [B, V, D]
        route_fwd = self.mamba_fwd(x)
        route_rev = torch.flip(x, dims=[1])
        route_bwd = self.mamba_bwd(route_rev)
        route_bwd = torch.flip(route_bwd, dims=[1])
        # Stable fusion
        # [B, V, D]
        x = (route_fwd + route_bwd) / 2

        # SCX semantic extraction
        # [B, V, D] -> [B, K, D]
        route_tokens, route_attn = self.dim_router_sender(x)

        # Semantic redistribution
        # [B, K, D] -> [B, 1, D] -> [B, V, D]
        semantic_global = route_tokens.mean(dim=1, keepdim=True)
        x = x + semantic_global

        # Residual connection
        # [B, V, D]
        x = x + residual

        # FFN refinement
        # [B, V, D] -> [B, V, D]
        x = self.norm_1(x)
        x = x + self.dropout(self.ffn(x))
        x = self.norm_2(x)

        return x, route_attn
    
# ==========================================
# DualRouteInteractionBlock
# ==========================================
# class DualRouteInteractionBlock_(nn.Module):
#     def __init__(self, factor, var_num, d_model, n_heads, dropout=0.1):
#         super().__init__()
        
#         # 1. 变量维度的 Mamba 交互
#         self.mamba_route = TemporalRoute(d_model=d_model, dropout=dropout)

#         # 2. 变量维度的稀疏聚类交互
#         self.dim_route = DimensionalRoute(
#             factor=factor,
#             var_num=var_num,
#             d_model=d_model,
#             n_heads=n_heads,
#             dropout=dropout
#         )

#     def forward(self, x):
#         # x: [B, V, D]
#         x_mamba, mamba_attn = self.mamba_route(x)
#         x_out, dim_attn = self.dim_route(x_mamba)
#         return x_out, (mamba_attn, dim_attn)
class DualRouteInteractionBlock_(nn.Module):
    def __init__(self, factor, var_num, d_model, n_heads, dropout=0.1):
        super().__init__()

        self.dim_route = DimensionalRoute(
            factor=factor,
            var_num=var_num,
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout
        )

    def forward(self, x):
        # x: [B, V, D]
        x, attn = self.dim_route(x)
        return x, attn
##############################################################################################################################################        
##############################################################################################################################################    
##############################################################################################################################################    
class DualRouteInteractionBlock(nn.Module):
    """
    Dual-Route Interaction (DRI) Block (Parallel Version)

    Integrates:
    - Temporal Interaction Path (TIP): global temporal modeling (no segmentation)
    - Dimensional Interaction Path (DIP): segmented variable interaction

    Input:  [B, V, N, D]
    Output: [B, V, D]
    """
    def __init__(self, seg_num, factor, var_num, d_model, n_heads, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model

        # B: batch_size;    D: d_model;
        # V: number of variate (tokens), can also includes covariates
        # N: Seg_num

        # Temporal route (iTransformer-style)
        # Input / Output: [B, V, D]
        self.temporal_route = TemporalRoute(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout
        )

        # Dimensional route (segmented)
        # Input / Output: [B, V, N, D]
        self.dim_route = DimensionalRoute(
            seg_num=seg_num,
            factor=factor,
            var_num=var_num,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout
        )

    def forward(self, queries):
        """
        queries: (tip_out, dip_out)
            tip_out: [B, V, D]
            dip_out: [B, V, N, D]
        """
        tip_in, dip_in = queries

        # Temporal Interaction Path (TIP)
        # [B, V, D] -> [B, V, D]
        temporal_out, temporal_attn = self.temporal_route(tip_in)

        # Dimensional Interaction Path (DIP)
        # [B, V, N, D] -> [B, V, N, D]
        dim_out, (route_select_attn, route_distribute_attn) = self.dim_route(dip_in)

        return (temporal_out, dim_out), (
            temporal_attn,
            # route_select_attn,
            # route_distribute_attn,
        )
    
class DimensionalInteractionBlock(nn.Module):
    """
    Dimensional Interaction Block (DIB)

    Models segmented inter-variable interactions via the Dimensional Interaction Path (DIP).

    Input:  [B, V, N, D]
    Output: [B, V, N, D]
    """
    def __init__(self, seg_num, factor, var_num, d_model, n_heads, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model

        # B: batch_size;    D: d_model;
        # V: number of variate (tokens), can also includes covariates
        # N: Seg_num

        # Dimensional route (segmented)
        # Input / Output: [B, V, N, D]
        self.dim_route = DimensionalRoute(
            seg_num=seg_num,
            factor=factor,
            var_num=var_num,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout
        )

    def forward(self, x):
        """
            dip_out: [B, V, N, D]
        """

        # Dimensional Interaction Path (DIP)
        # [B, V, N, D] -> [B, V, N, D]
        dim_out, (route_select_attn, route_distribute_attn) = self.dim_route(x)

        return dim_out, (
            route_select_attn,
            route_distribute_attn,
        )
        
class RouteFusion(nn.Module):
    """
    Route Fusion Module (RFM) - Balanced Parallel Version

    Adaptively fuses Temporal Interaction Path (TIP) and Dimensional Interaction Path (DIP)
    using a symmetric gating mechanism.

    Input:
        t: [B, V, D]  - Temporal route features
        d: [B, V, D]  - Dimensional route features (aggregated over segments)
    Output:
        out:   [B, V, D]
        alpha: [B, V, D]  (gate values, alpha -> 1 means T dominant)
    """
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # 1. Gating mechanism (token-wise, channel-wise)
        self.balance_gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Sigmoid()
        )

        # 2. Joint context projection
        self.context_proj = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, t, d):
        """
        Args:
            t: [B, V, D]
            d: [B, V, D]
        """

        # Joint evidence
        combined = torch.cat([t, d], dim=-1)  # [B, V, 2D]

        # Adaptive balance gate
        alpha = self.balance_gate(combined)   # [B, V, D]

        # Symmetric weighted selection
        selected = alpha * t + (1.0 - alpha) * d

        # Non-linear joint interaction
        joint_context = self.context_proj(combined)

        # Final fusion
        out = self.norm(selected + joint_context)

        return out, alpha