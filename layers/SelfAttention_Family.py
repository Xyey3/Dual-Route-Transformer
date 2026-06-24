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
try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None

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
    
class TemporalRoute(nn.Module):
    """
    Temporal Interaction Path (TIP)

    iTransformer-style temporal modeling.
    Time dimension is encoded into variable tokens directly.

    Input:  [B, V, D]
    Output: [B, V, D]
    """
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1):
        super().__init__()

        # B: batch_size;    D: d_model;
        # V: number of variate (tokens), can also includes covariates

        d_ff = d_ff or 4 * d_model

        self.attn = AttentionLayer(
            FullAttention(False, attention_dropout=dropout),
            d_model,
            n_heads
        )

        self.dropout = nn.Dropout(dropout)
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        # x: [B, V, D]
        ctx, attn = self.attn(x, x, x, None)

        res = x + self.dropout(ctx)
        norm = self.norm_1(res)
        out = norm + self.dropout(self.ffn(norm))
        out = self.norm_2(out)

        return out, attn

class SVSS(nn.Module): 
    """
    Selective Variable State Space
    
    Input shape:  [(B*N), V, D]
    Output shape: [(B*N), V, D]
    """
    def __init__(self, seg_num, var_num, heads, d_model, attn_dropout, groups=10, return_full_attn=False):
        super().__init__()
        self.N = seg_num          # number of segments
        self.V = var_num          # number of variables
        self.K = groups           # number of clusters (K)
        self.D = d_model
        
        # ===== Mamba Configuration (Mamba Version) =====
        # We use a single Mamba block to handle bidirectional scanning.
        if Mamba is None:
            raise ImportError('mamba_ssm is required for Mamba-based DRFormer/SCXFormer blocks.')
        self.mamba = Mamba(
            d_model=d_model,
            d_state=8,
            d_conv=2,
            expand=1
        )
        
        # ===== Gating Mechanism =====
        # The gate learns to modulate the Bi-Mamba output. 
        self.gate = nn.Sequential(
            nn.Linear(self.D, self.D),
            nn.Sigmoid()
        )
        nn.init.constant_(self.gate[0].bias, 2.0)
        
        # ===== Output Projection =====
        self.out = nn.Sequential(
            nn.Linear(self.D, self.D),
            nn.Dropout(attn_dropout)
        )
        
        self.return_full_attn = return_full_attn

    def forward(self, x):
        """
        Input:  [(B*N), V, D]
        Output: [(B*N), V, D]
        """
        res = x
        g = self.gate(res)
        
        # Step 1: Forward Selective Scan across variables
        m_forward = self.mamba(x) # [BN, V, D]
        
        # Step 2: Backward Selective Scan across variables
        # Flip along the Variable dimension (V)
        x_backward = torch.flip(x, dims=[1]).contiguous()
        m_backward = self.mamba(x_backward)
        m_backward = torch.flip(m_backward, dims=[1]).contiguous() # Restore original order
        
        # Combine bi-directional features to ensure global interaction
        # Each position now has context from the entire variable set
        mamba_out = (m_forward + m_backward) / 2
        x_gated = g * mamba_out
        
        # Step 5: Projection
        x = self.out(x_gated)
        
        # Direct return without pooling to K
        return x, (None, None)
    
class DimensionalRoute(nn.Module):
    """
    Dimensional Interaction Path (DIP) - Mamba Optimized Version
    
    Models inter-variable dependencies by scanning across the variable dimension (V).
    This version leverages the Mamba-based SCX_Block to refine variable features
    without redundant global aggregation or secondary gating.
    
    Input / Output shape: [B, V, N, D]
    """
    def __init__(self, seg_num, factor, var_num, d_model, n_heads, d_ff=None, dropout=0.1):
        super().__init__()
        self.seg_num = seg_num
        self.var_num = var_num
        self.d_model = d_model

        # 1. Local Temporal Context (Depth-wise Conv over segments)
        # Captures local trends for each variable independently before interaction
        self.temporal_pre = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            groups=d_model
        )
        self.temporal_norm = nn.LayerNorm(d_model)
        self.alpha_pre = nn.Parameter(torch.zeros(1))

        # 2. Variable Interaction Core (Mamba-based SVSS)
        # Scans across V (e.g., 321) with O(V) complexity
        self.dim_router_sender = SVSS(
            seg_num, var_num, n_heads, d_model, dropout, groups=factor
        )

        # 3. Final Refinement
        self.dropout = nn.Dropout(dropout)
        self.norm_final = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Forward pass for Dimensional Interaction.
        x: [B, V, N, D]
        """
        B, V, N, D = x.shape

        # Step 1: Local Temporal Enhancement
        # [B, V, N, D] -> [(B*V), D, N] for 1D convolution
        x_pre = rearrange(x, 'B V N D -> (B V) D N')
        x_pre = self.temporal_pre(x_pre)
        x_pre = rearrange(x_pre, '(B V) D N -> B V N D', B=B)
        x_pre = self.temporal_norm(x_pre)
        
        # Integrate local temporal context
        x_in = x + self.alpha_pre * x_pre 
        
        # Step 2: Cross-Variable Interaction (The Mamba Scan)
        # Fold dimensions to [(B*N), V, D] where V is the sequence length
        dim_tokens_in = rearrange(x_in, 'B V N D -> (B N) V D')
        
        # SCX_Block now performs a Selective Scan across V variables
        # Output shape: [(B*N), V, D]
        dim_interacted, _ = self.dim_router_sender(dim_tokens_in)
        
        # Step 3: Residual Fusion and Projection
        # We directly fuse the interacted features with the input tokens
        dim_res = dim_tokens_in + self.dropout(dim_interacted)
        dim_out = self.norm_final(dim_res)

        # Step 4: Restore Original Shape
        # [(B*N), V, D] -> [B, V, N, D]
        out = rearrange(dim_out, '(B N) V D -> B V N D', B=B, N=N)

        # Return output and None (to maintain compatibility with multi-return interfaces)
        return out, (None, None)

        
class DualRouteInteractionBlock_(nn.Module):
    """
    Dual-Route Interaction (DRI) Block (Serial Version)

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

    def forward(self, input):
        """
            input: [B, V, N, D]
            output:[B, V, N, D]
        """
        B = input.shape[0]
        
        # Temporal Interaction Path (TIP)
        # [BV, N, D] -> [BV, N, D]        
        time_in = rearrange(input, 'B V N D -> (B V) N D')
        time_out, time_attn = self.temporal_route(time_in)
        # Dimensional Interaction Path (DIP)
        # [BV, N, D] -> [B, V, N, D]     
        dim_in=rearrange(time_out, '(B V) N D -> B V N D', B=B)
        output, (route_select_attn, route_distribute_attn) = self.dim_route(dim_in)

        return output, (
            time_attn,
            route_select_attn,
            route_distribute_attn,
        ) 
    
# class SCX_Block(nn.Module): 
#     """Sparse Cluster Extractor
#     Input shape:  [(B*N), V, D]
#     Output shape: [(B*N), K, D]
#     B   : batch size
#     N   : number of temporal segments
#     V   : number of variables (input tokens per segment)
#     K   : number of clusters (groups)
#     P   : number of selected tokens per cluster
#     D   : model hidden dimension
#     H   : number of attention heads
#     Dh  : per-head dimension (D // H)
#     """
#     def __init__(self,seg_num, var_num, heads, d_model, attn_dropout, groups=10, return_full_attn=False):
#         super().__init__()
#         self.N = seg_num          # number of segments
#         self.V = var_num          # number of variables
#         self.K = groups           # number of clusters
#         self.H = heads            # number of heads
#         self.D = d_model
#         self.Dh = d_model // heads

#         # Tokens per cluster (P)
#         self.P = max(math.ceil(var_num / groups), 1)

#         # Learnable cluster centers: [N, K, D]
#         self.cluster = nn.Parameter(torch.randn(self.N, self.K, self.D))

#         # ===== Projections =====
#         self.q = nn.Linear(self.D, self.D)
#         # self.k = nn.Linear(self.D, self.D)
#         # self.v = nn.Linear(self.D, self.D)
#         self.kv = nn.Linear(self.D, 2 * self.D)
        
#         # ===== Sparse aggregation =====
#         # Aggregate P tokens per cluster
#         self.gather_layer = nn.Conv1d(in_channels=self.K * self.P,out_channels=self.K,groups=self.K,kernel_size=1)

#         # ===== Output =====
#         self.out = nn.Sequential(nn.Linear(self.D, self.D),nn.Dropout(attn_dropout))
#         self.dropout = nn.Dropout(attn_dropout)
#         self.scale = self.Dh ** -0.5
#         self.return_full_attn = return_full_attn
        
#         # Normalize per head
#         self.post_norm = nn.LayerNorm(self.Dh)
        
#     def forward(self, x):
#         """
#         x: [(B*N), V, D]
#         """
#         BN, V, D = x.shape
#         B = BN // self.N

#         # ===== Pos-enhancement =====
#         x = torch.log(F.relu(x) + 1.0)

#         # ===== Repeat cluster centers =====
#         # cluster: [(B*N), K, D]
#         cluster = repeat(self.cluster,'N K D -> (B N) K D',B=B)

#         # ===== QKV projection =====
#         # q: [(B*N), H, K, Dh]
#         q = self.q(cluster).reshape(BN, self.K, self.H, self.Dh).permute(0, 2, 1, 3)

#         # k, v: [(B*N), H, V, Dh]
#         kv = self.kv(x)
#         k, v = kv.chunk(2, dim=-1)
#         k = k.reshape(BN, V, self.H, self.Dh).permute(0, 2, 1, 3)
#         v = v.reshape(BN, V, self.H, self.Dh).permute(0, 2, 1, 3)

#         # ===== Sparse attention =====
#         # scores: [(B*N), H, K, V]
#         scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale

#         # Top-P token selection per cluster
#         # idx: [(B*N), H, K, P]
#         topk_scores, idx = torch.topk(scores, k=self.P, dim=-1)
        
#         # attn = self.dropout(F.softmax(topk_scores, dim=-1))
#         attn = None

#         # ===== Gather selected tokens =====
#         idx = idx.unsqueeze(-1).expand(-1, -1, -1, -1, self.Dh)
#         # xx: [(B*N), H, K, P, Dh]
#         xx = torch.take_along_dim(v.unsqueeze(2), idx, dim=3)

#         # Aggregate P tokens → 1 cluster token
#         x = self.gather_layer(
#             xx.reshape(BN * self.H, self.K * self.P, self.Dh)
#         ).reshape(BN, self.H, self.K, self.Dh)

#         # Output
#         x = self.post_norm(x)
#         out = self.out(rearrange(x, 'BN H K Dh -> BN K (H Dh)'))

#         if self.return_full_attn:
#             full_attn = self.dropout(F.softmax(scores, dim=-1))
#             return out, full_attn
#         else:
#             return out, (attn, idx)

# class DimensionalRoute(nn.Module):
#     """
#     Dimensional Interaction Path (DIP)
#     Models inter-variable dependencies via sparse routing at each time step.
#     Input / Output: [B, V, N, D]
#     """
#     def __init__(self, seg_num, factor, var_num, d_model, n_heads, d_ff=None, dropout=0.1):
#         super().__init__()
#         d_ff = d_ff or 4 * d_model

#         self.seg_num = seg_num
#         self.var_num = var_num
#         self.d_model = d_model
#         self.factor = factor

#         self.temporal_pre = nn.Conv1d(
#             in_channels=d_model,
#             out_channels=d_model,
#             kernel_size=3,
#             padding=1,
#             groups=d_model
#         )
#         self.temporal_norm = nn.LayerNorm(d_model)
        
#         self.dim_router_sender = SCX_Block(
#             seg_num, var_num, n_heads, d_model, dropout, groups=factor
#         )

#         # self.dim_router_receiver = AttentionLayer(
#         #     FullAttention(False, attention_dropout=dropout),
#         #     d_model,
#         #     n_heads
#         # )

#         self.dropout = nn.Dropout(dropout)
#         self.norm_1 = nn.LayerNorm(d_model)
#         self.norm_2 = nn.LayerNorm(d_model)

#         # self.ffn = nn.Sequential(
#         #     nn.Linear(d_model, d_ff),
#         #     nn.GELU(),
#         #     nn.Linear(d_ff, d_model)
#         # )
#         self.ffn = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             nn.Sigmoid()
#         )

#         self.alpha_pre = nn.Parameter(torch.tensor(0.5))

    # def forward(self, x):
    #     # x: [B, V, N, D]
    #     B, V, N, D = x.shape  # B=batch size, V=variables, N=segments, D=hidden dim

    #     x_pre = rearrange(x, 'B V N D -> (B V) D N')  # [(B*V), D, N]
    #     x_pre = self.temporal_pre(x_pre)              # [(B*V), D, N]
        
    #     x_pre = rearrange(x_pre, '(B V) D N -> B V N D', B=B)  # [B, V, N, D]
    #     x_pre = self.temporal_norm(x_pre)
        
    #     x_in = x + self.alpha_pre * x_pre              # [B, V, N, D]
    #     dim_tokens_in = rearrange(x_in, 'B V N D -> (B N) V D')  # [(B*N), V, D]

    #     route_tokens, route_select_attn = self.dim_router_sender(dim_tokens_in)
    #     route_tokens = self.dropout(route_tokens)  
    #     # route_tokens: [(B*N), K, D]
    #     # route_select_attn: sparse selection attn (top-k indices / weights)

    #     # dim_message, route_distribute_attn = self.dim_router_receiver(
    #     #     dim_tokens_in, route_tokens, route_tokens, None
    #     # )  
    #     # # dim_message: [(B*N), V, D]  route_tokens: [(B*N), K, D]
    #     # # route_distribute_attn: full attention map over route tokens
    #     # dim_res = dim_tokens_in + self.dropout(dim_message)  # [(B*N), V, D]
        
    #     route_global = route_tokens.mean(dim=1, keepdim=True)  # [(B*N), 1, D]
        
    #     dim_res = dim_tokens_in + route_global # [(B*N), V, D]

    #     dim_norm = self.norm_1(dim_res)        # [(B*N), V, D]

    #     gate = self.ffn(dim_norm)              # [(B*N), V, D]
    #     dim_out = self.norm_2(dim_norm * gate) # [(B*N), V, D]

    #     out = rearrange(dim_out, '(B N) V D -> B V N D', B=B, N=N)  # [B, V, N, D]

    #     # return out, (route_select_attn, route_distribute_attn)
    #     return out, (route_select_attn)
    
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
