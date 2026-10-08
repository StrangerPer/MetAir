# cauair_improved.py
import math
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.base.model import BaseModel  

logger = logging.getLogger(__name__)

# ---------------- FeedForward ----------------
class FeedForward(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=None, dropout=0.1):
        super().__init__()
        out_dim = out_dim if out_dim is not None else in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout)
        )
        self.res_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.res_proj(x)

# ---------------- Temporal Low-Rank Attention ----------------
class TemporalLowRankAttention(nn.Module):
    def __init__(self, hidden_dim, rank=32, max_len=24):  
        super().__init__()
        self.U = nn.Linear(hidden_dim, rank, bias=False)
        self.V = nn.Linear(hidden_dim, rank, bias=False)
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # initialize decay emphasizing recent timesteps
        init = torch.linspace(1.0, 0.2, steps=max_len)
        self.temporal_decay = nn.Parameter(init)

    def forward(self, x, H):
        """
        x: [B, N, D]  node-level pooled features
        H: [B, N, T, D] history sequence features
        returns: [B, N, D]
        """
        if H is None:
            return x
        B, N, D = x.shape
        _, _, T, _ = H.shape
        x_r = self.U(x).unsqueeze(2)         # [B,N,1,R]
        H_r = self.V(H)                      # [B,N,T,R]
        interaction = (x_r * H_r).sum(-1)    # [B,N,T]
        decay = self.temporal_decay[:T].view(1,1,T)  # [1,1,T]
        attn_w = F.softmax(interaction * decay, dim=-1).unsqueeze(-1)  # [B,N,T,1]
        temporal_fused = (H * attn_w).sum(dim=2)  # [B,N,D]
        return x + self.W(temporal_fused)

# ---------------- GatedTokenMixer ----------------
class GatedTokenMixer(nn.Module):
    def __init__(self, num_tokens, hidden_dim):
        super().__init__()
        # token mixer acts on tokens dimension (num_tokens)
        self.mlp = nn.Sequential(
            nn.Linear(num_tokens, num_tokens * 4),
            nn.GELU(),
            nn.Linear(num_tokens * 4, num_tokens)
        )
        self.gate = nn.Linear(num_tokens, num_tokens)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x: [B, G, D]  (token dimension is G)
        y = x.transpose(1,2)                   # [B,D,G]
        gate = torch.sigmoid(self.gate(y))     # [B,D,G]
        y_mlp = self.mlp(y)
        y = gate * y_mlp + (1 - gate) * y
        return self.dropout(y.transpose(1,2))  # [B,G,D]
        

# ---------------- Channel: Squeeze-and-Excitation + FFN ----------------
class ChannelSEFFN(nn.Module):
    """
    Channel mixer: SE-style gating (global pooling across nodes) + per-node FFN.
    Input x: [B, N, D]
    cond: optional [B, N, C] or [B, C] (will be pooled)
    """
    def __init__(self, dim, se_hidden=None, ff_hidden=None, dropout=0.1, use_cond=False, cond_dim=0):
        super().__init__()
        self.dim = dim
        self.use_cond = use_cond
        se_hidden = se_hidden or max(8, dim // 4)
        ff_hidden = ff_hidden or max(dim * 2, dim + 32)
        gate_in = dim + (cond_dim if use_cond else 0)
        self.se_mlp = nn.Sequential(
            nn.Linear(gate_in, se_hidden),
            nn.ReLU(),
            nn.Linear(se_hidden, dim),
            nn.Sigmoid()
        )
        self.ffn = FeedForward(dim, ff_hidden, dim, dropout=dropout)

    def forward(self, x, cond=None):
        # x: [B,N,D]
        pooled = x.mean(dim=1)  # [B,D]
        if self.use_cond and (cond is not None):
            if cond.dim() == 3:
                cond_pooled = cond.mean(dim=1)  # [B,C]
            else:
                cond_pooled = cond  # assume [B,C]
            se_in = torch.cat([pooled, cond_pooled], dim=-1)
        else:
            se_in = pooled
        gates = self.se_mlp(se_in)  # [B,D]
        x_gated = x * gates.unsqueeze(1)  # [B,N,D]
        out = self.ffn(x_gated)  # [B,N,D]
        return out

class S_MLPMixerBlock(nn.Module):
    def __init__(self, num_nodes, hidden_dim, top_k_moe=2, rank=32, max_history=24,
                 channel_type='ffn', channel_kwargs=None):   # CA是ffn好
        """
        Simplified S_MLPMixerBlock without any neighbor-adj based propagation.
        - num_nodes: number of nodes (G used for token mixer)
        - hidden_dim: embedding dim D
        - channel_type: 'seffn' (default), 'ffn', or 'chatt' (channel attention - not used by default)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        # learnable residual scales
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.1))

        # token mixer (group tokens expected to be computed by caller and passed via adj)
        self.ln_token = nn.LayerNorm(hidden_dim)
        self.token_mixer = GatedTokenMixer(num_nodes, hidden_dim)

        # temporal interaction
        self.temporal_attn = TemporalLowRankAttention(hidden_dim, rank=rank, max_len=max_history)

        # channel mixing (pre-norm)
        self.ln_channel = nn.LayerNorm(hidden_dim)
        channel_kwargs = channel_kwargs or {}
        if channel_type == 'seffn':
            use_cond = channel_kwargs.get('use_cond', False)
            cond_dim = channel_kwargs.get('cond_dim', 0)
            self.channel_mixer = ChannelSEFFN(hidden_dim,
                                              se_hidden=channel_kwargs.get('se_hidden', None),
                                              ff_hidden=channel_kwargs.get('ff_hidden', None),
                                              dropout=channel_kwargs.get('dropout', 0.1),
                                              use_cond=use_cond,
                                              cond_dim=cond_dim)
            self._channel_type = 'seffn'
        elif channel_type == 'ffn':
            self.channel_mixer = FeedForward(hidden_dim,
                                             channel_kwargs.get('ff_hidden', max(128, hidden_dim*2)),
                                             hidden_dim,
                                             dropout=channel_kwargs.get('dropout', 0.1))
            self._channel_type = 'ffn'
        else:
            # fallback to FFN for safety
            self.channel_mixer = FeedForward(hidden_dim, max(128, hidden_dim*2), hidden_dim, dropout=0.1)
            self._channel_type = 'ffn'

    def forward(self, x, adj, history_emb, cond_for_channel=None):
        """
        x: [B, N, D]   node features
        adj: [N, G]    node->group assignment (float); tokenization is performed here
        history_emb: [B, N, T, D]   per-node history embeddings (optional for temporal attn)
        cond_for_channel: optional conditioning tensor for channel mixer, shape [B,N,C] or [B,C]
        """
        # node -> group tokens
        xg = torch.einsum('ng,bnd->bgd', adj, x)  # [B,G,D]
        xg = self.ln_token(xg)
        y = self.token_mixer(xg)                  # [B,G,D]
        yn = torch.einsum('gn,bgd->bnd', adj.mT, y)  # [B,N,D]
        x = x + self.alpha * yn

        # temporal low-rank attention (node-level)
        x = self.temporal_attn(x, history_emb)

        # channel mixing
        x_norm = self.ln_channel(x)
        if self._channel_type == 'seffn':
            y_c = self.channel_mixer(x_norm, cond=cond_for_channel)
        else:
            y_c = self.channel_mixer(x_norm)
        x = x + self.beta * y_c

        return x

class CrossBranchMultiHead(nn.Module):
    """
    Query: branch A (past), Key/Value: branch B (future)
    x: [B,N,D], z: [B,N,D]  -> returns updated x (and optionally updated z)
    Use small num_heads to keep light.
    """
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.Wq = nn.Linear(dim, dim)
        self.Wk = nn.Linear(dim, dim)
        self.Wv = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.tensor(0.1))  # residual scale
        self.ln = nn.LayerNorm(dim)

    def forward(self, x, z):
        # snapshot to avoid sequential update leakage
        x0 = x
        z0 = z
        B,N,D = x.shape
        q = self.Wq(self.ln(x0)).view(B,N,self.num_heads,self.head_dim)  # [B,N,H,hd]
        k = self.Wk(self.ln(z0)).view(B,N,self.num_heads,self.head_dim) 
        v = self.Wv(self.ln(z0)).view(B,N,self.num_heads,self.head_dim)
        # attention over nodes: use einsum to compute per-head attention scores
        attn_scores = torch.einsum('bnhd,bmhd->bhnm', q, k) / math.sqrt(self.head_dim)  # [B,H,N,N]
        attn = torch.softmax(attn_scores, dim=-1)
        out = torch.einsum('bhnm,bmhd->bnhd', attn, v)  # [B,N,H,hd]
        out = out.reshape(B,N,D)
        out = self.out(out)
        return x + self.alpha * self.dropout(out)  # residual update

class RMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed


class SwiGLU_FFN(nn.Module):
    def __init__(self, dim_in, dim_out, expand_ratio=4, dropout=0.3):
        super(SwiGLU_FFN, self).__init__()
        self.W1 = nn.Linear(dim_in, expand_ratio * dim_in)
        self.W2 = nn.Linear(dim_in, expand_ratio * dim_in)
        self.W3 = nn.Linear(expand_ratio * dim_in, dim_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.W3(self.dropout(F.silu(self.W1(x)) * self.W2(x)))
        

class CauAir(BaseModel):
    def __init__(self, dim, rank, head=4, num_groups=30, **args):
        super(CauAir, self).__init__(**args)
        self.seq_len = self.seq_len
        self.node_num = self.node_num
        self.input_dim = self.input_dim
        self.output_dim = self.output_dim
        self.horizon = self.horizon

        hidden_dim = dim

        self.group1 = nn.Parameter(torch.randn(self.node_num, num_groups))
        self.group2 = nn.Parameter(torch.randn(self.node_num, num_groups))

        self.encoderCro1 = nn.ModuleList([
            S_MLPMixerBlock(num_groups, hidden_dim, rank=rank)
            for _ in range(3)
        ])
        self.encoderCro2 = nn.ModuleList([
            S_MLPMixerBlock(num_groups, hidden_dim, rank=rank)
            for _ in range(3)
        ])
        print(rank)

        self.time_step_proj_p = nn.Linear(self.input_dim, hidden_dim)
        self.time_step_proj_f = nn.Linear(self.input_dim, hidden_dim)

        self.regression_layer = nn.Linear(hidden_dim, self.horizon)

        self.alp = nn.Linear(dim*2, dim)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.cross_attn1 = CrossBranchMultiHead(hidden_dim, num_heads=4)   # CA是4
        self.cross_attn2 = CrossBranchMultiHead(hidden_dim, num_heads=4)
        
        self.encoder11 = SwiGLU_FFN(self.seq_len, dim)
        self.encoder22 = SwiGLU_FFN(self.seq_len, dim)
        self.encoder1 = SwiGLU_FFN(self.seq_len * (self.input_dim - 1), dim)
        self.encoder2 = SwiGLU_FFN(self.seq_len * (self.input_dim - 1), dim)

        self.position1 = nn.Parameter(torch.zeros((self.node_num, dim)))
        self.position2 = nn.Parameter(torch.zeros((self.node_num, dim)))
        self.position3 = nn.Parameter(torch.zeros((self.node_num, dim)))
        self.position4 = nn.Parameter(torch.zeros((self.node_num, dim)))

        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.norm3 = RMSNorm(dim)
        self.norm4 = RMSNorm(dim)
       


    def forward(self, x, label=None):
        """
        x: [B, T, N, input_dim]
           input_dim=1: AQI, input_dim>1: covariates
        label: [B, T, N, D], future covariates
        """
        B, T, N, D_in = x.shape
        aqi = x[..., :1]  # [B,T,N,1] 
        past_cov = x[..., 1:]  # [B,T,N,D_in-1] 
        future_cov = label  # [B,T,N,D_in-1] 

        branch_p = torch.cat([aqi, past_cov], dim=-1)  # B,T,N,D_in
        branch_f = torch.cat([aqi, future_cov], dim=-1)  # B,T,N,D_in
        
        aqi_p = past_cov.transpose(1, 2).reshape(-1, self.node_num, self.seq_len * (self.input_dim - 1))
        aqi_f = future_cov.transpose(1, 2).reshape(-1, self.node_num, self.seq_len * (self.input_dim - 1))
        
        a_p = self.encoder2(aqi_p) + self.norm2(self.position2)
        a_f = self.encoder1(aqi_f) + self.norm1(self.position1)
        
        
        x1 = self.encoder11(x[..., 0].transpose(1, -1)) + self.norm3(self.position3)
        x2 = self.encoder22(x[..., 0].transpose(1, -1)) + self.norm4(self.position4)
        time_emb_p = x1 + a_p
        time_emb_f = x2 + a_f

        his_p = self.time_step_proj_p(branch_p.permute(0, 2, 1, 3))  # B,N,T,D_h
        his_f = self.time_step_proj_f(branch_f.permute(0, 2, 1, 3))  # B,N,T,D_h

        group_adj1 = F.softmax(self.group1, dim=1)  # N,g
        group_adj2 = F.softmax(self.group2, dim=1)  # N,g

        fused_p = time_emb_p
        fused_f = time_emb_f

        for i in range(len(self.encoderCro1)):
            fused_p = self.encoderCro1[i](fused_p, group_adj1, history_emb=his_p)
            fused_f = self.encoderCro2[i](fused_f, group_adj2, history_emb=his_f)

            # Cross-branch Attention
            p_update = self.cross_attn1(fused_p, fused_f)   
            f_update = self.cross_attn2(fused_f, fused_p)  # symmetric optionally using same module or different
            fused_p = p_update
            fused_f = f_update

        gate = torch.sigmoid(self.alp(torch.cat([fused_p, fused_f], dim=-1)))
        fused_node = gate * fused_p + (1 - gate) * fused_f

        out = self.regression_layer(fused_node)  # [B,N,horizon]
        return out.unsqueeze(-1).permute(0, 2, 1, 3)  # [B,horizon,N,1]