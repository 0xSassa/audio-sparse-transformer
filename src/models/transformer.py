"""Encoder transformer, reimplementato senza nn.MultiheadAttention.

Costo di un blocco, per n elementi in dimensione d:

    proiezioni Q, K, V, O   4 n d^2      lineare in n
    punteggi Q K^T           n^2 d       quadratico in n
    aggregazione A V         n^2 d       quadratico in n
    feed-forward (ratio r)  2 r n d^2    lineare in n

Il termine quadratico domina solo per n > (2 + r) d, cioe' n > 768 con d=128 e
r=4. Le sequenze qui sono molto piu' corte, quindi il guadagno del modello
sparso si misura col contatore di FLOPs invece di dedurlo da (51/4)^2. I
parametri, (4 + 2r) d^2 piu' bias e norme, non dipendono da n.
"""

from __future__ import annotations

import math

import torch
from einops import rearrange
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    """Self-attention multi-testa, scritta per esteso.

    Lo scaling 1/sqrt(d_head) tiene la varianza dei punteggi a 1: senza, la
    softmax satura e il gradiente svanisce al crescere della dimensione.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} non divisibile per num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Q, K, V in una sola proiezione: un GEMM invece di tre
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        qkv = self.qkv(x)
        q, k, v = rearrange(
            qkv, "b n (three h d) -> three b h n d", three=3, h=self.num_heads
        )

        scores = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale   # [B, H, N, N]
        attn = self.attn_drop(scores.softmax(dim=-1))
        ctx = torch.einsum("bhij,bhjd->bhid", attn, v)                # [B, H, N, d]

        ctx = rearrange(ctx, "b h n d -> b n (h d)")
        return self.proj_drop(self.out(ctx))


class FeedForward(nn.Module):
    """MLP posizionale con GELU, la scelta di entrambi i paper."""

    def __init__(self, dim: int, ratio: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = ratio * dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderBlock(nn.Module):
    """Blocco pre-norm: x + Attn(LN(x)), poi x + FFN(LN(x)).

    Il post-norm di Vaswani et al. 2017 richiede warmup del learning rate per
    non divergere; col pre-norm il ramo residuo resta pulito.
    """

    def __init__(self, dim: int, num_heads: int, ratio: int = 4,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    """Stack di blocchi pre-norm, con LayerNorm finale.

    d=128 x 8 sullo sparso, dichiarato dal paper; d=224 x 8 sul denso, dove la
    larghezza e' ricostruita (README, sez. 3).
    """

    def __init__(self, dim: int, depth: int, num_heads: int, ratio: int = 4,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [EncoderBlock(dim, num_heads, ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.norm(x)
