"""Encoder transformer, reimplementato.

Non si usa nn.MultiheadAttention: il progetto chiede una reimplementazione
autonoma e all'orale va spiegato ogni passaggio. Il costo e' trascurabile,
perche' con sequenze da 26 elementi il collo di bottiglia e' il lancio dei
kernel, non la moltiplicazione di matrici.

COMPLESSITA' — la derivazione centrale del progetto.

Per una sequenza di n elementi in dimensione d, un singolo blocco costa:

    proiezioni Q, K, V, O   4 n d^2      lineare in n
    punteggi Q K^T           n^2 d       QUADRATICO in n
    aggregazione A V         n^2 d       QUADRATICO in n
    feed-forward (ratio r)  2 r n d^2    lineare in n

Totale ~ (4 + 2r) n d^2 + 2 n^2 d. Il termine quadratico domina quando
n > (2 + r) d, cioe' — con d = 128 e r = 4 — per n > 768. Le nostre
sequenze sono molto piu' corte, quindi in ASSOLUTO domina il termine
lineare in n.

E' un punto sottile ma importante per capire il paper: passare da n = 26
frame a n = 4 token riduce il termine quadratico di (26/4)^2 = 42 volte,
ma il termine lineare "solo" di 6.5 volte. Il guadagno reale sta in mezzo,
ed e' esattamente quello che misureremo col contatore di FLOPs invece di
dedurlo dalla formula.

Il numero di parametri di un blocco e' invece indipendente da n:
    (4 + 2r) d^2  piu' i bias e le due LayerNorm.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from einops import rearrange
from torch import nn

PosEncoding = Literal["none", "sinusoidal", "learned"]


class MultiHeadSelfAttention(nn.Module):
    """Self-attention multi-testa, scritta per esteso.

    Lo scaling per 1/sqrt(d_head) serve a controllare la varianza dei
    punteggi: se q e k hanno componenti indipendenti a varianza 1, il loro
    prodotto scalare su d_head dimensioni ha varianza d_head. Senza
    riscalare, i logit crescono con la dimensione, la softmax satura e il
    gradiente svanisce.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} non divisibile per num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Una sola proiezione per Q, K, V: e' un'ottimizzazione, non una
        # differenza di modello (un unico GEMM invece di tre).
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
    """MLP posizionale con GELU.

    GELU e non ReLU perche' e' la scelta di entrambi i paper. Vale la pena
    ricordare il legame col dropout visto a lezione (25/10): GELU(x) e'
    x * P(X <= x) con X gaussiana, cioe' il valore atteso di un dropout
    che azzera l'input in modo deterministico in base alla sua ampiezza,
    invece che casualmente.
    """

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

    Vaswani et al. 2017 usa post-norm — LN(x + Attn(x)) — ma richiede
    warmup del learning rate per non divergere. Con il pre-norm il ramo
    residuo resta una strada pulita dall'ingresso all'uscita, i gradienti
    passano senza attraversare la normalizzazione, e si addestra stabile
    anche senza warmup. E' la ragione per cui e' diventato lo standard.
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


class PositionalEncoding(nn.Module):
    """Codifica posizionale, opzionale.

    >>> PUNTO SPERIMENTALE <<<
    Il paper dichiara che il modello SPARSO non usa positional encoding,
    perche' i token latenti non hanno un ordine intrinseco. Sul modello
    DENSO non dice nulla — ma i frame temporali un ordine ce l'hanno, e
    senza codifica il transformer e' invariante a permutazioni del tempo.

    Quanto conti davvero l'ordine temporale per riconoscere una parola
    isolata di un secondo e' una domanda aperta e interessante: la si
    risponde addestrando nelle due varianti.
    """

    def __init__(self, dim: int, kind: PosEncoding = "none",
                 max_len: int = 512) -> None:
        super().__init__()
        self.kind = kind
        if kind == "learned":
            self.pos = nn.Parameter(torch.zeros(1, max_len, dim))
            nn.init.trunc_normal_(self.pos, std=0.02)
        elif kind == "sinusoidal":
            self.register_buffer("pos", self._sinusoidal(max_len, dim),
                                 persistent=False)
        elif kind != "none":
            raise ValueError(f"positional encoding sconosciuto: {kind}")

    @staticmethod
    def _sinusoidal(max_len: int, dim: int) -> torch.Tensor:
        pos = torch.arange(max_len).unsqueeze(1).float()
        i = torch.arange(0, dim, 2).float()
        freq = torch.exp(-math.log(10_000.0) * i / dim)
        pe = torch.zeros(1, max_len, dim)
        pe[0, :, 0::2] = torch.sin(pos * freq)
        pe[0, :, 1::2] = torch.cos(pos * freq)
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "none":
            return x
        return x + self.pos[:, : x.shape[1]]


class TransformerEncoder(nn.Module):
    """Stack di blocchi pre-norm, con LayerNorm finale.

    Attenzione al costo della PROFONDITA': a queste dimensioni la GPU e'
    limitata dal lancio dei kernel, non dal calcolo. Misurato sulla RTX
    5050, d=128 con 24 layer (4.96 M parametri) impiega 73 min per 100
    epoche contro i 39.5 min di d=224 con 8 layer (5.19 M): stessa
    capacita', quasi il doppio del tempo. A parita' di parametri conviene
    la larghezza.
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
