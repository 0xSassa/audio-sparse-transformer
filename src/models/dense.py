"""Dense-transformer: il baseline del confronto controllato del paper.

Stessa struttura del modello sparso, ma i token sono i frame temporali della
early convolution invece dei token latenti campionati. Il paper ne dichiara solo
4,80 M parametri e 0,645 GFLOP (Tab. 2), quindi la configurazione e' ricostruita
con quei due numeri come unici vincoli, da cui la parametrizzazione spinta della
classe.

`seq_mode`, cioe' come [B, C, F, T] diventa una sequenza:

    "flatten"    ogni istante porta tutti i suoi bin: N = T, dim = C*F
    "pool_freq"  media sull'asse frequenza:           N = T, dim = C

Si adotta "flatten", la tokenizzazione di KWT sullo stesso dataset. Nessuna
codifica posizionale: [ASSUNZIONE] per simmetria col modello sparso.
"""

from __future__ import annotations

from typing import Literal

import torch
from einops import rearrange
from torch import nn

from .frontend import ChannelReading, EarlyConv
from .transformer import TransformerEncoder

SeqMode = Literal["flatten", "pool_freq"]


class DenseAudioTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int = 35,
        n_mels: int = 64,
        n_frames: int = 101,
        *,
        # i default sono la configurazione adottata
        # (`configs/dense_flatten.yaml`), da cui vengono i risultati riportati
        channel_reading: ChannelReading = "c96",
        pool_stride: tuple[int, int] = (2, 1),
        seq_mode: SeqMode = "flatten",
        dim: int = 224,
        depth: int = 8,
        num_heads: int = 7,
        ffn_ratio: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.seq_mode = seq_mode

        self.frontend = EarlyConv(
            channel_reading=channel_reading, pool_stride=pool_stride,
        )
        shape = self.frontend.output_shape(n_mels, n_frames)
        self.feature_shape = shape

        if seq_mode == "flatten":
            token_dim, seq_len = shape.channels * shape.freq, shape.time
        elif seq_mode == "pool_freq":
            token_dim, seq_len = shape.channels, shape.time
        else:
            raise ValueError(f"seq_mode sconosciuto: {seq_mode}")
        self.seq_len = seq_len

        self.proj = nn.Linear(token_dim, dim)
        self.encoder = TransformerEncoder(dim, depth, num_heads, ffn_ratio, dropout)
        self.head = nn.Linear(dim, num_classes)

    def to_sequence(self, feat: torch.Tensor) -> torch.Tensor:
        """[B, C, F, T] -> [B, N, token_dim]."""
        if self.seq_mode == "flatten":
            return rearrange(feat, "b c f t -> b t (c f)")
        return rearrange(feat.mean(dim=2), "b c t -> b t c")

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """spec: [B, 1, n_mels, n_frames] -> logits [B, num_classes]."""
        feat = self.frontend(spec)
        seq = self.proj(self.to_sequence(feat))
        seq = self.encoder(seq)
        return self.head(seq.mean(dim=1))
