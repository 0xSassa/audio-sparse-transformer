"""Dense-transformer: il baseline del confronto controllato di Kavaki & Mandel.

Stessa struttura del modello sparso, ma i token del transformer sono i
FRAME TEMPORALI prodotti dalla early convolution, non i token latenti
campionati. Il paper lo usa come termine di paragone diretto (96.67 %,
4.80 M parametri, 0.645 G FLOPs) e da esso ricava il -91.47 % di costo.

>>> AMBIGUITA' <<<
Del dense-transformer il paper NON dichiara alcun iperparametro: solo il
totale di parametri e FLOPs. La configurazione va quindi RICOSTRUITA
usando quei due numeri come vincoli. Da qui la parametrizzazione spinta di
questa classe: `seq_mode`, `pool_stride`, `dim`, `depth`, `ffn_ratio` sono
gradi di liberta' su cui si cerca, non scelte gia' fatte.

`seq_mode` — come una feature map [B, C, F, T] diventa una sequenza:

    "flatten"    ogni istante porta tutti i suoi bin: N = T, dim = C*F.
                 Conserva tutto ma la proiezione iniziale e' costosa
                 (C*F*dim parametri).
    "pool_freq"  media sull'asse frequenza: N = T, dim = C.
                 Proiezione economica; l'informazione spettrale resta nei
                 canali della convoluzione.
    "patches2d"  ogni cella (f, t) e' un token: N = F*T, dim = C.
                 Massima risoluzione, sequenza molto piu' lunga.
"""

from __future__ import annotations

from typing import Literal

import torch
from einops import rearrange
from torch import nn

from .frontend import ChannelReading, EarlyConv, NormKind, StemKind
from .transformer import PosEncoding, PositionalEncoding, TransformerEncoder

SeqMode = Literal["flatten", "pool_freq", "patches2d"]


class DenseAudioTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int = 35,
        n_mels: int = 64,
        n_frames: int = 101,
        *,
        channel_reading: ChannelReading = "c96",
        stem: StemKind = "paper",
        num_conv_layers: int = 1,
        norm: NormKind = "layernorm",
        pool_stride: tuple[int, int] = (2, 2),
        seq_mode: SeqMode = "pool_freq",
        dim: int = 224,
        depth: int = 8,
        num_heads: int = 8,
        ffn_ratio: int = 4,
        pos_encoding: PosEncoding = "none",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.seq_mode = seq_mode

        self.frontend = EarlyConv(
            channel_reading=channel_reading, stem=stem, pool_stride=pool_stride,
            num_conv_layers=num_conv_layers, norm=norm,
        )
        shape = self.frontend.output_shape(n_mels, n_frames)
        self.feature_shape = shape

        if seq_mode == "flatten":
            token_dim, seq_len = shape.channels * shape.freq, shape.time
        elif seq_mode == "pool_freq":
            token_dim, seq_len = shape.channels, shape.time
        elif seq_mode == "patches2d":
            token_dim, seq_len = shape.channels, shape.freq * shape.time
        else:
            raise ValueError(f"seq_mode sconosciuto: {seq_mode}")
        self.seq_len = seq_len

        self.proj = nn.Linear(token_dim, dim)
        self.pos = PositionalEncoding(dim, pos_encoding, max_len=seq_len)
        self.encoder = TransformerEncoder(dim, depth, num_heads, ffn_ratio, dropout)
        self.head = nn.Linear(dim, num_classes)

    def to_sequence(self, feat: torch.Tensor) -> torch.Tensor:
        """[B, C, F, T] -> [B, N, token_dim]."""
        if self.seq_mode == "flatten":
            return rearrange(feat, "b c f t -> b t (c f)")
        if self.seq_mode == "pool_freq":
            return rearrange(feat.mean(dim=2), "b c t -> b t c")
        return rearrange(feat, "b c f t -> b (f t) c")

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """spec: [B, 1, n_mels, n_frames] -> logits [B, num_classes]."""
        feat = self.frontend(spec)
        seq = self.proj(self.to_sequence(feat))
        seq = self.encoder(self.pos(seq))
        return self.head(seq.mean(dim=1))
