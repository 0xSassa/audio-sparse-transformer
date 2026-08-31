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

E' la scelta piu' importante della ricostruzione: vale 2,92 punti, e con
"pool_freq" il claim del paper si riproduce mentre con "flatten" no. Vedi
l'intestazione di configs/dense_flatten.yaml.

>>> NIENTE CODIFICA POSIZIONALE <<<
Il paper la dichiara assente per il modello SPARSO — i token latenti non
hanno un ordine intrinseco. Sul DENSO tace, e i frame temporali un ordine
ce l'hanno: senza codifica il transformer e' invariante a permutazioni del
tempo. E' quindi una nostra ASSUNZIONE, presa per simmetria col modello
sparso. Testarla nelle due varianti era previsto e non e' stato fatto: e'
un esperimento mancante, non una scelta misurata.
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
        # I default sono la CONFIGURAZIONE ADOTTATA, quella di
        # `configs/dense_flatten.yaml` da cui vengono i risultati riportati.
        # Le alternative esplorate restano negli YAML, che e' dove sono
        # documentate: `configs/dense.yaml` tiene `pool_freq` come ablation.
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
