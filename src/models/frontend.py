"""Early convolution: il ponte fra spettrogramma e transformer.

Oltre a ridurre la risoluzione serve a rendere ADDESTRABILE il campionamento
sparso: il gradiente sulle coordinate e' una differenza finita fra celle
adiacenti, rumorosa su uno spettrogramma grezzo (paper, sez. 3.2).

Sequenza come nel codice ufficiale di SparseFormer
(github.com/showlab/sparseformer -> imagenet/models/sparseformer.py):

    conv 7x7 stride 2 pad 3 (bias=True) -> ReLU -> maxpool 3x3 -> LayerNorm(C)

Ambiguita' del paper: "96 dimensional feature" contro "196 kernels".
`channel_reading` espone le due letture; si adotta c96, che e' la lettura di
SparseFormer: "a 7x7 stride-2 convolution, a ReLU, and a 3x3 stride-2 max
pooling to extract initial 96-d image features".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

ChannelReading = Literal["c96", "c196_proj96"]


@dataclass(frozen=True)
class FrontendShape:
    """Forma dell'uscita, calcolabile senza istanziare nulla."""

    channels: int
    freq: int
    time: int

    def __str__(self) -> str:
        return f"[C={self.channels}, F={self.freq}, T={self.time}]"


def _conv_out(size: int, kernel: int, stride: int, padding: int) -> int:
    return (size + 2 * padding - kernel) // stride + 1


class ChannelLayerNorm(nn.Module):
    """LayerNorm sull'asse dei canali di un tensore [B, C, H, W].

    Il `layer_norm_by_dim(x, 1)` di SparseFormer: normalizza ogni posizione
    spaziale rispetto ai propri canali. Non e' una BatchNorm2d, che usa
    statistiche sul batch.
    """

    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class EarlyConv(nn.Module):
    """Spettrogramma [B, 1, F, T] -> feature map [B, C, F', T'].

    Lettura letterale della sez. 3.2, confermata da SparseFormer:
    conv 7x7 stride 2 con bias -> ReLU -> max pooling -> LayerNorm sui canali.
    """

    def __init__(
        self,
        in_channels: int = 1,
        channel_reading: ChannelReading = "c96",
        pool_stride: tuple[int, int] = (2, 1),
    ) -> None:
        """`pool_stride` e' (frequenza, tempo).

        (2, 2) dimezza entrambi gli assi -> 101 frame diventano 26.
        (2, 1) dimezza solo la frequenza  -> restano 51 frame.

        Il paper dice solo "max pooling", senza stride. SparseFormer usa
        stride 2 su entrambi gli assi, che su un'immagine e' naturale perche'
        le due dimensioni sono omogenee. Una mappa tempo-frequenza non lo e':
        il tempo e' l'asse lungo cui il modello denso costruisce la propria
        sequenza, e su clip di un secondo dimezzarlo lascerebbe 26 frame,
        cioe' un passo di 40 ms. Il pooling agisce quindi sulla sola
        frequenza.
        """
        super().__init__()
        self.channel_reading = channel_reading
        self.pool_stride = pool_stride

        conv_channels, out_channels = self._resolve_channels(channel_reading)
        self.conv_channels = conv_channels
        self.out_channels = out_channels

        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, conv_channels, 7, stride=2, padding=3,
                      bias=True),
            nn.ReLU(inplace=True),
        ]
        # UN SOLO strato convolutivo, come SparseFormer: il plurale del
        # paper ("convolutional layers") descrive la sequenza conv+ReLU+pool,
        # non piu' convoluzioni.
        layers.append(nn.MaxPool2d(3, stride=pool_stride, padding=1))

        # seconda lettura dell'ambiguita': 196 kernel, poi proiezione a 96
        if out_channels != conv_channels:
            layers.append(nn.Conv2d(conv_channels, out_channels, 1))

        # normalizzazione DOPO il pooling, come in SparseFormer
        layers.append(ChannelLayerNorm(out_channels))

        self.body = nn.Sequential(*layers)

    @staticmethod
    def _resolve_channels(reading: ChannelReading) -> tuple[int, int]:
        """(canali delle conv, canali in uscita)."""
        if reading == "c96":
            return 96, 96          # "96 dimensional feature", il 196 e' un refuso
        if reading == "c196_proj96":
            return 196, 96         # entrambi i numeri veri: 196 kernel, poi 1x1
        raise ValueError(f"lettura sconosciuta: {reading}")

    def output_shape(self, freq: int, time: int) -> FrontendShape:
        """Forma dell'uscita senza eseguire il forward."""
        f = _conv_out(freq, 7, 2, 3)
        t = _conv_out(time, 7, 2, 3)
        f = _conv_out(f, 3, self.pool_stride[0], 1)
        t = _conv_out(t, 3, self.pool_stride[1], 1)
        return FrontendShape(channels=self.out_channels, freq=f, time=t)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        if spec.dim() != 4:
            raise ValueError(f"atteso [B, 1, F, T], ricevuto {tuple(spec.shape)}")
        return self.body(spec)
