"""Early convolution — il ponte fra spettrogramma e transformer.

RUOLO NEL MODELLO. Kavaki & Mandel (sez. 3.2) sono espliciti sul perche'
questo blocco esiste:

    "Gradient computation for region adjustment through bilinear
     interpolation can be very noisy due to local and limited sampling
     points. Therefore, directly sampling from the spectrogram is not
     effective. Instead, we apply the spectrogram to early convolution,
     so that sampling is performed on the output of the early convolution."

Non serve solo a ridurre la risoluzione: serve a rendere ADDESTRABILE il
campionamento sparso. Il gradiente rispetto alle coordinate e' una
differenza finita fra celle adiacenti (manuale, eq. 7.6); su una superficie
ruvida come uno spettrogramma grezzo e' rumore, dopo qualche convoluzione
diventa un segnale di direzione affidabile.

>>> LA SEQUENZA SEGUE IL CODICE UFFICIALE DI SPARSEFORMER <<<
    github.com/showlab/sparseformer -> imagenet/models/sparseformer.py

        conv 7x7 stride 2 pad 3 (bias=True) -> ReLU -> maxpool 3x3 -> LayerNorm(C)

Il paper di Kavaki & Mandel descrive "conv -> ReLU -> max pooling" senza
normalizzazione, e SparseFormer conferma: nessuna BatchNorm dentro lo stem,
ma una LayerNorm sui CANALI applicata DOPO il pooling. Una nostra versione
precedente inseriva una BatchNorm fra conv e ReLU: era una deviazione su
tre punti (layer sbagliato, posizione sbagliata, bias della conv disattivato).

>>> AMBIGUITA' DEL PAPER <<<
La sez. 3.2 si contraddice: "extract 96 dimensional feature" ma "196 kernels
with a size 7x7 and a stride 2". Il modulo supporta le tre letture possibili
via `channel_reading`, cosi' la scelta si fa sul conteggio dei parametri
(4.80 M dichiarati per il denso) invece che a intuito.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

ChannelReading = Literal["c96", "c196", "c196_proj96"]


@dataclass(frozen=True)
class FrontendShape:
    """Forma dell'uscita, calcolabile senza istanziare nulla."""

    channels: int
    freq: int
    time: int

    @property
    def num_frames(self) -> int:
        return self.time

    @property
    def frame_dim(self) -> int:
        return self.channels * self.freq

    def __str__(self) -> str:
        return f"[C={self.channels}, F={self.freq}, T={self.time}]"


def _conv_out(size: int, kernel: int, stride: int, padding: int) -> int:
    return (size + 2 * padding - kernel) // stride + 1


class ChannelLayerNorm(nn.Module):
    """LayerNorm sull'asse dei canali di un tensore [B, C, H, W].

    Equivale al `layer_norm_by_dim(x, 1)` di SparseFormer: normalizza ogni
    posizione spaziale rispetto ai propri canali. Da non confondere con la
    BatchNorm2d, che normalizza ogni canale rispetto al batch e alle
    posizioni — statistiche diverse e dipendenti dal batch.
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

    >>> AMBIGUITA' DEL PAPER <<<
    La sez. 3.2 si contraddice: "extract 96 dimensional feature" ma "196
    kernels with a size 7x7 and a stride 2". Il modulo supporta tutte e tre
    le letture possibili via `channel_reading`, cosi' la scelta si e' fatta
    sul conteggio dei parametri invece che a intuito.
    """

    def __init__(
        self,
        in_channels: int = 1,
        channel_reading: ChannelReading = "c96",
        num_conv_layers: int = 1,
        pool_stride: tuple[int, int] = (2, 1),
    ) -> None:
        """`pool_stride` e' (frequenza, tempo).

        (2, 2) dimezza entrambi gli assi -> 101 frame diventano 26.
        (2, 1) dimezza solo la frequenza  -> restano 51 frame.

        Il paper dice solo "max pooling", senza stride. La ricerca sulle
        configurazioni mostra che deve essere (2, 1) — che e' il default —
        perche' con (2, 2) il costo si ferma al 70% di quello dichiarato e
        nessun altro grado di liberta' recupera il fattore mancante.
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
        # `num_conv_layers` > 1 esiste per testare l'ipotesi sul plurale di
        # "convolutional layers". Confutata: due strati portano i FLOPs a
        # +164% invece che a +11%. Tenuto per documentazione.
        for _ in range(num_conv_layers - 1):
            layers += [
                nn.Conv2d(conv_channels, conv_channels, 3, stride=1,
                          padding=1, bias=True),
                nn.ReLU(inplace=True),
            ]
        layers.append(nn.MaxPool2d(3, stride=pool_stride, padding=1))

        # Terza lettura dell'ambiguita': 196 kernel, poi proiezione a 96.
        if out_channels != conv_channels:
            layers.append(nn.Conv2d(conv_channels, out_channels, 1))

        # Normalizzazione DOPO il pooling, come in SparseFormer.
        layers.append(ChannelLayerNorm(out_channels))

        self.body = nn.Sequential(*layers)

    @staticmethod
    def _resolve_channels(reading: ChannelReading) -> tuple[int, int]:
        """(canali delle conv, canali in uscita)."""
        if reading == "c96":
            return 96, 96          # "96 dimensional feature", il 196 e' un refuso
        if reading == "c196":
            return 196, 196        # "196 kernels", il 96 e' un refuso
        if reading == "c196_proj96":
            return 196, 96         # entrambi veri: 196 kernel, poi 1x1 a 96
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
