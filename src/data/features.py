"""Front-end tempo-frequenza.

Il paper non dichiara alcun parametro dello spettrogramma. Finestra 25 ms e
hop 10 ms sono la convenzione della letteratura su questo dataset (AST:
"25 ms Hamming window every 10 ms"; KWT: 30 ms / 10 ms), e su un secondo di
audio danno 101 frame. I 64 bin mel sono un'assunzione, fra i 40 di KWT e i
128 di AST; dopo lo stem danno F=16. Nessuna normalizzazione per campione,
come nel paper e negli ancestor.
"""

from __future__ import annotations

import torch
import torchaudio.transforms as T
from torch import nn


class LogMelSpectrogram(nn.Module):
    """Waveform [B, 16000] -> log-mel [B, 1, n_mels, n_frames].

    Vive sul device del modello: si chiama sul batch gia' in GPU.
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        n_fft: int = 400,        # 25 ms
        hop_length: int = 160,   # 10 ms
        n_mels: int = 64,
        f_min: float = 20.0,
        f_max: float = 7_600.0,
        top_db: float = 80.0,
    ) -> None:
        super().__init__()
        self.mel = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            power=2.0,
            center=True,
        )
        self.to_db = T.AmplitudeToDB(stype="power", top_db=top_db)

    @property
    def n_mels(self) -> int:
        return self.mel.n_mels

    def n_frames(self, n_samples: int) -> int:
        return n_samples // self.mel.hop_length + 1

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        if wave.dim() != 2:
            raise ValueError(f"atteso [B, T], ricevuto {tuple(wave.shape)}")
        spec = self.to_db(self.mel(wave))          # [B, n_mels, n_frames]
        return spec.unsqueeze(1)                    # [B, 1, n_mels, n_frames]
