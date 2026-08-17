"""Front-end tempo-frequenza.

>>> ASSUNZIONE DOCUMENTATA <<<
Kavaki & Mandel NON dichiarano alcun parametro dello spettrogramma: ne'
il tipo (mel o lineare), ne' n_fft, hop, numero di bin. E' il parametro
libero piu' impattante dell'intera riproduzione.

I default qui sotto sono la configurazione standard per keyword spotting
su Speech Commands (finestra 25 ms, hop 10 ms, 64 bin mel), che produce
101 frame per una clip da 1 s. Vanno DICHIARATI nel README e trattati
come iperparametro, non come dato di fatto.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio.transforms as T


class LogMelSpectrogram(nn.Module):
    """Waveform [B, 16000] -> log-mel [B, 1, n_mels, n_frames].

    Il modulo vive sul device del modello: si chiama sul batch gia' in GPU.
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
        normalize: bool = True,
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
        self.normalize = normalize

    @property
    def n_mels(self) -> int:
        return self.mel.n_mels

    def n_frames(self, n_samples: int) -> int:
        return n_samples // self.mel.hop_length + 1

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        if wave.dim() != 2:
            raise ValueError(f"atteso [B, T], ricevuto {tuple(wave.shape)}")
        spec = self.to_db(self.mel(wave))          # [B, n_mels, n_frames]
        if self.normalize:
            # Normalizzazione per-campione: rende il modello insensibile
            # al livello di registrazione, che in SC-V2 varia molto.
            mean = spec.mean(dim=(1, 2), keepdim=True)
            std = spec.std(dim=(1, 2), keepdim=True).clamp_min(1e-5)
            spec = (spec - mean) / std
        return spec.unsqueeze(1)                    # [B, 1, n_mels, n_frames]
