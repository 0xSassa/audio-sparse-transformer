"""Front-end tempo-frequenza.

>>> ASSUNZIONE DOCUMENTATA <<<
Kavaki & Mandel NON dichiarano alcun parametro dello spettrogramma: ne'
il tipo (mel o lineare), ne' n_fft, hop, numero di bin.

Temevamo fosse il grado di liberta' piu' impattante della riproduzione.
Misurato con una sonda lineare (regressione logistica su log-mel mediati
su 8 finestre, 12k train / 4k val) il parametro risulta quasi PIATTO:

    32 bin 41.60%   40 bin 41.93%   64 bin 42.18%   80 bin 42.40%   128 bin 42.03%

0.8 punti di escursione su un fattore 4 di risoluzione, e curva non
monotona. Adottiamo 64 bin: a 0.22 punti dal migliore (dentro il rumore)
ed e' l'unico valore che dopo lo stem da' F=16, potenza di due, utile alla
geometria del campionamento sparso.

>>> NIENTE NORMALIZZAZIONE PER-CAMPIONE, ED E' UNA MISURA <<<
Avevamo aggiunto un'opzione che portava ogni spettrogramma a media 0 e
deviazione 1, motivata dal fatto che il livello di registrazione in SC-V2
varia di 28.5 dB fra 1o e 99o percentile (fattore 27 in ampiezza). La
premessa e' vera, ma la sonda lineare sui bin mel da' 42.83% SENZA
normalizzazione contro 42.33% CON: il livello assoluto porta un po' di
segnale utile e azzerarlo lo distrugge.

Ne' il paper ne' gli ancestor (SparseFormer, EAT) normalizzano. L'opzione
e' stata rimossa invece che lasciata a False: nessuno dei run riportati
l'ha mai attivata, e una via di codice mai percorsa e' solo un modo di
suggerire una scelta che non e' stata fatta.
"""

from __future__ import annotations

import torch
import torchaudio.transforms as T
from torch import nn


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
