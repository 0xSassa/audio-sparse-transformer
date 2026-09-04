"""Augmentation di label-mixing, a livello di batch e su GPU.

Le due tecniche dichiarate dal paper, entrambe di EAT (Gazneli et al. 2022).
Implementate seguendo il codice ufficiale (datasets/batch_augs.py), consultato
perche' le descrizioni testuali sono ambigue; nessuna riga copiata.

Due cose che non sono quello che sembrano: `mixing` non e' il mixup di Zhang et
al. ma il between-class learning di Tokozume et al. 2017, con i due segnali
bilanciati per livello sonoro; `phasemix` preserva l'ampiezza di un campione e
prende solo la fase dell'altro.
"""

from __future__ import annotations

import torch

# STFT di phasemix. EAT non la dichiara: [ASSUNZIONE] stessi parametri del
# front-end log-mel, 25 ms / 10 ms.
STFT_N_FFT = 400
STFT_HOP = 160

_WINDOWS: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}


def _hann(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (device, dtype)
    if key not in _WINDOWS:
        _WINDOWS[key] = torch.hann_window(STFT_N_FFT, device=device, dtype=dtype)
    return _WINDOWS[key]


# Oltre questa soglia il secondo suono e' troppo debole per meritare la propria
# etichetta. Valore preso dal codice di EAT.
LAM_LABEL_THRESHOLD = 0.9


def one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return torch.zeros(
        labels.shape[0], num_classes, device=labels.device, dtype=torch.float32
    ).scatter_(1, labels.unsqueeze(1), 1.0)


def mix_targets(
    targets: torch.Tensor, targets_other: torch.Tensor, lam: torch.Tensor
) -> torch.Tensor:
    """Target di un batch mixato, come fa EAT nel ramo BCE.

        target = clamp(one_hot(y1) + one_hot(y2) * (lam < 0.9), max=1)

    Multi-hot, non morbido: `lam` entra come soglia e non come peso. Il clamp
    copre il caso di due campioni della stessa classe.
    """
    keep = (lam < LAM_LABEL_THRESHOLD).to(targets.dtype).unsqueeze(1)
    return torch.clamp(targets + targets_other * keep, max=1.0)


def mixing(
    wave: torch.Tensor, targets: torch.Tensor, *, lam_min: float = 0.1,
    lam: torch.Tensor | float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Between-class mixing con correzione di potenza (EAT, `mixup`).

        G  = 10 log10 E[x^2]                        livello in dB
        p  = 1 / (1 + 10^((G1-G2)/20) * (1-lam)/lam)
        out = (x1 p + x2 (1-p)) / sqrt(p^2 + (1-p)^2)

    I dati pesano `p`, corretto per il livello; le etichette pesano `lam`, il
    rapporto percettivo voluto. La divisione tiene costante l'energia attesa,
    cosi' il volume non diventa un indizio dell'augmentation.

    lam ~ Uniform[lam_min, 1] come in EAT, non una Beta.
    """
    b = wave.shape[0]
    perm = torch.randperm(b, device=wave.device)
    if lam is None:
        lam = lam_min + (1.0 - lam_min) * torch.rand(b, device=wave.device)
    else:
        lam = torch.as_tensor(lam, device=wave.device, dtype=wave.dtype).expand(b)

    power = 10.0 * torch.log10(wave.pow(2).mean(dim=1).clamp_min(1e-12))
    ratio = torch.pow(10.0, (power - power[perm]) / 20.0)
    p = 1.0 / (1.0 + ratio * (1.0 - lam) / lam)

    pw = p.view(-1, 1)
    mixed = (wave * pw + wave[perm] * (1.0 - pw)) / torch.sqrt(
        pw.pow(2) + (1.0 - pw).pow(2)
    )
    return mixed, mix_targets(targets, targets[perm], lam)


def phasemix(
    wave: torch.Tensor, targets: torch.Tensor, *,
    lam: torch.Tensor | float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mixing della sola fase, ad ampiezza preservata (EAT, `phasemix`).

        X      = STFT(x1)
        ph     = lam * angle(X1) + (1-lam) * angle(X2)
        X_new  = |X1| * exp(j ph)
        out    = iSTFT(X_new)

    Le fasi sono interpolate sugli angoli, come fa EAT, discontinuita' a +/-pi
    compresa. L'ampiezza resta preservata sulla matrice STFT ma non sul segnale
    ricostruito, perche' la iSTFT proietta sul sottospazio delle STFT
    consistenti: e' una proprieta' del metodo, non dell'implementazione.
    """
    b, n = wave.shape
    perm = torch.randperm(b, device=wave.device)
    if lam is None:
        lam = torch.rand(b, device=wave.device)
    else:
        lam = torch.as_tensor(lam, device=wave.device, dtype=wave.dtype).expand(b)
    window = _hann(wave.device, wave.dtype)

    spec = torch.stft(
        wave, STFT_N_FFT, hop_length=STFT_HOP, window=window,
        center=True, return_complex=True,
    )
    phase = torch.angle(spec)
    lam_s = lam.view(-1, 1, 1)
    mixed_phase = lam_s * phase + (1.0 - lam_s) * phase[perm]
    mixed_spec = torch.polar(spec.abs(), mixed_phase)

    mixed = torch.istft(
        mixed_spec, STFT_N_FFT, hop_length=STFT_HOP, window=window,
        center=True, length=n,
    )

    # lam dell'etichetta rimappato in [0.5, 1], come in EAT: mescolare la sola
    # fase altera il segnale meno che mescolare tutto.
    lam_label = lam * 0.5 + 0.5
    return mixed, mix_targets(targets, targets[perm], lam_label)


AUGMENTATIONS = {"mixing": mixing, "phasemix": phasemix}


def apply_augmentations(
    wave: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    *,
    names: tuple[str, ...] = ("mixing", "phasemix"),
    mix_ratio: float = 1.0,
    epoch: int = 10**9,
    epoch_mix: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sceglie a caso una delle augmentation attive, o nessuna.

    Dispatch come in `BatchAugs.__call__` di EAT: se sorteggiata, una sola
    augmentation si applica all'intero batch. `mix_ratio` e' la probabilita' di
    mescolare, `epoch_mix` la soglia sotto cui il mixing resta spento.

    Restituisce sempre target [B, C], anche senza augmentation, cosi' il
    percorso di loss e' unico.
    """
    targets = one_hot(labels, num_classes)
    if not names or epoch <= epoch_mix or float(torch.rand(())) > mix_ratio:
        return wave, targets
    idx = int(torch.randint(len(names), ()))
    return AUGMENTATIONS[names[idx]](wave, targets)
