"""Augmentation a livello di batch, su GPU.

Kavaki & Mandel dichiarano di usare "mixing and phasemix augmentation [1]",
cioe' due delle tecniche di label-mixing introdotte da EAT (Gazneli et al.
2022). Entrambe mescolano DUE campioni e producono un target morbido, per
cui la loss e' BCE e non cross-entropy: con target mixati la CE su softmax
non e' piu' il log-likelihood del modello generativo sottostante.

>>> ASSUNZIONE DOCUMENTATA <<<
EAT descrive PhaseMix in una riga ("mixes phase components while preserving
amplitude") senza formula. L'implementazione qui sotto interpola le fasi e
mescola le ampiezze. Prima di riportare numeri definitivi, confronta con il
codice ufficiale di EAT (github.com/Alibaba-MIIL/AudioClassfication) e
dichiara nel README quale interpretazione hai usato.
"""

from __future__ import annotations

import torch


def _lam(batch_size: int, alpha: float, device: torch.device) -> torch.Tensor:
    """Coefficienti di mixing ~ Beta(alpha, alpha), uno per campione."""
    beta = torch.distributions.Beta(alpha, alpha)
    return beta.sample((batch_size,)).to(device)


def one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return torch.zeros(
        labels.shape[0], num_classes, device=labels.device, dtype=torch.float32
    ).scatter_(1, labels.unsqueeze(1), 1.0)


def mixing(
    wave: torch.Tensor, targets: torch.Tensor, alpha: float = 0.2
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mixup standard sulla forma d'onda.

    wave    [B, T] float
    targets [B, C] one-hot o gia' morbido
    """
    perm = torch.randperm(wave.shape[0], device=wave.device)
    lam = _lam(wave.shape[0], alpha, wave.device)
    lam_w = lam.view(-1, 1)
    mixed = lam_w * wave + (1.0 - lam_w) * wave[perm]
    mixed_t = lam.view(-1, 1) * targets + (1.0 - lam.view(-1, 1)) * targets[perm]
    return mixed, mixed_t


def phasemix(
    wave: torch.Tensor, targets: torch.Tensor, alpha: float = 0.2
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mixing nel dominio della frequenza: ampiezze mescolate, fasi interpolate.

    L'interpolazione delle fasi e' fatta sui vettori unitari e^{j*phi} e non
    sugli angoli, per evitare il problema del wrapping a +/- pi.
    """
    perm = torch.randperm(wave.shape[0], device=wave.device)
    lam = _lam(wave.shape[0], alpha, wave.device).view(-1, 1)

    spec_a = torch.fft.rfft(wave, dim=-1)
    spec_b = torch.fft.rfft(wave[perm], dim=-1)

    amp = lam * spec_a.abs() + (1.0 - lam) * spec_b.abs()

    unit_a = spec_a / spec_a.abs().clamp_min(1e-12)
    unit_b = spec_b / spec_b.abs().clamp_min(1e-12)
    unit = lam * unit_a + (1.0 - lam) * unit_b
    unit = unit / unit.abs().clamp_min(1e-12)

    mixed = torch.fft.irfft(amp * unit, n=wave.shape[-1], dim=-1)
    mixed_t = lam * targets + (1.0 - lam) * targets[perm]
    return mixed, mixed_t


AUGMENTATIONS = {"mixing": mixing, "phasemix": phasemix}


def apply_augmentations(
    wave: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    *,
    names: tuple[str, ...] = ("mixing", "phasemix"),
    alpha: float = 0.2,
    prob: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sceglie a caso una delle augmentation attive, o nessuna.

    Ritorna sempre target morbidi [B, C]: il training loop usa BCE in
    entrambi i casi, cosi' il percorso di loss e' unico.
    """
    targets = one_hot(labels, num_classes)
    if not names or torch.rand(()) >= prob:
        return wave, targets
    idx = int(torch.randint(len(names), ()))
    return AUGMENTATIONS[names[idx]](wave, targets, alpha)
