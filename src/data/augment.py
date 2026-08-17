"""Augmentation di label-mixing, a livello di batch e su GPU.

Kavaki & Mandel dichiarano di usare "mixing and phasemix augmentation [1]",
cioe' due delle tecniche introdotte da EAT (Gazneli et al. 2022). EAT le
descrive in una riga ciascuna, senza formule.

>>> QUESTE IMPLEMENTAZIONI SEGUONO IL CODICE UFFICIALE DI EAT <<<
    github.com/Alibaba-MIIL/AudioClassfication  ->  datasets/batch_augs.py
Consultato per sciogliere l'ambiguita' delle descrizioni testuali; nessuna
riga copiata. Una prima versione scritta dalla sola descrizione del paper
era sbagliata su quasi tutti i dettagli (vedi il diario, 18/08), il che e'
un buon promemoria di quanto sia rischioso dedurre un metodo da una frase.

DUE COSE CHE NON SONO QUELLO CHE SEMBRANO

1. `mixing` NON e' il mixup di Zhang et al. E' il "between-class learning"
   di Tokozume et al. 2017 (citato da EAT come [14]): i due segnali vengono
   bilanciati in base al loro LIVELLO SONORO e il risultato e' rinormalizzato
   in energia. Ha senso fisico: sommare due suoni a livelli molto diversi con
   pesi uguali significa che uno copre l'altro.

2. `phasemix` PRESERVA l'ampiezza di un campione e prende solo la fase
   dell'altro. La frase di EAT — "mixes phase components while preserving
   amplitude" — va letta cosi': l'ampiezza non si mescola affatto.
"""

from __future__ import annotations

import torch

# Parametri della STFT usata da phasemix. EAT non li dichiara: adottiamo gli
# stessi del front-end log-mel (25 ms / 10 ms) per coerenza.  [ASSUNZIONE]
STFT_N_FFT = 400
STFT_HOP = 160

_WINDOWS: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}


def _hann(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (device, dtype)
    if key not in _WINDOWS:
        _WINDOWS[key] = torch.hann_window(STFT_N_FFT, device=device, dtype=dtype)
    return _WINDOWS[key]


# Soglia oltre la quale il secondo suono e' considerato troppo debole per
# meritare la propria etichetta. Valore preso dal codice di EAT.
LAM_LABEL_THRESHOLD = 0.9


def one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return torch.zeros(
        labels.shape[0], num_classes, device=labels.device, dtype=torch.float32
    ).scatter_(1, labels.unsqueeze(1), 1.0)


def mix_targets(
    targets: torch.Tensor, targets_other: torch.Tensor, lam: torch.Tensor
) -> torch.Tensor:
    """Costruisce il target di un batch mixato, come fa EAT nel ramo BCE.

        target_shuffled = one_hot(y2) * (lam < 0.9)
        target          = clamp(one_hot(y1) + target_shuffled, max=1)

    NON e' un target morbido pesato per lam. E' un target MULTI-HOT: se la
    miscela contiene udibilmente entrambe le parole, entrambe le classi
    valgono 1. E' la formulazione naturale per la BCE, che tratta ogni
    classe come una domanda binaria indipendente "questa parola c'e'?".

    lam entra come SOGLIA, non come peso: se lam >= 0.9 il secondo suono e'
    troppo debole e la sua etichetta viene scartata. Il clamp serve al caso
    in cui i due campioni appartengano alla stessa classe.

    Una nostra versione precedente usava target morbidi lam*y1+(1-lam)*y2.
    E' la semantica del ramo 'ce' di EAT, non di quello 'bce' che usiamo.
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

    Il peso dei DATI e' `p`, corretto per il livello; il peso delle
    ETICHETTE e' `lam`, cioe' il rapporto percettivo voluto. La divisione
    per sqrt(p^2+(1-p)^2) mantiene costante l'energia attesa della miscela,
    evitando che il modello usi il volume come indizio dell'augmentation.

    lam ~ Uniform[lam_min, 1] come nel codice EAT (non una Beta).
    `lam` puo' essere forzato per rendere deterministici i test.
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
        X_new  = |X1| * exp(j ph)                 <- ampiezza di x1, intatta
        out    = iSTFT(X_new)

    Il peso dell'etichetta e' `lam*0.5 + 0.5`, quindi sta in [0.5, 1]:
    mescolare la sola fase altera il segnale meno che mescolare tutto, e
    l'etichetta resta piu' vicina all'originale. E' un dettaglio che dalla
    descrizione testuale sarebbe stato impossibile indovinare.

    Nota 1: le fasi sono interpolate sugli ANGOLI, come fa EAT. E' una
    scelta discutibile — l'interpolazione lineare attraversa la
    discontinuita' a +/-pi — ma qui interessa la fedelta' al riferimento,
    non il miglioramento del metodo.

    Nota 2 — CONSISTENZA DELLA STFT. "Ampiezza preservata" vale sulla
    MATRICE STFT modificata, non sul segnale ricostruito. Una matrice
    complessa arbitraria in generale NON e' la STFT di alcun segnale reale:
    con finestre sovrapposte (qui al 60%) i frame vicini si vincolano a
    vicenda, e alterare le fasi rende la matrice inconsistente. La iSTFT
    fa overlap-add, cioe' proietta sul sottospazio delle STFT consistenti,
    e quella proiezione cambia anche i moduli — misurato: circa il 34% in
    errore relativo medio. Non e' un difetto dell'implementazione ma una
    proprieta' del metodo, e contribuisce all'effetto di augmentation.

    `lam` puo' essere forzato per rendere deterministici i test.
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

    # lam efficace per l'etichetta: rimappato in [0.5, 1] perche' mescolare
    # la sola fase altera il segnale meno che mescolare tutto. Con la soglia
    # a 0.9 questo significa che la seconda etichetta sopravvive quando
    # lam < 0.8, cioe' nell'80% dei casi.
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

    La logica di dispatch segue `BatchAugs.__call__` di EAT:

        se augs non vuoto  e  rand() <= mix_ratio  e  epoch > epoch_mix:
            scegli UNA augmentation a caso e applicala all'INTERO batch

    `mix_ratio` e' quindi la probabilita' di mescolare (default 1, cioe'
    SEMPRE — non 0.5 come avevamo assunto), e `epoch_mix` una soglia di
    epoca sotto la quale il mixing e' disattivato, utile per far partire il
    training su dati puliti. Il default `epoch_mix=-1` con `epoch` grande
    tiene il mixing sempre attivo.

    Restituisce SEMPRE target [B, C], anche senza augmentation: il percorso
    di loss e' unico e la BCE non ha rami condizionali.
    """
    targets = one_hot(labels, num_classes)
    if not names or epoch <= epoch_mix or float(torch.rand(())) > mix_ratio:
        return wave, targets
    idx = int(torch.randint(len(names), ()))
    return AUGMENTATIONS[names[idx]](wave, targets)
