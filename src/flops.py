"""Contatore di FLOPs — l'infrastruttura di misura della Fase 2.

Il claim da riprodurre e' -85.25% di FLOPs rispetto a EAT-S. Un contatore
scritto DOPO il modello tende a essere costruito per confermare il risultato
atteso: per questo si scrive prima, e si usano due strumenti indipendenti.

TRE AVVERTENZE, tutte capaci da sole di invalidare il confronto.

1. FLOPs != MAC. fvcore conta MAC ma li chiama "flops"; il contatore nativo
   di torch conta le moltiplicazioni-addizioni come 2 operazioni. I due
   numeri differiscono quindi di circa 2x. Il paper non dichiara quale
   convenzione usi.

   CALIBRAZIONE: implementa EAT-S, misuralo con entrambi i contatori e vedi
   quale dei due si avvicina a 0.373 G. Quella e' la convenzione del paper.
   Finche' non l'hai fatto, confronta i TUOI modelli fra loro, mai coi
   numeri pubblicati.

2. `grid_sample` (l'interpolazione bilineare del campionamento sparso) NON
   e' contata da nessuno dei due strumenti: e' un'op di gather+lerp senza
   matmul. Va aggiunta a mano — vedi `grid_sample_flops()`.

3. Misura sempre su UN input da 1 secondo, in inferenza, con `eval()` e
   `no_grad()`. Il training costa circa 3x, ma non e' quello che si riporta.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class FlopReport:
    native_flops: float | None
    fvcore_macs: float | None
    manual_extra_flops: float
    params: int

    @property
    def native_total(self) -> float | None:
        if self.native_flops is None:
            return None
        return self.native_flops + self.manual_extra_flops

    def __str__(self) -> str:
        lines = [f"parametri            : {self.params / 1e6:8.3f} M"]
        if self.native_flops is not None:
            lines.append(f"FLOPs (torch nativo) : {self.native_flops / 1e9:8.4f} G")
        if self.fvcore_macs is not None:
            lines.append(f"MAC   (fvcore)       : {self.fvcore_macs / 1e9:8.4f} G")
            lines.append(f"  -> x2 in FLOPs     : {2 * self.fvcore_macs / 1e9:8.4f} G")
        if self.manual_extra_flops:
            lines.append(f"extra manuali        : {self.manual_extra_flops / 1e9:8.4f} G")
            if self.native_total is not None:
                lines.append(f"TOTALE (nativo+extra): {self.native_total / 1e9:8.4f} G")
        return "\n".join(lines)


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    ps = model.parameters()
    if trainable_only:
        ps = (p for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in ps)


def grid_sample_flops(num_points: int, channels: int) -> float:
    """FLOPs dell'interpolazione bilineare, contati a mano.

    Per ogni punto e canale: 4 letture pesate + 3 somme = 7 operazioni,
    piu' ~4 operazioni per calcolare i pesi (condivisi fra i canali).
    """
    return num_points * (4.0 + channels * 7.0)


def analyze(
    model: nn.Module,
    input_shape: tuple[int, ...],
    *,
    device: str | torch.device = "cpu",
    manual_extra_flops: float = 0.0,
) -> FlopReport:
    """Misura il costo di un forward con batch 1.

    `input_shape` e' la forma SENZA la dimensione di batch,
    es. (16000,) per una waveform da 1 s.
    """
    model = model.to(device).eval()
    x = torch.zeros(1, *input_shape, device=device)

    native = _count_native(model, x)
    fvcore = _count_fvcore(model, x)

    return FlopReport(
        native_flops=native,
        fvcore_macs=fvcore,
        manual_extra_flops=manual_extra_flops,
        params=count_parameters(model),
    )


def _count_native(model: nn.Module, x: torch.Tensor) -> float | None:
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        return None
    counter = FlopCounterMode(display=False)
    with torch.no_grad(), counter:
        model(x)
    return float(counter.get_total_flops())


def _count_fvcore(model: nn.Module, x: torch.Tensor) -> float | None:
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        return None
    analysis = FlopCountAnalysis(model, x)
    analysis.unsupported_ops_warnings(False)
    analysis.uncalled_modules_warnings(False)
    with torch.no_grad():
        return float(analysis.total())


def benchmark_latency(
    model: nn.Module,
    input_shape: tuple[int, ...],
    *,
    device: str | torch.device = "cuda",
    warmup: int = 20,
    iters: int = 100,
) -> float:
    """Latenza media in ms per un forward con batch 1.

    Serve a mostrare che il guadagno in FLOPs non si traduce
    automaticamente in latenza: e' un'osservazione che vale una slide.
    """
    import time

    model = model.to(device).eval()
    x = torch.zeros(1, *input_shape, device=device)
    is_cuda = torch.device(device).type == "cuda"

    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            model(x)
        if is_cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    return 1000.0 * elapsed / iters
