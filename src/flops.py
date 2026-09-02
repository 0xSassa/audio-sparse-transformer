"""Contatore di FLOPs: la metrica centrale del progetto.

Tre avvertenze, ognuna capace da sola di invalidare il confronto:

1. QUI SI CONTANO FLOPs, NON MAC. `torch.utils.flop_counter` conta la
   moltiplicazione-addizione come 2 operazioni. Il paper non dichiara la
   convenzione: e' dedotta dai suoi numeri, che tornano in FLOPs sulla
   configurazione di default. La conversione MAC -> FLOPs vale 2 per
   definizione, ma due contatori reali non concordano esattamente perche'
   coprono insiemi di operazioni diversi: `results/benchmark.json`, prodotto
   quando il repository aveva anche un contatore fvcore, riporta un rapporto
   di 1.99 sul denso e 1.95 sullo sparso. La differenza fra contatori e'
   quindi di pochi punti percentuali, contro il +78 % di divergenza dalla
   Tabella 3 del paper (README, sez. 5).
2. `grid_sample` non e' contata da nessuno strumento — e' gather e
   interpolazione, senza matmul — e va aggiunta a mano con
   `grid_sample_flops()`. Lo fa `scripts/cost_ablation.py`, che ne riporta
   la quota: 0.60 % del modello sparso.
3. Si misura su UN input da 1 secondo, in inferenza, con batch 1. Il
   training costa circa 3x, ma non e' quello che si riporta ne' qui ne' nel
   paper.

Non si contano softmax, GELU e LayerNorm: e' la convenzione con cui sono
riportati i FLOPs in letteratura, quella del paper compresa, e cambiarla
renderebbe i due numeri non confrontabili.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class FlopReport:
    """Costo di un forward. `flops` e' None se il contatore non e' disponibile."""

    flops: float | None
    params: int

    def __str__(self) -> str:
        lines = [f"parametri : {self.params / 1e6:8.3f} M"]
        if self.flops is not None:
            lines.append(f"FLOPs     : {self.flops / 1e9:8.4f} G")
        return "\n".join(lines)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def grid_sample_flops(num_points: int, channels: int) -> float:
    """FLOPs dell'interpolazione bilineare, contati a mano.

    Per punto e canale: 4 letture pesate + 3 somme = 7 operazioni, piu' ~4
    per i pesi, condivisi fra i canali.
    """
    return num_points * (4.0 + channels * 7.0)


def analyze(
    model: nn.Module,
    input_shape: tuple[int, ...],
    *,
    device: str | torch.device = "cpu",
) -> FlopReport:
    """Misura il costo di un forward con batch 1.

    `input_shape` e' la forma SENZA la dimensione di batch,
    es. (1, 64, 101) per uno spettrogramma log-mel.
    """
    model = model.to(device).eval()
    x = torch.zeros(1, *input_shape, device=device)
    return FlopReport(flops=_count_native(model, x), params=count_parameters(model))


def _count_native(model: nn.Module, x: torch.Tensor) -> float | None:
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        return None
    # Niente torch.no_grad(), ed e' voluto: sotto no_grad il ModuleTracker di
    # FlopCounterMode fallisce sui Parameter espansi (le regioni iniziali del
    # modello sparso), che non hanno grad_fn. Il grafo attivo non cambia il
    # conteggio, che riguarda solo le op del forward.
    counter = FlopCounterMode(display=False)
    with counter:
        model(x)
    return float(counter.get_total_flops())


def benchmark_latency(
    model: nn.Module,
    input_shape: tuple[int, ...],
    *,
    device: str | torch.device = "cuda",
    warmup: int = 20,
    iters: int = 100,
    batch_size: int = 1,
) -> float:
    """Latenza media in ms per un forward, sul solo modello.

    Mostra che il guadagno in FLOPs non si traduce automaticamente in
    latenza. `batch_size` cambia la domanda: a 1 e' lo scenario del paper —
    una parola alla volta — dove i lanci di kernel non sono ammortizzati e i
    FLOPs predicono male; a 64 e' il regime di training, dove il rapporto si
    avvicina a quello aritmetico.

    Il `warmup` non e' un dettaglio: le prime esecuzioni pagano allocazione
    della cache, scelta degli algoritmi cuDNN e compilazione dei kernel.
    """
    import time

    model = model.to(device).eval()
    x = torch.zeros(batch_size, *input_shape, device=device)
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
