"""Contatore di FLOPs.

Tre avvertenze, ognuna capace da sola di invalidare il confronto:

1. Si contano FLOPs, non MAC: `torch.utils.flop_counter` conta la
   moltiplicazione-addizione come 2 operazioni. Il paper non dichiara la
   propria convenzione, quindi i suoi numeri e i nostri non sono confrontabili
   a meno di un fattore che non conosciamo.
2. `grid_sample` non e' contata da nessuno strumento, perche' e' gather e
   interpolazione senza matmul. Nella configurazione adottata varrebbe lo
   0,60 % del costo del modello sparso.
3. Si misura su un input da 1 secondo, in inferenza, con batch 1. Il training
   costa circa 3x, ma non e' quello che si riporta ne' qui ne' nel paper.

Softmax, GELU e LayerNorm non si contano: e' la convenzione con cui sono
riportati i FLOPs in letteratura, quella del paper compresa.
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


def analyze(
    model: nn.Module,
    input_shape: tuple[int, ...],
    *,
    device: str | torch.device = "cpu",
) -> FlopReport:
    """Misura il costo di un forward con batch 1.

    `input_shape` e' la forma senza la dimensione di batch, es. (1, 64, 101)
    per uno spettrogramma log-mel.
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
    # conteggio.
    counter = FlopCounterMode(display=False)
    with counter:
        model(x)
    return float(counter.get_total_flops())
