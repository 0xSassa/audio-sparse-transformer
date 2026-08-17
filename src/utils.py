"""Utilita' trasversali: seed, config, EMA."""

from __future__ import annotations

import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Fissa i seed. `deterministic=True` costa velocita' ma rende i run ripetibili.

    Nota per l'orale: anche con lo stesso seed, kernel cuDNN non deterministici
    producono run diversi. Per questo si riportano media e deviazione standard
    su >= 3 seed, non un singolo numero.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True


def load_config(path: str | Path) -> dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


class ModelEMA:
    """Media mobile esponenziale dei pesi (Tarvainen & Valpola 2017).

    Sia EAT sia il paper sparso usano decay 0.995. Il modello EMA e' quello
    che si valuta: dimenticarlo costa in genere qualche decimo di punto.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.995) -> None:
        self.decay = decay
        self.module = deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for ema_p, p in zip(self.module.state_dict().values(), model.state_dict().values()):
            if ema_p.dtype.is_floating_point:
                ema_p.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
            else:
                ema_p.copy_(p)
