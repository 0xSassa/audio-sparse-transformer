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


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Fonde `override` dentro `base`, ricorsivamente sui dizionari."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    """Carica un YAML risolvendo la chiave `defaults:`.

    `defaults: base.yaml` fa ereditare tutti i blocchi non ridefiniti. Serve
    a tenere in un solo posto cio' che e' comune a tutti gli esperimenti e a
    far vedere nel file specifico solo cio' che cambia — che e' anche il
    modo in cui un revisore capisce al volo cosa distingue un run dall'altro.
    """
    import yaml

    path = Path(path)
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    parent_name = cfg.pop("defaults", None)
    if parent_name is None:
        return cfg
    parent = load_config(path.parent / parent_name)
    return deep_merge(parent, cfg)


def param_groups(model: torch.nn.Module, weight_decay: float,
                 skip_1d: bool = True) -> list[dict[str, Any]]:
    """Gruppi di parametri per il weight decay.

    Criterio preso dal `add_weight_decay` di EAT: ogni parametro con
    `len(shape) == 1` e' escluso dal decay. Cattura in un colpo solo i bias
    di convoluzioni e linear e i pesi e bias di LayerNorm e BatchNorm.

    La motivazione: spingere verso zero un parametro di scala di
    normalizzazione non ha senso statistico, perche' non controlla la
    complessita' del modello ma la sua parametrizzazione interna.
    """
    if not skip_1d:
        return [{"params": [p for p in model.parameters() if p.requires_grad],
                 "weight_decay": weight_decay}]

    decay, no_decay = [], []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        (no_decay if param.ndim == 1 else decay).append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def rng_state() -> dict[str, Any]:
    """Stato di TUTTI i generatori, per un resume davvero identico.

    Senza questo, riprendere da un checkpoint riparte con generatori
    reinizializzati: l'ordine dello shuffle, le permutazioni delle
    augmentation e i lambda sono diversi da quelli che il run avrebbe
    avuto proseguendo. Il risultato resta valido, ma un run interrotto e
    ripreso non e' piu' confrontabile bit a bit con uno mai interrotto —
    e su un portatile le interruzioni non sono l'eccezione.
    """
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def load_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu() if hasattr(state["torch"], "cpu")
                        else state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


@torch.no_grad()
def grad_global_norm(model: torch.nn.Module) -> float:
    """Norma L2 globale del gradiente, senza modificarlo.

    `clip_grad_norm_` con soglia infinita calcola la norma e non taglia
    nulla: e' il modo idiomatico di misurarla. Serve come diagnostica —
    una norma che esplode o che collassa a zero dice subito che qualcosa
    non va, molto prima che si veda sulla loss.
    """
    return float(
        torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
    )


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
