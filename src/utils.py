"""Utilita' trasversali: seed, config, EMA."""

from __future__ import annotations

import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Operazioni senza implementazione deterministica su CUDA che il progetto usa:
# il backward di `grid_sample` accumula con atomicAdd, quindi l'ordine delle
# somme in virgola mobile varia fra esecuzioni.
NONDETERMINISTIC_OPS = {"sparse": ("grid_sampler_2d_backward_cuda",)}


def seed_everything(seed: int, *, deterministic: bool = False,
                    warn_only: bool = False) -> str:
    """Fissa i seed. Ritorna il livello di determinismo ottenuto.

    Il modello sparso non puo' girare in determinismo stretto su GPU, perche'
    `grid_sampler_2d_backward_cuda` non ha implementazione deterministica e
    PyTorch solleva un errore. Con `warn_only=True` resta deterministico tutto
    il resto, ma il livello va registrato accanto ai risultati.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if not deterministic:
        torch.backends.cudnn.benchmark = True
        return "off"

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    return "warn_only" if warn_only else "strict"


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

    `defaults: base.yaml` fa ereditare i blocchi non ridefiniti, cosi' nel
    file specifico si vede solo cio' che cambia.
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


def apply_overrides(cfg: dict[str, Any], assignments: list[str]) -> list[str]:
    """Applica assegnamenti `chiave.annidata=valore` alla config gia' caricata.

        apply_overrides(cfg, ["model.num_tokens=9", "optim.epochs=50"])

    Evita gli 8 YAML quasi identici delle due ablation. Due guardie: la chiave
    deve gia' esistere, altrimenti un refuso come `num_token` farebbe girare il
    run sul default fingendo di essere un'ablation, e il tipo deve combaciare.
    Ritorna gli assegnamenti applicati, per stamparli.
    """
    import yaml

    applied: list[str] = []
    for item in assignments:
        if "=" not in item:
            raise ValueError(f"--set vuole chiave=valore, ricevuto {item!r}")
        path, raw = item.split("=", 1)
        keys = path.strip().split(".")

        node: Any = cfg
        for i, key in enumerate(keys[:-1]):
            if not isinstance(node, dict) or key not in node:
                prefix = ".".join(keys[: i + 1])
                raise KeyError(
                    f"--set {path}: il blocco {prefix!r} non esiste nella config"
                )
            node = node[key]

        leaf = keys[-1]
        if not isinstance(node, dict) or leaf not in node:
            vicini = sorted(node) if isinstance(node, dict) else []
            raise KeyError(
                f"--set {path}: la chiave {leaf!r} non esiste. "
                f"Chiavi disponibili in {'.'.join(keys[:-1]) or 'radice'}: {vicini}"
            )

        old = node[leaf]
        new = yaml.safe_load(raw)
        # int accettato dove c'e' un float: 4 al posto di 4.0 e' innocuo
        int_per_float = isinstance(old, float) and isinstance(new, int)
        if old is not None and not isinstance(new, type(old)) and not int_per_float:
            raise TypeError(
                f"--set {path}: atteso {type(old).__name__}, "
                f"ricevuto {type(new).__name__} ({raw!r})"
            )
        node[leaf] = new
        applied.append(f"{path}: {old!r} -> {new!r}")
    return applied


def param_groups(model: torch.nn.Module, weight_decay: float,
                 skip_1d: bool = True) -> list[dict[str, Any]]:
    """Gruppi di parametri per il weight decay.

    Criterio del `add_weight_decay` di EAT: i parametri 1-D restano esclusi,
    cioe' bias e parametri di normalizzazione.
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


def seed_epoch(base_seed: int, epoch: int) -> None:
    """Riporta i generatori a uno stato funzione pura di (seed, epoca).

    Rende il flusso casuale di un'epoca indipendente da quante estrazioni sono
    state fatte prima: un run ripreso ne consuma una in piu' per il `base_seed`
    dei worker, sfasando la scelta delle augmentation.
    """
    torch.manual_seed(base_seed * 1_000_003 + epoch)   # copre anche CUDA
    np.random.seed((base_seed * 1_000_003 + epoch) % (2**32))
    random.seed(base_seed * 1_000_003 + epoch)


def provenance() -> dict[str, Any]:
    """Da quale stato del codice e su quale macchina viene un risultato.

    `dirty` segnala modifiche non committate: quel risultato non e'
    riproducibile dal solo commit.
    """
    import platform
    import subprocess

    info: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda

    root = Path(__file__).resolve().parents[1]
    try:
        info["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        # `dirty` conta solo i file tracciati, gli unici che cambiano il codice
        # eseguito: contare anche i non tracciati marcava sporco ogni run per un
        # file di appunti nella radice. I non tracciati si contano a parte.
        info["dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root, text=True, stderr=subprocess.DEVNULL,
        ).strip())
        info["untracked"] = len(subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=root, text=True, stderr=subprocess.DEVNULL,
        ).split())
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        info["commit"] = None
        info["dirty"] = None
    return info


def rng_state() -> dict[str, Any]:
    """Stato di tutti i generatori, per un resume identico.

    Senza, riprendere da un checkpoint riparte con generatori reinizializzati:
    il run resta valido ma non e' piu' confrontabile bit a bit.
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

    Con soglia infinita `clip_grad_norm_` misura e non taglia: una norma che
    esplode o collassa si vede prima che sulla loss.
    """
    return float(
        torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
    )


class ModelEMA:
    """Media mobile esponenziale dei pesi (Tarvainen & Valpola 2017).

    Decay 0,995 come in entrambi i paper. E' il modello EMA quello che si
    valuta.
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
