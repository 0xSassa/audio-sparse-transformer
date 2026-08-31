"""Tracciamento degli esperimenti: CSV e TensorBoard.

Due destinazioni dietro una sola interfaccia:

- il CSV e' l'unica fonte di verita' che sopravvive a tutto: sta nella
  cartella del run, si legge con pandas, si committa in git se serve, ed e'
  cio' da cui `scripts/plot_results.py` ricostruisce ogni figura;
- TensorBoard e' locale e istantaneo, utile mentre il run gira.

Il progetto ha usato per un periodo anche Weights & Biases come terza
destinazione. E' stato rimosso: il confronto FRA run che giustificava W&B lo
fa gia' `plot_results.py` leggendo i CSV, e un servizio esterno con account,
rete e stato remoto e' un requisito in piu' per chiunque debba solo
riprodurre i risultati. Nessun numero riportato veniva da li'.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


class RunTracker:
    """Raccoglie le metriche e le manda dove serve."""

    def __init__(
        self,
        run_dir: Path,
        cfg: dict[str, Any],
        tag: str,
        seed: int,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.tag = tag
        self.seed = seed
        self._csv = self.run_dir / "metrics.csv"
        self._fields: list[str] | None = None
        self._tb = None

        if cfg.get("log", {}).get("tensorboard", True):
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._tb = SummaryWriter(self.run_dir / "tb")
            except ImportError:
                print("[tracking] tensorboard non disponibile")

    # ------------------------------------------------------------------
    def log(self, step: int, metrics: dict[str, float]) -> None:
        row = {"epoch": step, **{k: v for k, v in metrics.items()}}

        if self._fields is None:
            self._fields = list(row)
            if not self._csv.exists():
                with open(self._csv, "w", newline="", encoding="utf-8") as fh:
                    csv.DictWriter(fh, self._fields).writeheader()
        with open(self._csv, "a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, self._fields, extrasaction="ignore").writerow(row)

        if self._tb is not None:
            for key, value in metrics.items():
                self._tb.add_scalar(key.replace("_", "/", 1), value, step)

    def summary(self, **values: Any) -> None:
        with open(self.run_dir / "summary.json", "w", encoding="utf-8") as fh:
            json.dump(values, fh, indent=2)

    def finish(self) -> None:
        if self._tb is not None:
            self._tb.close()
