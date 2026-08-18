"""Tracciamento degli esperimenti: CSV, TensorBoard e Weights & Biases.

Tre destinazioni dietro una sola interfaccia, con degrado controllato: se
W&B non e' installato, non c'e' login o non c'e' rete, il run PROSEGUE e
scrive comunque CSV e TensorBoard. Un esperimento da due ore non deve
morire perche' un servizio esterno non risponde.

PERCHE' TUTTE E TRE

- il CSV e' l'unica fonte di verita' che sopravvive a tutto: sta nella
  cartella del run, si legge con pandas, si commetta in git se serve;
- TensorBoard e' locale e istantaneo, utile mentre il run gira;
- W&B da' il confronto FRA run, che e' esattamente cio' che serve quando
  si hanno tre seed per configurazione e piu' configurazioni da mettere
  a confronto — e produce i grafici della presentazione.

MODALITA' OFFLINE. Con `mode: offline` W&B scrive in locale senza rete e i
run si sincronizzano dopo con `wandb sync`. Su un portatile che addestra
la sera senza connessione stabile e' la scelta piu' sicura.
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
        *,
        resume: bool = False,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.tag = tag
        self._csv = self.run_dir / "metrics.csv"
        self._fields: list[str] | None = None
        self._tb = None
        self._wandb = None

        log_cfg = cfg.get("log", {})

        if log_cfg.get("tensorboard", True):
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._tb = SummaryWriter(self.run_dir / "tb")
            except ImportError:
                print("[tracking] tensorboard non disponibile")

        wb = log_cfg.get("wandb", {})
        if wb.get("enabled", False):
            self._init_wandb(wb, cfg, tag, seed, resume)

    # ------------------------------------------------------------------
    def _init_wandb(self, wb: dict[str, Any], cfg: dict[str, Any], tag: str,
                    seed: int, resume: bool) -> None:
        try:
            import wandb
        except ImportError:
            print("[tracking] wandb non installato: si prosegue senza")
            return

        # L'id si conserva nella cartella del run, cosi' --resume riprende
        # la stessa curva invece di aprirne una nuova accanto.
        id_file = self.run_dir / "wandb_id.txt"
        run_id = id_file.read_text().strip() if (resume and id_file.exists()) else None

        try:
            self._wandb = wandb.init(
                project=wb.get("project", "audio-sparse-transformer"),
                entity=wb.get("entity") or None,
                mode=wb.get("mode", "online"),
                name=tag,
                id=run_id,
                resume="allow" if run_id else None,
                config=cfg,
                tags=[f"seed{seed}", cfg.get("model", {}).get("kind", "model")],
                dir=str(self.run_dir),
            )
            id_file.write_text(self._wandb.id, encoding="utf-8")
            print(f"[tracking] wandb attivo ({wb.get('mode', 'online')}): {self._wandb.url}")
        except Exception as exc:  # noqa: BLE001 — qualunque errore non deve fermare il run
            print(f"[tracking] wandb non inizializzato ({type(exc).__name__}: {exc})")
            print("[tracking] il run prosegue con CSV e TensorBoard")
            self._wandb = None

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

        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def summary(self, **values: Any) -> None:
        with open(self.run_dir / "summary.json", "w", encoding="utf-8") as fh:
            json.dump(values, fh, indent=2)
        if self._wandb is not None:
            for key, value in values.items():
                self._wandb.summary[key] = value

    def save_artifact(self, path: Path, kind: str = "model") -> None:
        """Carica un file su W&B, se attivo. In locale il file c'e' comunque."""
        if self._wandb is None:
            return
        try:
            self._wandb.log_artifact(str(path), type=kind)
        except Exception as exc:  # noqa: BLE001
            print(f"[tracking] artifact non caricato ({type(exc).__name__})")

    def finish(self) -> None:
        if self._tb is not None:
            self._tb.close()
        if self._wandb is not None:
            self._wandb.finish()
