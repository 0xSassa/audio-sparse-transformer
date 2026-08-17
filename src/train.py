"""Training loop.

    python -m src.train --config configs/dense.yaml --seed 0
    python -m src.train --config configs/dense.yaml --epochs 5 --tag smoke

Ogni run e' completamente determinato dal file di configurazione piu' il
seed: la config viene archiviata accanto ai risultati, cosi' un run e'
riproducibile senza dover ricordare quali flag erano stati passati.

TRE SCELTE CHE VALE LA PENA CONOSCERE PRIMA DI LEGGERE IL CODICE

1. Si valuta il modello EMA, non i pesi correnti. E' quello che entrambi i
   paper riportano; dimenticarlo costa qualche decimo di punto.

2. L'accuratezza di TRAINING non e' interpretabile. Con mixing e phasemix
   attive a ogni batch il modello vede target multi-hot: il numero che si
   guarda e' solo quello di validation.

3. Il test set si tocca UNA VOLTA SOLA, alla fine, e c'e' una guardia nel
   codice che lo impedisce due volte (`scripts/evaluate.py`). Selezionare
   il checkpoint guardando il test e' il modo piu' comune di produrre
   numeri non confrontabili con la letteratura.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from src.data.augment import apply_augmentations, one_hot
from src.data.dataset import make_loader
from src.data.features import LogMelSpectrogram
from src.models.dense import DenseAudioTransformer
from src.utils import ModelEMA, load_config, param_groups, seed_everything

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# Costruzione dei pezzi
# --------------------------------------------------------------------------

def build_frontend(cfg: dict[str, Any]) -> LogMelSpectrogram:
    f = cfg["features"]
    return LogMelSpectrogram(
        sample_rate=cfg["data"]["sample_rate"],
        n_fft=f["n_fft"], hop_length=f["hop_length"], n_mels=f["n_mels"],
        f_min=f["f_min"], f_max=f["f_max"], top_db=f["top_db"],
        normalize=f["normalize"],
    )


def build_model(cfg: dict[str, Any], n_mels: int, n_frames: int) -> nn.Module:
    m = dict(cfg["model"])
    m.pop("kind", None)
    pool = m.pop("pool_stride", (2, 2))
    return DenseAudioTransformer(
        num_classes=cfg["data"]["num_classes"],
        n_mels=n_mels, n_frames=n_frames,
        pool_stride=tuple(pool), **m,
    )


def build_optimizer(model: nn.Module, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    o = cfg["optim"]
    groups = param_groups(model, o["weight_decay"], skip_1d=o["wd_skip_1d_params"])
    return torch.optim.AdamW(
        groups, lr=o["lr"], betas=tuple(o["betas"]), eps=o["eps"], weight_decay=0.0
    )


# --------------------------------------------------------------------------
# Un'epoca
# --------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module, frontend: nn.Module, ema: ModelEMA, loader,
    optimizer: torch.optim.Optimizer, scheduler, cfg: dict[str, Any],
    device: torch.device, epoch: int,
) -> dict[str, float]:
    model.train()
    a = cfg["augment"]
    num_classes = cfg["data"]["num_classes"]
    total_loss, seen = 0.0, 0
    start = time.perf_counter()

    for wave, labels in loader:
        wave = wave.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        wave, targets = apply_augmentations(
            wave, labels, num_classes,
            names=tuple(a["names"]), mix_ratio=a["mix_ratio"],
            epoch=epoch, epoch_mix=a["epoch_mix"],
        )
        logits = model(frontend(wave))
        loss = F.binary_cross_entropy_with_logits(logits, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        ema.update(model)

        total_loss += loss.detach().item() * wave.shape[0]
        seen += wave.shape[0]

    if device.type == "cuda":
        torch.cuda.synchronize()
    return {
        "loss": total_loss / seen,
        "lr": scheduler.get_last_lr()[0],
        "seconds": time.perf_counter() - start,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module, frontend: nn.Module, loader, device: torch.device,
    num_classes: int,
) -> dict[str, float]:
    """Accuratezza top-1 e loss su target one-hot (nessuna augmentation)."""
    model.eval()
    correct, seen, total_loss = 0, 0, 0.0
    for wave, labels in loader:
        wave = wave.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(frontend(wave))
        total_loss += F.binary_cross_entropy_with_logits(
            logits, one_hot(labels, num_classes)
        ).item() * wave.shape[0]
        correct += int((logits.argmax(dim=1) == labels).sum())
        seen += wave.shape[0]
    model.train()
    return {"accuracy": correct / seen, "loss": total_loss / seen, "n": seen}


# --------------------------------------------------------------------------
# Checkpoint
# --------------------------------------------------------------------------

def save_checkpoint(path: Path, **state: Any) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)          # atomico: un'interruzione non lascia file rotti


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/dense.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None, help="sovrascrive la config")
    ap.add_argument("--tag", type=str, default=None, help="nome della cartella del run")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--deterministic", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["optim"]["epochs"] = args.epochs
    seed_everything(args.seed, deterministic=args.deterministic)

    tag = args.tag or f"{args.config.stem}_seed{args.seed}"
    run_dir = ROOT / cfg["log"]["dir"] / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, run_dir / "config.yaml")
    with open(run_dir / "resolved_config.json", "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = ROOT / cfg["data"]["cache_dir"]
    workers = cfg["data"]["num_workers"]
    batch = cfg["optim"]["batch_size"]

    train_loader = make_loader(cache, "train", batch_size=batch, num_workers=workers)
    val_loader = make_loader(cache, "validation", batch_size=256, num_workers=workers)

    frontend = build_frontend(cfg).to(device)
    n_frames = frontend.n_frames(cfg["data"]["clip_samples"])
    model = build_model(cfg, frontend.n_mels, n_frames).to(device)
    ema = ModelEMA(model, decay=cfg["optim"]["ema_decay"])
    ema.module.to(device)

    optimizer = build_optimizer(model, cfg)
    epochs = cfg["optim"]["epochs"]
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg["optim"]["lr"], epochs=epochs,
        steps_per_epoch=len(train_loader), pct_start=cfg["optim"]["pct_start"],
        div_factor=cfg["optim"]["div_factor"],
        final_div_factor=cfg["optim"]["final_div_factor"],
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"run          : {tag}")
    print(f"device       : {device} · {torch.cuda.get_device_name(0) if device.type=='cuda' else ''}")
    print(f"modello      : {n_params:,} parametri · sequenza {model.seq_len} · "
          f"feature {model.feature_shape}")
    print(f"dati         : train {len(train_loader.dataset):,} · "
          f"val {len(val_loader.dataset):,} · batch {batch}")
    print(f"training     : {epochs} epoche · lr max {cfg['optim']['lr']} · "
          f"EMA {cfg['optim']['ema_decay']} · augment {cfg['augment']['names']}")
    print()

    start_epoch, best_acc = 1, 0.0
    ckpt_last, ckpt_best = run_dir / "last.pt", run_dir / "best.pt"
    if args.resume and ckpt_last.exists():
        state = torch.load(ckpt_last, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        ema.module.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, best_acc = state["epoch"] + 1, state["best_acc"]
        print(f"ripreso da epoca {state['epoch']} (best {100*best_acc:.2f}%)\n")

    csv_path = run_dir / "metrics.csv"
    if not csv_path.exists():
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(
                ["epoch", "train_loss", "lr", "val_acc_ema", "val_acc_raw",
                 "val_loss_ema", "seconds"]
            )

    writer = None
    if cfg["log"].get("tensorboard"):
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(run_dir / "tb")
        except ImportError:
            pass

    for epoch in range(start_epoch, epochs + 1):
        tr = train_one_epoch(model, frontend, ema, train_loader, optimizer,
                             scheduler, cfg, device, epoch)
        ev_ema = evaluate(ema.module, frontend, val_loader, device,
                          cfg["data"]["num_classes"])
        ev_raw = evaluate(model, frontend, val_loader, device,
                          cfg["data"]["num_classes"])

        is_best = ev_ema["accuracy"] > best_acc
        best_acc = max(best_acc, ev_ema["accuracy"])
        mark = "  <-- best" if is_best else ""
        print(f"epoca {epoch:>3}/{epochs}  loss {tr['loss']:.4f}  "
              f"lr {tr['lr']:.2e}  val(EMA) {100*ev_ema['accuracy']:.2f}%  "
              f"val(raw) {100*ev_raw['accuracy']:.2f}%  {tr['seconds']:.0f}s{mark}")

        with open(csv_path, "a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow([
                epoch, f"{tr['loss']:.6f}", f"{tr['lr']:.3e}",
                f"{ev_ema['accuracy']:.6f}", f"{ev_raw['accuracy']:.6f}",
                f"{ev_ema['loss']:.6f}", f"{tr['seconds']:.1f}",
            ])
        if writer is not None:
            writer.add_scalar("train/loss", tr["loss"], epoch)
            writer.add_scalar("train/lr", tr["lr"], epoch)
            writer.add_scalar("val/acc_ema", ev_ema["accuracy"], epoch)
            writer.add_scalar("val/acc_raw", ev_raw["accuracy"], epoch)

        state = {
            "epoch": epoch, "best_acc": best_acc, "seed": args.seed, "config": cfg,
            "model": model.state_dict(), "ema": ema.module.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        }
        save_checkpoint(ckpt_last, **state)
        if is_best:
            save_checkpoint(ckpt_best, **state)

    if writer is not None:
        writer.close()

    peak = torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else 0
    summary = {
        "tag": tag, "seed": args.seed, "epochs": epochs,
        "best_val_accuracy": best_acc, "params": n_params,
        "peak_gpu_mb": round(peak),
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nmigliore accuratezza di validation: {100*best_acc:.2f}%")
    print(f"picco memoria GPU: {peak:.0f} MB")
    print(f"risultati in {run_dir}")
    print("\nIl test set NON e' stato toccato. Per la valutazione finale:")
    print(f"  python scripts/evaluate.py --run {run_dir.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
