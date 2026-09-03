"""Training loop.

    python -m src.train --config configs/dense.yaml --seed 0
    python -m src.train --config configs/dense.yaml --epochs 5 --tag smoke

Ogni run e' determinato dalla config piu' il seed, e la config viene
archiviata accanto ai risultati.

Tre scelte da conoscere prima di leggere il codice:

1. Si valuta il modello EMA, non i pesi correnti, come in entrambi i paper.
2. L'accuratezza di TRAINING non e' interpretabile: con le augmentation
   attive a ogni batch il modello vede target multi-hot. Si guarda solo
   quella di validation.
3. Il test set si tocca UNA VOLTA SOLA, alla fine, con una guardia in
   `scripts/evaluate.py` che lo impedisce due volte.
"""

from __future__ import annotations

import argparse
import inspect
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
from src.data.speech_commands import LABELS
from src.metrics import classification_summary, print_summary, save_summary
from src.models.dense import DenseAudioTransformer
from src.models.sparse_model import SparseAudioTransformer
from src.tracking import RunTracker
from src.utils import (
    ModelEMA,
    apply_overrides,
    grad_global_norm,
    load_config,
    load_rng_state,
    param_groups,
    provenance,
    rng_state,
    seed_epoch,
    seed_everything,
)

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
    )


MODELS = {"dense": DenseAudioTransformer, "sparse": SparseAudioTransformer}

# Chiavi che i checkpoint e i `resolved_config.json` ARCHIVIATI portano
# ancora ma che il codice non conosce piu': `build_model` le scarta per poter
# ricaricare i pesi vecchi. Il valore archiviato coincide con l'unico
# comportamento rimasto.
OBSOLETE_MODEL_KEYS = ("stem", "norm", "pos_encoding", "num_conv_layers")


def model_config_with_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    """La config di un modello archiviato, completata coi default correnti.

    Un `resolved_config.json` scritto prima che un'opzione esistesse non la
    contiene, e confrontarlo con uno scritto dopo direbbe che sono due modelli
    diversi quando sono lo stesso. I default si leggono dalla firma della
    classe, cosi' non esiste un secondo elenco da tenere allineato; le opzioni
    RIMOSSE dalla firma non si leggono piu' e vanno scartate a mano, ed e' a
    cosa serve `OBSOLETE_MODEL_KEYS`.
    """
    model = MODELS.get(cfg.get("kind"))
    if model is None:
        return dict(cfg)
    out = {k: v for k, v in cfg.items() if k not in OBSOLETE_MODEL_KEYS}
    for name, param in inspect.signature(model.__init__).parameters.items():
        if param.default is not inspect.Parameter.empty and name not in out:
            out[name] = param.default
    return out


def build_model(cfg: dict[str, Any], n_mels: int, n_frames: int) -> nn.Module:
    """Costruisce il modello dalla config, scegliendo su `model.kind`.

    I due modelli espongono la stessa interfaccia — spettrogramma in, logits
    fuori, piu' `seq_len` e `feature_shape` — quindi training loop,
    valutazione e contatore di FLOPs non li distinguono.
    """
    m = dict(cfg["model"])
    kind = m.pop("kind", "dense")
    if kind not in MODELS:
        raise ValueError(f"model.kind sconosciuto: {kind!r} (attesi {sorted(MODELS)})")

    # tiene ricaricabili i checkpoint archiviati, da cui `evaluate.py` e
    # `region_geometry.py` ricostruiscono il modello
    for obsoleta in OBSOLETE_MODEL_KEYS:
        m.pop(obsoleta, None)

    # `pool_stride` invece non ha default: e' DEDOTTO dai vincoli del paper, e
    # ricadere in silenzio su (2,2) darebbe 26 frame invece di 51, cioe' un
    # altro esperimento.
    if "pool_stride" not in m:
        raise ValueError(
            "la configurazione non dichiara model.pool_stride: e' un parametro "
            "dedotto, non un dettaglio con un default sensato (atteso [2, 1])"
        )
    pool = m.pop("pool_stride")
    return MODELS[kind](
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

# Tutto in fp32, training compreso: l'autocast in bf16 non e' mai stata usata
# in nessuno dei run riportati ed e' stata rimossa.


def train_one_epoch(
    model: nn.Module, frontend: nn.Module, ema: ModelEMA, loader,
    optimizer: torch.optim.Optimizer, scheduler, cfg: dict[str, Any],
    device: torch.device, epoch: int, base_seed: int = 0,
) -> dict[str, float]:
    model.train()
    a = cfg["augment"]
    num_classes = cfg["data"]["num_classes"]
    total_loss, total_gnorm, seen, steps = 0.0, 0.0, 0, 0
    start = time.perf_counter()

    # L'iteratore si crea QUI, non al `for`, e si semina DOPO: costruirlo
    # consuma estrazioni dal generatore globale (`base_seed` dei worker) solo
    # nel processo che lo costruisce, quindi un run ripreso e uno continuo
    # divergerebbero. Cosi' lo stato a inizio ciclo e' identico nei due casi.
    data_iter = iter(loader)
    seed_epoch(base_seed, epoch)

    try:
        import sys

        from tqdm import tqdm

        # disattivata fuori dal terminale: in un file di log una barra di
        # avanzamento produce migliaia di righe inutili
        iterator = tqdm(data_iter, desc=f"epoca {epoch}", leave=False,
                        unit="batch", total=len(loader),
                        disable=not sys.stderr.isatty())
    except ImportError:
        iterator = data_iter

    for wave, labels in iterator:
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
        total_gnorm += grad_global_norm(model)
        optimizer.step()
        scheduler.step()
        ema.update(model)

        total_loss += loss.detach().item() * wave.shape[0]
        seen += wave.shape[0]
        steps += 1

    if device.type == "cuda":
        torch.cuda.synchronize()
    return {
        "train_loss": total_loss / seen,
        "train_grad_norm": total_gnorm / steps,
        "lr": scheduler.get_last_lr()[0],
        "seconds": time.perf_counter() - start,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module, frontend: nn.Module, loader, device: torch.device,
    num_classes: int, *, collect: bool = False,
) -> dict[str, Any]:
    """Accuratezza top-1 e loss su target one-hot (nessuna augmentation).

    Con `collect=True` restituisce anche etichette e predizioni, per le
    metriche per classe.
    """
    was_training = model.training
    model.eval()
    correct, seen, total_loss = 0, 0, 0.0
    all_true: list[torch.Tensor] = []
    all_pred: list[torch.Tensor] = []

    for wave, labels in loader:
        wave = wave.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(frontend(wave))
        total_loss += F.binary_cross_entropy_with_logits(
            logits, one_hot(labels, num_classes)
        ).item() * wave.shape[0]
        pred = logits.argmax(dim=1)
        correct += int((pred == labels).sum())
        seen += wave.shape[0]
        if collect:
            all_true.append(labels.cpu())
            all_pred.append(pred.cpu())

    # si ripristina il modo precedente invece di forzare train(): la funzione
    # gira anche sul modello EMA, che deve restare in eval
    model.train(was_training)
    out: dict[str, Any] = {"accuracy": correct / seen, "loss": total_loss / seen,
                           "n": seen}
    if collect:
        out["y_true"] = torch.cat(all_true).numpy()
        out["y_pred"] = torch.cat(all_pred).numpy()
    return out


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
    ap.add_argument("--deterministic", dest="deterministic", action="store_true",
                    default=None, help="forza il determinismo (default dal config)")
    ap.add_argument("--no-deterministic", dest="deterministic",
                    action="store_false", help="disattiva il determinismo")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="CHIAVE=VALORE",
                    help="sovrascrive una voce della config, es. "
                         "--set model.num_tokens=9. Ripetibile. Richiede "
                         "--tag, per non sovrascrivere il run di base")
    ap.add_argument("--stop-after", type=int, default=None, metavar="N",
                    help="ferma dopo N epoche in QUESTA invocazione, lasciando "
                         "lo schedule configurato per il totale. Serve a "
                         "spezzare un run lungo su piu' sessioni: si riprende "
                         "con --resume e la curva del learning rate resta "
                         "quella giusta")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["optim"]["epochs"] = args.epochs

    # Override applicati PRIMA di archiviare resolved_config.json, cosi' il
    # file riporta i valori realmente usati. Senza --tag non si parte:
    # un'ablation che scrive nella cartella del run di base lo distrugge in
    # silenzio.
    if args.overrides:
        if args.tag is None:
            print("[errore] --set richiede --tag: senza, questo run scriverebbe "
                  "nella cartella del run di base e lo sovrascriverebbe.")
            print(f"         Esempio: --tag {args.config.stem}_"
                  f"{args.overrides[0].split('=')[0].split('.')[-1]}"
                  f"{args.overrides[0].split('=')[1]}_seed{args.seed}")
            return 2
        try:
            for line in apply_overrides(cfg, args.overrides):
                print(f"[override] {line}")
        except (KeyError, TypeError, ValueError) as exc:
            # niente traceback: l'errore e' nella riga di comando, e il
            # messaggio elenca gia' le chiavi disponibili
            print(f"[errore] {exc}")
            return 2

    deterministic = (cfg.get("deterministic", False) if args.deterministic is None
                     else args.deterministic)
    cfg["deterministic"] = deterministic

    # Il backward di `grid_sample` non e' deterministico su CUDA: in modalita'
    # stretta PyTorch solleverebbe un errore. Si rilassa a `warn_only` SOLO
    # per il modello sparso, e il livello ottenuto finisce nel summary.
    needs_relaxed = cfg["model"].get("kind") == "sparse"
    determinism = seed_everything(args.seed, deterministic=deterministic,
                                  warn_only=needs_relaxed)
    cfg["determinism_level"] = determinism

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

    train_loader = make_loader(cache, "train", batch_size=batch,
                               num_workers=workers, seed=args.seed)
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

    # FLOPs misurati sul modello di QUESTO run e registrati col resto: sono la
    # metrica centrale, non si ricalcolano dopo sperando che la configurazione
    # fosse la stessa.
    from src.flops import analyze

    cost = analyze(build_model(cfg, frontend.n_mels, n_frames),
                   (1, frontend.n_mels, n_frames))
    prov = provenance()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"run          : {tag}")
    print(f"codice       : commit {(prov['commit'] or '?')[:8]}"
          f"{'  ALBERO SPORCO' if prov['dirty'] else ''}")
    print(f"costo        : {cost.flops/1e9:.4f} GFLOP")
    print(f"device       : {device} · {torch.cuda.get_device_name(0) if device.type=='cuda' else ''}")
    print(f"modello      : {n_params:,} parametri · sequenza {model.seq_len} · "
          f"feature {model.feature_shape}")
    print(f"dati         : train {len(train_loader.dataset):,} · "
          f"val {len(val_loader.dataset):,} · batch {batch}")
    print(f"training     : {epochs} epoche · lr max {cfg['optim']['lr']} · "
          f"EMA {cfg['optim']['ema_decay']} · augment {cfg['augment']['names']}")
    note = {"strict": "bit-riproducibile", "warn_only": "grid_sample non deterministico",
            "off": "non riproducibile"}[determinism]
    print(f"riproducibile: determinismo={determinism} ({note})")
    print()

    start_epoch, best_acc = 1, 0.0
    ckpt_last, ckpt_best = run_dir / "last.pt", run_dir / "best.pt"
    if args.resume and ckpt_last.exists():
        state = torch.load(ckpt_last, map_location=device, weights_only=False)

        # Il numero di epoche NON e' modificabile in ripresa: one-cycle lega la
        # forma dello schedule al totale dei passi, che `load_state_dict`
        # ripristina dal checkpoint. Riprendere con un valore diverso da' un
        # errore criptico o, se maggiore, uno schedule silenziosamente
        # sbagliato: meglio fermarsi subito.
        old = state.get("config", {}).get("optim", {})
        mismatches = [
            f"{k}: checkpoint {old.get(k)!r} vs richiesto {cfg['optim'][k]!r}"
            for k in ("epochs", "batch_size", "lr", "pct_start")
            if k in old and old[k] != cfg["optim"][k]
        ]
        if mismatches:
            print("[errore] il checkpoint e' stato prodotto con una "
                  "configurazione diversa:")
            for m in mismatches:
                print(f"         {m}")
            print("         Per un run diverso usa un --tag diverso; per "
                  "riprendere questo, ripeti gli stessi valori.")
            return 2

        model.load_state_dict(state["model"])
        ema.module.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        load_rng_state(state.get("rng"))
        start_epoch, best_acc = state["epoch"] + 1, state["best_acc"]
        rng_note = "con stato RNG" if state.get("rng") else "SENZA stato RNG (checkpoint vecchio)"
        print(f"ripreso da epoca {state['epoch']} (best {100*best_acc:.2f}%) — {rng_note}\n")

    tracker = RunTracker(run_dir, cfg, tag, args.seed)

    for epoch in range(start_epoch, epochs + 1):
        # ordine dei dati e flusso casuale delle augmentation dipendono solo da
        # (seed, epoca): identici anche riprendendo in un processo nuovo
        train_loader.sampler.set_epoch(epoch)
        tr = train_one_epoch(model, frontend, ema, train_loader, optimizer,
                             scheduler, cfg, device, epoch, base_seed=args.seed)
        ev_ema = evaluate(ema.module, frontend, val_loader, device,
                          cfg["data"]["num_classes"])
        ev_raw = evaluate(model, frontend, val_loader, device,
                          cfg["data"]["num_classes"])

        is_best = ev_ema["accuracy"] > best_acc
        best_acc = max(best_acc, ev_ema["accuracy"])
        mark = "  <-- best" if is_best else ""
        print(f"epoca {epoch:>3}/{epochs}  loss {tr['train_loss']:.4f}  "
              f"|g| {tr['train_grad_norm']:.2f}  lr {tr['lr']:.2e}  "
              f"val(EMA) {100*ev_ema['accuracy']:.2f}%  "
              f"val(raw) {100*ev_raw['accuracy']:.2f}%  {tr['seconds']:.0f}s{mark}")

        # per lo sparso: le regioni si stanno muovendo? E' l'unico modo di
        # accorgersi che il meccanismo centrale del paper e' inerte.
        diag = model.diagnostics() if hasattr(model, "diagnostics") else {}
        if diag:
            inerte = "   <-- MECCANISMO INERTE" if diag["region_adjust_norm"] == 0 else ""
            print(f"           regioni: spostamento {diag['region_init_drift']:.4f}  "
                  f"|W_adjust| {diag['region_adjust_norm']:.4f}{inerte}")

        tracker.log(epoch, {
            **tr, **diag,
            "val_accuracy_ema": ev_ema["accuracy"],
            "val_accuracy_raw": ev_raw["accuracy"],
            "val_loss_ema": ev_ema["loss"],
            "best_accuracy": best_acc,
        })

        state = {
            "epoch": epoch, "best_acc": best_acc, "seed": args.seed, "config": cfg,
            "model": model.state_dict(), "ema": ema.module.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "rng": rng_state(),
        }
        save_checkpoint(ckpt_last, **state)
        if is_best:
            save_checkpoint(ckpt_best, **state)

        if args.stop_after and (epoch - start_epoch + 1) >= args.stop_after:
            tracker.finish()
            print(f"\nfermato dopo {args.stop_after} epoche di questa sessione "
                  f"(siamo a {epoch}/{epochs}).")
            print(f"Per continuare:  python -m src.train --config {args.config} "
                  f"--seed {args.seed} --tag {tag} --resume")
            return 0

    # Analisi per classe sul modello migliore, su VALIDATION.
    best_state = torch.load(ckpt_best, map_location=device, weights_only=False)
    ema.module.load_state_dict(best_state["ema"])
    final = evaluate(ema.module, frontend, val_loader, device,
                     cfg["data"]["num_classes"], collect=True)
    report = classification_summary(final["y_true"], final["y_pred"], list(LABELS))
    save_summary(report, run_dir / "validation_report.json")
    print("\n--- analisi per classe (validation, modello migliore) ---")
    print_summary(report)

    peak = torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else 0
    tracker.summary(
        balanced_val_accuracy=report["balanced_accuracy"],
        tag=tag, seed=args.seed, epochs=epochs, params=n_params,
        best_val_accuracy=best_acc, peak_gpu_mb=round(peak),
        seq_len=model.seq_len,
        gflops=cost.flops / 1e9,
        provenance=prov, determinism=determinism,
    )
    tracker.finish()

    print(f"\nmigliore accuratezza di validation: {100*best_acc:.2f}%")
    print(f"picco memoria GPU: {peak:.0f} MB")
    print(f"risultati in {run_dir}")
    print("\nIl test set NON e' stato toccato. Per la valutazione finale:")
    print(f"  python scripts/evaluate.py --run {run_dir.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
