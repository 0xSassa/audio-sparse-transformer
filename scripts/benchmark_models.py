"""Costo e latenza dei modelli, misurati fianco a fianco.

    python scripts/benchmark_models.py

PERCHE' ESISTE UNO SCRIPT DEDICATO. Il tempo per epoca non e' una misura di
latenza: mescola caricamento dei dati, log-mel, augmentation (che per
phasemix include STFT e iSTFT), EMA, norma del gradiente e DUE passate di
validation. Confrontare due modelli sul tempo per epoca risponde a
«quanto dura il mio training», non a «quanto costa questo modello» — e le
due domande hanno risposte diverse.

Peggio: una stima ricavata da un run di una o due epoche paga l'avvio dei
worker e la cache fredda ammortizzati su pochissime epoche, e sbaglia di
un fattore. E' successo due volte in questo progetto, in entrambe le
direzioni. Da qui la regola: la latenza si misura con warmup, su GPU
scarica, sul solo modello.

CHE COSA SI MISURA

  batch 1    lo scenario dichiarato dal paper: un dispositivo che
             classifica una parola alla volta. E' il numero che rende
             onesto o disonesto il claim sui FLOPs.
  batch 64   il regime di training, dove la GPU ha lavoro sufficiente a
             nascondere la latenza dei lanci.

Il front-end log-mel e' ESCLUSO da entrambe: e' identico nei due modelli e
nessun contatore di FLOPs lo include, quindi tenerlo dentro sposterebbe
entrambe le colonne della stessa quantita' nascondendo il rapporto.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.flops import analyze, benchmark_latency
from src.train import build_frontend, build_model
from src.utils import load_config, provenance

# `dense_flatten` non e' opzionale: e' il baseline ADOTTATO, quello su cui
# poggia il confronto centrale. `dense` (pool_freq) resta perche' e' la
# ricostruzione scartata e il suo costo serve al confronto fra le due
# letture, ma la riga da leggere nei risultati e' quella flatten.
DEFAULT_CONFIGS = [
    "configs/dense.yaml",
    "configs/dense_flatten.yaml",
    "configs/sparse.yaml",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", type=str, nargs="+", default=DEFAULT_CONFIGS)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 64])
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--out", type=str, default="results/benchmark.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        # `mem_get_info` e' a livello di DISPOSITIVO e include il contesto
        # CUDA di questo stesso processo, che da solo vale ~1 GB: leggerlo
        # dopo aver inizializzato torch e chiamarlo "occupato da altri" e'
        # un falso positivo garantito. Si misura quindi la crescita
        # rispetto al contesto appena creato, e si stampa come contesto e
        # non come avviso — la verifica seria si fa con `nvidia-smi` prima
        # di lanciare.
        torch.zeros(1, device="cuda")
        free_after_ctx, total = torch.cuda.mem_get_info()
        ctx_mb = (total - free_after_ctx) / 1e6
        print(f"[gpu] {torch.cuda.get_device_name(0)} · {total / 1e9:.1f} GB · "
              f"contesto di questo processo {ctx_mb:.0f} MB")
        print("[gpu] per una misura pulita nessun altro processo deve usare "
              "la GPU: verificare con `nvidia-smi` prima di lanciare")

    report: dict[str, dict] = {}
    for cfg_path in args.configs:
        cfg = load_config(ROOT / cfg_path)
        frontend = build_frontend(cfg)
        n_frames = frontend.n_frames(cfg["data"]["clip_samples"])
        shape = (1, frontend.n_mels, n_frames)

        model = build_model(cfg, frontend.n_mels, n_frames)
        cost = analyze(build_model(cfg, frontend.n_mels, n_frames), shape)

        entry: dict = {
            "kind": cfg["model"].get("kind"),
            "params": cost.params,
            "gflops": cost.native_flops / 1e9,
            "gmacs": (cost.fvcore_macs / 1e9) if cost.fvcore_macs else None,
            "seq_len": model.seq_len,
            "latency_ms": {},
        }
        for batch in args.batches:
            ms = benchmark_latency(model, shape, device=device,
                                   warmup=args.warmup, iters=args.iters,
                                   batch_size=batch)
            entry["latency_ms"][str(batch)] = ms
        report[Path(cfg_path).stem] = entry

    # ------------------------------------------------------------------
    name_w = max(len(k) for k in report)
    head = f"{'modello':<{name_w}}  {'params':>10}  {'GFLOP':>8}  {'token':>6}"
    for batch in args.batches:
        head += f"  {f'ms @b{batch}':>10}"
    print()
    print(head)
    print("-" * len(head))
    for name, e in report.items():
        line = (f"{name:<{name_w}}  {e['params']:>10,}  {e['gflops']:>8.4f}  "
                f"{e['seq_len']:>6}")
        for batch in args.batches:
            line += f"  {e['latency_ms'][str(batch)]:>10.3f}"
        print(line)

    # I rapporti sono il punto: un fattore N sui FLOPs quanti fattori vale
    # sull'orologio? E' la domanda a cui il paper non risponde.
    if len(report) == 2:
        (na, a), (nb, b) = report.items()
        print()
        print(f"rapporti {nb} / {na}:")
        print(f"  parametri {b['params'] / a['params']:.3f}x    "
              f"FLOPs {b['gflops'] / a['gflops']:.3f}x")
        for batch in args.batches:
            ra = a["latency_ms"][str(batch)]
            rb = b["latency_ms"][str(batch)]
            print(f"  latenza @batch {batch:<3} {rb / ra:.3f}x"
                  f"   ({ra:.3f} -> {rb:.3f} ms)")
        speedup = a["gflops"] / b["gflops"]
        real = a["latency_ms"][str(args.batches[0])] / b["latency_ms"][str(args.batches[0])]
        print(f"\n  un fattore {speedup:.1f} sui FLOPs vale un fattore "
              f"{real:.2f} sull'orologio (batch {args.batches[0]})")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"device": device, "iters": args.iters,
                   "warmup": args.warmup, "provenance": provenance(),
                   "models": report}, fh, indent=2)
    print(f"\nscritto in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
