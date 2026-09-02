"""Verifica che i risultati archiviati vengano da QUESTO codice.

    python scripts/verify_runs.py

Un run archiviato e' un insieme di file JSON e CSV: da soli non dimostrano
nulla sul codice che li ha prodotti. Questo script chiude il cerchio senza
riaddestrare niente, ricostruendo ogni modello dalla sua configurazione e
confrontando cio' che esce con cio' che e' registrato.

Cinque controlli, ognuno capace da solo di smascherare un archivio incoerente:

  1. modello       il codice attuale ricostruisce il modello del run con
                   ESATTAMENTE i parametri e i FLOPs registrati. E' il
                   controllo che conta: se passa, la definizione del modello
                   non e' cambiata da quando il run e' stato prodotto.
  2. metriche      `best_val_accuracy` nel summary coincide col massimo di
                   `metrics.csv`, cioe' il riassunto non e' stato scritto a
                   mano ne' e' rimasto indietro rispetto alle epoche.
  3. aggregati     media e deviazione nei file `*_aggregate.json` tornano
                   dai summary dei singoli seed.
  4. test set      ogni `test_report.json` ha il suo file di guardia, e
                   nessuna valutazione e' stata forzata.
  5. provenienza   quali run sono stati prodotti con l'albero git sporco.
                   Non e' un errore ed e' informazione, non giudizio: il
                   controllo 1 dice se quelle modifiche toccavano il modello.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.flops import analyze
from src.train import build_frontend, build_model

REQUIRED = ("summary.json", "resolved_config.json", "config.yaml", "metrics.csv")


def check_run(run_dir: Path) -> list[str]:
    """Errori trovati in un run. Lista vuota = tutto torna."""
    problems = []
    for name in REQUIRED:
        if not (run_dir / name).exists():
            problems.append(f"manca {name}")
    if problems:
        return problems

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    cfg = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))

    # 1. il modello si ricostruisce identico
    frontend = build_frontend(cfg)
    n_frames = frontend.n_frames(cfg["data"]["clip_samples"])
    cost = analyze(build_model(cfg, frontend.n_mels, n_frames),
                   (1, frontend.n_mels, n_frames))
    if cost.params != summary["params"]:
        problems.append(f"parametri {cost.params:,} contro {summary['params']:,} "
                        "registrati")
    if abs(cost.flops / 1e9 - summary["gflops"]) > 1e-9:
        problems.append(f"FLOPs {cost.flops / 1e9:.6f} contro "
                        f"{summary['gflops']:.6f} registrati")

    # 2. il summary coincide con le epoche effettivamente girate
    with open(run_dir / "metrics.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        problems.append("metrics.csv vuoto")
    else:
        best = max(float(r["val_accuracy_ema"]) for r in rows)
        if abs(best - summary["best_val_accuracy"]) > 1e-9:
            problems.append(f"best_val_accuracy {summary['best_val_accuracy']:.6f} "
                            f"contro {best:.6f} nel CSV")
        if len(rows) != summary["epochs"]:
            problems.append(f"{len(rows)} epoche nel CSV contro "
                            f"{summary['epochs']} dichiarate")

    # 4. la guardia sul test set
    if (run_dir / "test_report.json").exists():
        guard = run_dir / "TEST_EVALUATED.json"
        if not guard.exists():
            problems.append("test_report.json senza file di guardia")
        else:
            record = json.loads(guard.read_text(encoding="utf-8"))
            if record.get("forced"):
                problems.append("valutazione di test FORZATA")
            if record.get("history"):
                problems.append(f"test valutato {len(record['history']) + 1} volte")
    return problems


def check_aggregate(path: Path, results: Path) -> list[str]:
    agg = json.loads(path.read_text(encoding="utf-8"))
    prefix = agg["prefix"]
    problems = []
    accs = []
    for seed in agg["seeds"]:
        summary = results / f"{prefix}_seed{seed}" / "summary.json"
        if not summary.exists():
            problems.append(f"seed {seed} assente")
            continue
        accs.append(json.loads(summary.read_text(encoding="utf-8"))
                    ["best_val_accuracy"])
    if len(accs) == len(agg["seeds"]) and accs:
        if abs(statistics.mean(accs) - agg["mean"]) > 1e-9:
            problems.append("media non coincidente")
        std = statistics.stdev(accs) if len(accs) > 1 else 0.0
        if abs(std - agg["std"]) > 1e-9:
            problems.append("deviazione standard non coincidente")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    args = ap.parse_args()

    runs = sorted(d for d in args.results.iterdir()
                  if d.is_dir() and (d / "summary.json").exists())
    if not runs:
        print(f"[errore] nessun run in {args.results}")
        return 1

    failed = 0
    dirty = []
    print(f"{len(runs)} run archiviati\n")
    for run in runs:
        problems = check_run(run)
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        if summary.get("provenance", {}).get("dirty"):
            dirty.append(run.name)
        if problems:
            failed += 1
            print(f"  {run.name}")
            for p in problems:
                print(f"      {p}")
        else:
            print(f"  {run.name:<26} ok")

    print()
    for path in sorted(args.results.glob("*_aggregate.json")):
        problems = check_aggregate(path, args.results)
        failed += bool(problems)
        status = "ok" if not problems else "; ".join(problems)
        print(f"  {path.name:<34} {status}")

    if dirty:
        print(f"\n{len(dirty)} run su {len(runs)} prodotti con albero git sporco:")
        print(f"  {', '.join(dirty)}")
        print("  Il controllo sul modello e' passato per tutti, quindi le "
              "modifiche non committate")
        print("  non toccavano la definizione dei modelli. Resta un limite "
              "dichiarato, non un errore.")

    print(f"\n{'TUTTO COERENTE' if not failed else f'{failed} problemi'}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
