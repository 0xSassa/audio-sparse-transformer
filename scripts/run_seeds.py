"""Esegue la stessa configurazione su piu' seed e aggrega media e deviazione.

    python scripts/run_seeds.py --config configs/dense.yaml --seeds 0 1 2

Serve perche' il rumore da seed non e' trascurabile e non e' lo stesso nei
due modelli: 0.04 punti sul denso, 0.37 sullo sparso, dieci volte tanto,
perche' il backward di `grid_sample` non e' deterministico su CUDA. Diverse
delle differenze che il paper discute sono di quell'ordine, e sono riportate
su un solo seed.

Salta i seed gia' completati, cosi' si puo' interrompere e riprendere.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/dense.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--prefix", type=str, default=None)
    ap.add_argument("--force", action="store_true", help="rifai anche i seed gia' fatti")
    args = ap.parse_args()

    prefix = args.prefix or args.config.stem
    results: dict[int, dict] = {}

    for seed in args.seeds:
        tag = f"{prefix}_seed{seed}"
        summary = ROOT / "results" / tag / "summary.json"

        if summary.exists() and not args.force:
            print(f"[skip] {tag} gia' completato")
        else:
            cmd = [sys.executable, "-m", "src.train",
                   "--config", str(args.config), "--seed", str(seed), "--tag", tag]
            if args.epochs is not None:
                cmd += ["--epochs", str(args.epochs)]
            print(f"\n[run] {' '.join(cmd)}\n")
            # check=False per poter riportare quale seed ha fallito
            proc = subprocess.run(cmd, cwd=ROOT, check=False)
            if proc.returncode != 0:
                print(f"[errore] seed {seed} terminato con codice {proc.returncode}")
                return proc.returncode

        if summary.exists():
            results[seed] = json.loads(summary.read_text(encoding="utf-8"))

    if not results:
        return 1

    accs = [r["best_val_accuracy"] for r in results.values()]
    print("\n" + "=" * 60)
    print(f"{prefix}  —  {len(accs)} seed")
    print("=" * 60)
    for seed, r in sorted(results.items()):
        print(f"  seed {seed}: {100*r['best_val_accuracy']:.2f}%")
    mean = statistics.mean(accs)
    std = statistics.stdev(accs) if len(accs) > 1 else 0.0
    print("-" * 60)
    print(f"  media +/- dev.std : {100*mean:.2f} +/- {100*std:.2f} %")
    print(f"  parametri         : {next(iter(results.values()))['params']:,}")

    out = ROOT / "results" / f"{prefix}_aggregate.json"
    out.write_text(json.dumps({
        "prefix": prefix, "seeds": sorted(results),
        "accuracies": {str(k): v["best_val_accuracy"] for k, v in results.items()},
        "mean": mean, "std": std,
    }, indent=2), encoding="utf-8")
    print(f"\n  aggregato in {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
