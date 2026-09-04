"""Verifica la consegna con un comando solo, senza dataset e senza GPU.

    python scripts/verify.py

Tre controlli:

1. i tre config si caricano, i modelli si costruiscono e il forward torna
   [B, 35] con i parametri attesi;
2. i parametri delle nove configurazioni della Tabella 3 stanno entro il 2 %
   dei valori dichiarati dal paper;
3. i numeri riportati nel README si ricalcolano dai run archiviati in
   `results/`, e i sei run valutati sul test set hanno il file di guardia.

Esce con 0 se tutto torna, con 1 altrimenti. Dura una ventina di secondi.
Riprodurre i run da zero e' un'altra cosa, e sta nella sez. 6 del README.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.flops import analyze
from src.train import build_model
from src.utils import load_config

N_FRAMES = 101

# Valori dichiarati da Kavaki & Mandel, trascritti dalle Tabelle 2 e 3.
PAPER_PARAMS_M = {
    ("num_tokens", 4): 2.87, ("num_tokens", 9): 2.87, ("num_tokens", 16): 2.87,
    ("num_tokens", 25): 2.87, ("num_tokens", 36): 2.87,
    ("num_points", 16): 2.44, ("num_points", 36): 2.87,
    ("num_points", 64): 3.53, ("num_points", 128): 5.35,
}
PARAM_TOLERANCE = 0.02

# Numeri riportati nel README, ricalcolati qui dai file archiviati.
EXPECTED_PARAMS = {
    "sparse": 2_822_711, "dense_flatten": 5_197_795, "dense": 4_875_235,
}
EXPECTED_TEST = {"dense_flatten": (97.03, 0.20), "sparse": (95.52, 0.31)}
EXPECTED_RUNS = 22

ok = True


def check(condition: bool, message: str) -> None:
    global ok
    print(f"  [{'ok ' if condition else 'NO '}] {message}")
    ok = ok and condition


def check_models() -> None:
    print("\n1. i modelli si costruiscono dai config")
    for name, expected in EXPECTED_PARAMS.items():
        cfg = load_config(ROOT / "configs" / f"{name}.yaml")
        n_mels = cfg["features"]["n_mels"]
        model = build_model(cfg, n_mels, N_FRAMES).eval()
        with torch.no_grad():
            out = model(torch.randn(2, 1, n_mels, N_FRAMES))
        params = sum(p.numel() for p in model.parameters())
        cost = analyze(build_model(cfg, n_mels, N_FRAMES), (1, n_mels, N_FRAMES))
        check(tuple(out.shape) == (2, 35) and params == expected,
              f"{name + '.yaml':20s} out={tuple(out.shape)}  "
              f"params={params:,}  {cost.flops / 1e9:.4f} GFLOP")


def check_ablation_params() -> None:
    print("\n2. parametri delle nove configurazioni della Tabella 3")
    cfg = load_config(ROOT / "configs" / "sparse.yaml")
    for (key, value), declared in PAPER_PARAMS_M.items():
        c = {**cfg, "model": {**cfg["model"], key: value}}
        params = sum(p.numel() for p in
                     build_model(c, c["features"]["n_mels"], N_FRAMES).parameters())
        gap = params / (declared * 1e6) - 1.0
        check(abs(gap) <= PARAM_TOLERANCE,
              f"{key}={value:<4} {params:>9,}  dichiarati {declared:.2f} M  "
              f"scarto {100 * gap:+.1f} %")


def _accuracy(run: str, filename: str, field: str) -> float | None:
    path = ROOT / "results" / run / filename
    if not path.exists():
        return None
    return 100.0 * json.loads(path.read_text(encoding="utf-8"))[field]


def check_results() -> None:
    print("\n3. i numeri del README vengono dai run archiviati")
    runs = sorted(d for d in (ROOT / "results").iterdir()
                  if (d / "summary.json").exists())
    check(len(runs) == EXPECTED_RUNS, f"{len(runs)} run archiviati")

    incompleti = [d.name for d in runs if not all(
        (d / f).exists() for f in
        ("config.yaml", "resolved_config.json", "metrics.csv", "summary.json"))]
    dettaglio = "" if not incompleti else ": mancano in " + ", ".join(incompleti)
    check(not incompleti, f"ogni run ha config, metriche e summary{dettaglio}")

    for prefix, (mean_atteso, std_atteso) in EXPECTED_TEST.items():
        seeds = [f"{prefix}_seed{s}" for s in (0, 1, 2)]
        acc = [_accuracy(r, "test_report.json", "accuracy") for r in seeds]
        if None in acc:
            check(False, f"{prefix}: manca un test_report.json")
            continue
        mean, std = statistics.mean(acc), statistics.stdev(acc)
        check(abs(mean - mean_atteso) < 0.005 and abs(std - std_atteso) < 0.005,
              f"{prefix + ' (test, 3 seed)':32s} {mean:.2f} % +/- {std:.2f}  "
              f"(README: {mean_atteso:.2f} +/- {std_atteso:.2f})")
        check(all((ROOT / "results" / r / "TEST_EVALUATED.json").exists()
                  for r in seeds),
              f"{prefix}: i tre run hanno il file di guardia sul test set")

    valutati = [d.name for d in runs if (d / "test_report.json").exists()]
    check(len(valutati) == 6,
          f"il test set e' stato toccato su {len(valutati)} run, gli altri "
          f"{len(runs) - len(valutati)} solo su validation")


def main() -> int:
    print("Audio Sparse-Transformer, verifica della consegna")
    print(f"torch {torch.__version__}, {ROOT}")
    check_models()
    check_ablation_params()
    check_results()
    print("\nTutto torna." if ok else "\nQualche controllo non torna.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
