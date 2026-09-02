"""Quanto del campionamento sparso resta dentro il piano tempo-frequenza.

    python scripts/region_geometry.py --run sparse_seed0

La Figura 2 del paper (`scripts/plot_sampling.py`) mostra quattro esempi. Qui
si conta la stessa cosa sull'intero split di validation, e il risultato
finisce in `results/<run>/region_geometry.json`, che e' versionato: la
misura diventa citabile senza rieseguire nulla.

Due grandezze, che rispondono a due domande diverse.

PUNTI DENTRO IL PIANO. Il metodo non impedisce alle regioni di uscire da [0,1]^2, e
`grid_sample` serve i punti fuori con `padding_mode="border"`: leggono il
bordo della feature map. Il modello continua ad addestrarsi, la loss scende,
e una parte del campionamento non guarda piu' il segnale. E' il modo in cui
il meccanismo centrale del paper puo' degenerare restando invisibile a
qualunque metrica di accuratezza.

SCALE TAGLIATE. `MAX_LOG_SCALE` e' una NOSTRA aggiunta, non c'e' nel paper:
senza, `exp()` di un delta anomalo darebbe regioni infinite e NaN nel
gradiente. Serve a sapere quanto spesso morde, cioe' quanto il modello
addestrato si discosta dalla lettera del metodo. Zero = il clamp e' inerte e
l'aggiunta e' gratis; molto sopra zero = il modello vive sul limite che gli
abbiamo imposto, e va detto.

Il test set non e' fra le opzioni: questa e' diagnostica, e la diagnostica su
test e' comunque uso del test.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import make_loader
from src.train import (build_frontend, build_model,
                       model_config_with_defaults)
from src.utils import provenance


def aggregate(model, frontend, loader, device, max_batches: int | None) -> list[dict]:
    """Somma i conteggi per ripetizione su tutto lo split."""
    totals: list[dict] = []
    for i, (wave, _target) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        spec = frontend(wave.to(device))
        for k, stage in enumerate(model.sampling_geometry(spec)):
            if k == len(totals):
                totals.append({key: 0 for key in stage})
            for key, value in stage.items():
                if key == "size_max":
                    totals[k][key] = max(totals[k][key], value)
                else:
                    totals[k][key] += value
    return totals


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=str, required=True,
                    help="nome della cartella in results/")
    ap.add_argument("--checkpoint", type=str, default="best.pt")
    ap.add_argument("--split", type=str, default="validation",
                    choices=["validation", "train"])
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-batches", type=int, default=None,
                    help="per una misura rapida; default: tutto lo split")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    run_dir = ROOT / "results" / args.run
    ckpt = run_dir / args.checkpoint
    if not ckpt.exists():
        print(f"[errore] checkpoint assente: {ckpt}")
        print("         i pesi non sono versionati: serve il run in locale")
        return 1

    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = state["config"]
    # i run archiviati prima che un'opzione esistesse non la portano
    model_cfg = model_config_with_defaults(cfg["model"])
    if model_cfg["kind"] != "sparse":
        print(f"[errore] '{args.run}' e' un modello {model_cfg['kind']}: "
              "le regioni esistono solo nel modello sparso")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frontend = build_frontend(cfg).to(device)
    n_frames = frontend.n_frames(cfg["data"]["clip_samples"])
    model = build_model(cfg, frontend.n_mels, n_frames).to(device)
    model.load_state_dict(state["ema"])      # lo stesso modello che si valuta
    model.eval()

    loader = make_loader(ROOT / cfg["data"]["cache_dir"], args.split,
                         batch_size=args.batch_size,
                         num_workers=cfg["data"]["num_workers"])
    totals = aggregate(model, frontend, loader, device, args.max_batches)

    stages = []
    for k, t in enumerate(totals):
        stages.append({
            "repeat": k + 1,
            "inside_fraction": t["inside"] / t["points"],
            "size_mean": t["size_sum"] / t["boxes"],
            "size_max": t["size_max"],
            "clamped_fraction": t["clamped"] / t["scales"],
            "points": t["points"],
        })

    report = {
        "run": args.run,
        "checkpoint": args.checkpoint,
        "epoch": state["epoch"],
        "split": args.split,
        "num_tokens": model_cfg["num_tokens"],
        "num_points": model_cfg["num_points"],
        "repeats": model_cfg["repeats"],
        "region_constraint": model_cfg["region_constraint"],
        "region_mode": model_cfg["region_mode"],
        "unit": model_cfg["unit"],
        "stages": stages,
        "provenance": provenance(),
    }
    out = Path(args.out) if args.out else run_dir / "region_geometry.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"run       : {args.run}  (epoca {state['epoch']}, split {args.split}, "
          f"{stages[0]['points']:,} punti per ripetizione)")
    print(f"vincolo   : region_constraint={report['region_constraint']}  "
          f"region_mode={report['region_mode']}")
    print()
    print("  rip.   punti dentro il piano   lato medio   lato massimo   "
          "scale tagliate")
    for s in stages:
        flag = "" if s["inside_fraction"] > 0.9 else "   <-- degenerato"
        print(f"  {s['repeat']:>4}   {100 * s['inside_fraction']:>19.1f} %   "
              f"{s['size_mean']:>10.3f}   {s['size_max']:>12.3f}   "
              f"{100 * s['clamped_fraction']:>12.1f} %{flag}")
    print()
    print(f"scritto   : {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
