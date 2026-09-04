"""Valutazione finale sul test set, con guardia contro l'uso ripetuto.

    python scripts/evaluate.py --run dense_seed0

Valutare piu' volte il test set, cambiando qualcosa in mezzo, equivale a
selezionare su di esso, ed e' un errore senza sintomi visibili. Al primo uso lo
script scrive `TEST_EVALUATED.json` nella cartella del run e poi si rifiuta di
ripartire, salvo `--force`, che resta registrato nel file. Il checkpoint e' gia'
stato selezionato sul validation, durante il training.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import make_loader
from src.data.speech_commands import LABELS
from src.metrics import classification_summary, print_summary, save_summary
from src.train import build_frontend, build_model, evaluate


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=str, required=True, help="nome della cartella in results/")
    ap.add_argument("--checkpoint", type=str, default="best.pt")
    ap.add_argument("--split", type=str, default="test", choices=["test", "validation"])
    ap.add_argument("--force", action="store_true",
                    help="rivaluta il test set nonostante la guardia (registrato)")
    args = ap.parse_args()

    run_dir = ROOT / "results" / args.run
    if not run_dir.exists():
        print(f"[errore] run inesistente: {run_dir}")
        return 1

    guard = run_dir / "TEST_EVALUATED.json"
    if args.split == "test" and guard.exists() and not args.force:
        prev = json.loads(guard.read_text(encoding="utf-8"))
        print(f"[bloccato] il test set di questo run e' gia' stato valutato "
              f"il {prev['when']}: accuratezza {100*prev['accuracy']:.2f}%.")
        print("           Rivalutarlo significa selezionare sul test set.")
        print("           Se e' davvero necessario: --force (verra' registrato).")
        return 2

    ckpt_path = run_dir / args.checkpoint
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = state["config"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frontend = build_frontend(cfg).to(device)
    n_frames = frontend.n_frames(cfg["data"]["clip_samples"])
    model = build_model(cfg, frontend.n_mels, n_frames).to(device)
    model.load_state_dict(state["ema"])          # si valuta il modello EMA

    loader = make_loader(ROOT / cfg["data"]["cache_dir"], args.split,
                         batch_size=256, num_workers=cfg["data"]["num_workers"])
    result = evaluate(model, frontend, loader, device, cfg["data"]["num_classes"],
                      collect=True)
    report = classification_summary(result["y_true"], result["y_pred"], list(LABELS))
    save_summary(report, run_dir / f"{args.split}_report.json")

    print(f"run          : {args.run}  (checkpoint {args.checkpoint}, epoca {state['epoch']})")
    print(f"split        : {args.split}  ({result['n']:,} campioni)")
    print(f"loss         : {result['loss']:.4f}")
    print()
    print_summary(report)

    if args.split == "test":
        record = {
            "when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "checkpoint": args.checkpoint,
            "epoch": state["epoch"],
            "accuracy": result["accuracy"],
            "n": result["n"],
            "forced": bool(args.force),
        }
        history = []
        if guard.exists():
            old = json.loads(guard.read_text(encoding="utf-8"))
            history = old.get("history", []) + [{k: old[k] for k in
                                                 ("when", "accuracy", "forced")}]
        record["history"] = history
        guard.write_text(json.dumps(record, indent=2), encoding="utf-8")
        if history:
            print(f"\n[attenzione] valutazione forzata numero {len(history)+1} "
                  f"su questo run: il numero non e' piu' una stima onesta.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
