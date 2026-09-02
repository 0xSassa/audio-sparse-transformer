"""Figura 2 del paper — dove il modello sparso decide di guardare.

    python scripts/plot_sampling.py --run sparse_seed0

Sovrappone allo spettrogramma, per ognuna delle L_rep ripetizioni, le N
regioni e i P punti campionati, un colore per token.

Completa la diagnostica numerica: `region_init_drift` e `region_adjust_norm`
dicono che le regioni si muovono, non DOVE vanno — potrebbero finire tutte
sullo stesso punto, o fuori dal parlato, e le due misure sarebbero identiche.

COORDINATE. Il trace le restituisce normalizzate in [0, 1] sulla FEATURE MAP,
non sullo spettrogramma: i due condividono gli assi fisici, quindi si
disegnano direttamente sopra con `extent=[0, 1, 0, 1]`. `points[..., 0]` e'
l'asse x di `grid_sample`, cioe' il TEMPO; `points[..., 1]` la FREQUENZA.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import SpeechCommandsCached
from src.data.speech_commands import LABELS
from src.train import build_frontend, build_model


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=str, required=True,
                    help="nome della cartella in results/")
    ap.add_argument("--checkpoint", type=str, default="best.pt")
    ap.add_argument("--split", type=str, default="validation",
                    choices=["validation", "test", "train"],
                    help="'test' NON e' ammesso di fatto: si userebbe il test "
                         "set per produrre una figura, che e' comunque un uso")
    ap.add_argument("--samples", type=int, nargs="+", default=None,
                    help="indici dei campioni; default: uno per ognuna di "
                         "quattro parole scelte")
    ap.add_argument("--words", type=str, nargs="+",
                    default=["yes", "no", "left", "stop"],
                    help="parole da cercare, se --samples non e' dato")
    ap.add_argument("--out", type=str, default=None,
                    help="file PNG; default: results/<run>/sampling_trace.png")
    ap.add_argument("--dpi", type=int, default=180)
    ap.add_argument("--device", type=str, default="auto",
                    choices=["auto", "cpu", "cuda"],
                    help="'cpu' per non disturbare un training in corso: "
                         "la figura richiede un solo forward")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")          # nessuna finestra: si scrive un file
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    run_dir = ROOT / "results" / args.run
    ckpt_path = run_dir / args.checkpoint
    if not ckpt_path.exists():
        print(f"[errore] checkpoint inesistente: {ckpt_path}")
        return 1

    # La config si legge dal CHECKPOINT, non da `config.yaml`: quest'ultimo e'
    # la copia del sorgente e contiene ancora `defaults: base.yaml`, che dalla
    # cartella del run non si risolve. Il checkpoint porta la config risolta e
    # REALMENTE usata, override `--set` compresi — come fa `evaluate.py`.
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = state["config"]
    if cfg["model"].get("kind") != "sparse":
        print(f"[errore] il run '{args.run}' non e' un modello sparso: "
              f"kind={cfg['model'].get('kind')!r}. Questa figura ha senso "
              f"solo per il metodo che campiona.")
        return 1

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    frontend = build_frontend(cfg).to(device)
    n_frames = frontend.n_frames(cfg["data"]["clip_samples"])
    model = build_model(cfg, frontend.n_mels, n_frames).to(device)

    # si disegna il modello EMA, lo stesso che produce i numeri riportati:
    # altri pesi mostrerebbero un modello che non sta in nessuna tabella
    model.load_state_dict(state["ema"])
    model.eval()
    print(f"checkpoint  : {args.checkpoint}, epoca {state['epoch']}, "
          f"best {100 * state['best_acc']:.2f}% (pesi EMA)")

    dataset = SpeechCommandsCached(ROOT / cfg["data"]["cache_dir"], args.split)
    labels = np.asarray(dataset.labels)

    if args.samples is not None:
        indices = list(args.samples)
    else:
        indices = []
        for word in args.words:
            if word not in LABELS:
                print(f"[avviso] parola sconosciuta, saltata: {word}")
                continue
            found = np.flatnonzero(labels == LABELS.index(word))
            if found.size:
                indices.append(int(found[0]))
    if not indices:
        print("[errore] nessun campione selezionato")
        return 1

    waves = torch.stack([dataset[i][0] for i in indices]).to(device)
    with torch.no_grad():
        spec = frontend(waves)                      # [B, 1, n_mels, n_frames]
        trace = model.sampling_trace(spec)

    spec_np = spec.squeeze(1).cpu().numpy()
    num_tokens = trace[0]["boxes"].shape[1]
    colors = plt.get_cmap("tab10")(np.arange(num_tokens) % 10)

    rows, cols = len(indices), len(trace)
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 2.35 * rows),
                             squeeze=False)

    for r, idx in enumerate(indices):
        word = LABELS[int(labels[idx])]
        for c, stage in enumerate(trace):
            ax = axes[r][c]
            ax.imshow(spec_np[r], origin="lower", aspect="auto",
                      extent=(0.0, 1.0, 0.0, 1.0), cmap="magma")

            boxes = stage["boxes"][r].cpu().numpy()      # [N, 4] x1,y1,x2,y2
            points = stage["points"][r].cpu().numpy()    # [N, P, 2] x,y

            for t in range(num_tokens):
                x1, y1, x2, y2 = boxes[t]
                ax.add_patch(Rectangle(
                    (x1, y1), x2 - x1, y2 - y1, fill=False,
                    edgecolor=colors[t], linewidth=1.1, alpha=0.9,
                ))
                ax.scatter(points[t, :, 0], points[t, :, 1], s=3.2,
                           color=colors[t], edgecolors="none", alpha=0.95)

            # limiti oltre [0,1]: le regioni possono uscire dal piano, e si
            # vuole vedere di quanto
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"ripetizione {c + 1}", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"«{word}»\nfrequenza →", fontsize=8)
            if r == rows - 1:
                ax.set_xlabel("tempo →", fontsize=8)

    fig.suptitle(
        f"Campionamento sparso lungo le {len(trace)} ripetizioni — "
        f"{args.run}, N={num_tokens}, P={trace[0]['points'].shape[2]}",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out = Path(args.out) if args.out else run_dir / "sampling_trace.png"
    fig.savefig(out, dpi=args.dpi)
    print(f"figura      : {out}")

    # Riscontro quantitativo dell'impressione visiva. La grandezza che conta e'
    # quanti punti restano DENTRO il piano: fuori, `padding_mode="border"` li
    # fa leggere il bordo e il campionamento smette di essere selettivo.
    from src.models.sparse import geometry_from_trace

    print()
    print("geometria per ripetizione (sui soli campioni disegnati):")
    for k, g in enumerate(geometry_from_trace(trace)):
        inside = g["inside"] / g["points"]
        flag = "" if inside > 0.9 else "   <-- meta' dei punti legge il bordo"
        print(f"  rip. {k + 1}:  dimensione media "
              f"{g['size_sum'] / g['boxes']:9.3f}  max {g['size_max']:10.3f}   "
              f"punti dentro il piano {100 * inside:5.1f}%{flag}")
    print()
    print("(regioni molto piu' grandi del piano = il campionamento degenera:")
    print(" fuori dal piano il padding 'border' fa leggere il bordo, e il")
    print(" campionamento smette di essere selettivo)")
    print()
    print("gli stessi conteggi su TUTTO lo split, e su file:")
    print(f"  python scripts/region_geometry.py --run {args.run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
