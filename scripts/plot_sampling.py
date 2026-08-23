"""Figura 2 del paper — dove il modello sparso decide di guardare.

    python scripts/plot_sampling.py --run sparse_seed0

Sovrappone allo spettrogramma, per ognuna delle L_rep ripetizioni, le N
regioni e i P punti effettivamente campionati, un colore per token. E' la
verifica QUALITATIVA che il meccanismo faccia quello che dichiara: nel
paper si vede il campionamento passare da uniforme (la griglia iniziale)
a concentrato sulle zone informative.

PERCHE' NON BASTA LA DIAGNOSTICA NUMERICA. `region_init_drift` e
`region_adjust_norm` dicono che le regioni si muovono, non DOVE vanno. Un
modello potrebbe spostarle in modo consistente ma insensato — per esempio
tutte sullo stesso punto, o fuori dal parlato — e le due misure sarebbero
identiche. Questa figura e' l'unico modo di vederlo.

COORDINATE. Il trace restituisce coordinate normalizzate in [0, 1]
riferite alla FEATURE MAP [C, F', T'], non allo spettrogramma. Poiche' i
due condividono gli assi fisici (tempo e frequenza) e la normalizzazione
li rende adimensionali, si disegnano direttamente sopra lo spettrogramma
con `extent=[0, 1, 0, 1]`. E' anche cio' che fa la figura del paper.

Convenzione: `points[..., 0]` e' l'asse x di `grid_sample`, cioe' la
larghezza della feature map, cioe' il TEMPO; `points[..., 1]` e' l'altezza,
cioe' la FREQUENZA.
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

    # La configurazione si legge dal CHECKPOINT, non da `config.yaml`.
    #
    # `config.yaml` e' la copia archiviata del sorgente, che contiene ancora
    # `defaults: base.yaml` — e `base.yaml` sta in configs/, non nella
    # cartella del run: risolverla da li' fallisce. Il checkpoint porta
    # invece la config gia' risolta e, cosa piu' importante, quella
    # REALMENTE usata, inclusi eventuali override da `--set`. E' anche cio'
    # che fa `scripts/evaluate.py`, quindi i due script non possono
    # divergere sulla configurazione dello stesso run.
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

    # Si disegna il modello EMA, lo stesso che produce i numeri riportati:
    # una figura fatta con pesi diversi da quelli valutati mostrerebbe un
    # modello che non esiste in nessuna tabella.
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

            # Le regioni possono uscire dal piano: `grid_sample` usa
            # padding_mode="border", quindi un punto fuori legge il bordo.
            # Si mostra comunque il riquadro unitario per far vedere quanto.
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

    # Geometria per stadio: il riscontro quantitativo dell'impressione
    # visiva, nella stessa esecuzione.
    #
    # Una versione precedente stampava un solo numero — lo scostamento delle
    # regioni FINALI dalla griglia iniziale — che valeva 19106 e non voleva
    # dire niente, perche' sommava stadi con geometrie incomparabili. La
    # grandezza che conta e' quanti punti restano DENTRO il piano: fuori,
    # `padding_mode="border"` li fa leggere il bordo, e il campionamento
    # smette di essere selettivo.
    from src.models.sparse import boxes_to_cwh

    print()
    print("geometria per ripetizione (sui campioni disegnati):")
    for k, stage in enumerate(trace):
        _, size = boxes_to_cwh(stage["boxes"])
        pts = stage["points"]
        inside = float(((pts >= 0) & (pts <= 1)).all(-1).float().mean())
        flag = "" if inside > 0.9 else "   <-- meta' dei punti legge il bordo"
        print(f"  rip. {k + 1}:  dimensione media {float(size.mean()):9.3f}  "
              f"max {float(size.max()):10.3f}   punti dentro il piano "
              f"{100 * inside:5.1f}%{flag}")
    print()
    print("(regioni molto piu' grandi del piano = il campionamento degenera:")
    print(" fuori dal piano il padding 'border' fa leggere il bordo, e il")
    print(" campionamento smette di essere selettivo)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
