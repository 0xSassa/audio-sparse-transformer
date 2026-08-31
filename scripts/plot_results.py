"""Il grafico centrale: accuratezza contro costo, nostro e del paper.

    python scripts/plot_results.py

Legge tutti i `summary.json` sotto `results/`, raggruppa i run per
configurazione (quindi i seed dello stesso modello finiscono insieme,
come media e deviazione standard) e produce tre pannelli:

  A  accuratezza contro FLOPs — il piano su cui il paper argomenta
  B  ablation sul numero di token N
  C  ablation sul numero di campioni P

>>> IL CONFRONTO NON E' ALLA PARI, E LA FIGURA LO DEVE DIRE <<<

I punti del paper e i nostri non sono misurati nello stesso modo, e
sovrapporli senza dirlo sarebbe fuorviante su tre fronti:

  SPLIT     i loro numeri sono su TEST, i nostri su VALIDATION. Il test
            set di questo progetto si tocca una volta sola, alla fine.
  EPOCHE    le nostre ablation girano a 50 epoche invece di 100. La
            penalita' MISURATA su N=4 e' 0,80 punti (95,57 -> 94,77), non
            0,4: un run da 50 epoche NON e' la prima meta' di uno da 100,
            perche' one-cycle lega la forma dello schedule al totale dei
            passi e ricuoce tutto nella meta' del tempo. L'ordinamento fra
            configurazioni si conserva; i livelli no.
  COSTO     i FLOPs dichiarati nella Tabella 3 non coincidono con i nostri,
            e lo scarto CRESCE con N (da -11% a +78%). Sull'asse x quindi
            ogni serie porta il proprio costo, non un costo comune.

Per questo i punti del paper sono disegnati vuoti e i nostri pieni, e le
tre avvertenze compaiono nella figura invece che solo qui.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.train import MODELS, OBSOLETE_MODEL_KEYS

# --------------------------------------------------------------------------
# Numeri dichiarati da Kavaki & Mandel, ICASSP 2025 — trascritti dal PDF.
# Tabella 2: confronto su Google Speech Commands V2 (accuratezza su test).
# Tabella 3: ablation su numero di token e di campioni.
# --------------------------------------------------------------------------
PAPER_TABLE2 = [
    # etichetta,                    params M, GFLOP, accuratezza %
    ("Sparse-transformer",              2.87, 0.055, 96.88),
    ("Dense-transformer",               4.80, 0.645, 96.67),
    ("EAT-S",                           5.30, 0.373, 98.15),
    ("HTS-AT",                         28.80, 4.066, 98.00),
    ("SimPF (MobileNetV2)",             3.40, 0.244, 95.60),
]
PAPER_TOKENS = [  # N, GFLOP, accuratezza %
    (4, 0.05461, 96.88), (9, 0.07366, 97.13), (16, 0.101, 97.53),
    (25, 0.135, 97.20), (36, 0.179, 97.94),
]
PAPER_POINTS = [  # P, GFLOP, accuratezza %
    (16, 0.05164, 96.61), (36, 0.05461, 96.88),
    (64, 0.05919, 96.98), (128, 0.07154, 96.95),
]


def load_runs(results: Path) -> list[dict]:
    """Un record per run, con la configurazione risolta accanto ai numeri."""
    runs = []
    for summary in sorted(results.glob("*/summary.json")):
        with open(summary, encoding="utf-8") as fh:
            s = json.load(fh)
        cfg_path = summary.parent / "resolved_config.json"
        if not cfg_path.exists():
            continue
        with open(cfg_path, encoding="utf-8") as fh:
            cfg = json.load(fh)
        m = cfg.get("model", {})
        # accuratezza su TEST, se quel run e' stato valutato. Il file lo
        # scrive `evaluate.py`, una volta sola per run: la sua presenza e'
        # essa stessa la garanzia che il test set non sia stato riusato.
        test = None
        guard = summary.parent / "TEST_EVALUATED.json"
        if guard.exists():
            with open(guard, encoding="utf-8") as fh:
                test = 100 * json.load(fh)["accuracy"]

        runs.append({
            "tag": s["tag"], "seed": s["seed"], "epochs": s["epochs"], "test": test,
            "acc": 100 * s["best_val_accuracy"], "gflops": s["gflops"],
            "params": s["params"], "kind": m.get("kind"),
            "num_tokens": m.get("num_tokens"), "num_points": m.get("num_points"),
            "repeats": m.get("repeats"), "seq_mode": m.get("seq_mode"),
            "channels": m.get("channel_reading"),
            "constraint": m.get("region_constraint", "none"),
            "mode": m.get("region_mode", "learned"),
            # La configurazione INTERA del modello, non solo i campi che
            # servono alle etichette: e' cio' che identifica un run.
            "model_cfg": m,
        })
    return runs


def group(runs: list[dict]) -> dict[tuple, dict]:
    """Raggruppa per configurazione: i seed dello stesso modello si fondono.

    La chiave NON include il seed, e include invece tutto cio' che cambia il
    modello o il protocollo. Cosi' `sparse_seed0/1/2` diventano un punto solo
    con media e deviazione, mentre `sparse_clip_seed0` resta separato.
    """
    import inspect
    import statistics

    def _hashable(v):
        return tuple(v) if isinstance(v, list) else v

    def _with_defaults(cfg: dict) -> dict:
        """Riempie le opzioni assenti coi default della classe del modello.

        Serve perche' `resolved_config.json` fotografa la config al momento
        del run: un'opzione aggiunta al codice DOPO non compare nei run
        precedenti. Senza questo, lo stesso modello addestrato prima e dopo
        l'aggiunta di un'opzione finirebbe in due gruppi diversi — che e'
        esattamente il difetto opposto a quello che la chiave derivata
        risolve, ed e' altrettanto silenzioso.

        I default si leggono dalla firma della classe, quindi anche questo
        non richiede che nessuno si ricordi di aggiornare un elenco.

        Il caso SPECULARE — un'opzione RIMOSSA dal codice dopo che dei run
        l'avevano archiviata — non si risolve dalla firma, perche' li' non
        c'e' piu' nulla da leggere. Va scartata esplicitamente, ed e' a cosa
        serve `OBSOLETE_MODEL_KEYS`: senza, gli stessi tre seed finirebbero
        in due gruppi a seconda che siano stati addestrati prima o dopo la
        rimozione.
        """
        model = MODELS.get(cfg.get("kind"))
        if model is None:
            return cfg
        out = {k: v for k, v in cfg.items() if k not in OBSOLETE_MODEL_KEYS}
        for name, param in inspect.signature(model.__init__).parameters.items():
            if param.default is not inspect.Parameter.empty and name not in out:
                out[name] = param.default
        return out

    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in runs:
        # LA CHIAVE SI DERIVA DALLA CONFIGURAZIONE INTERA, non da un elenco
        # scritto a mano.
        #
        # La prima versione elencava i campi a mano e ne dimenticava uno:
        # quando e' stata aggiunta l'opzione `region_mode`, i tre run a
        # regioni congelate si sono fusi con i tre a regioni apprese in un
        # unico gruppo da sei seed, con una media che non corrispondeva ad
        # alcun modello esistente. E il difetto era invisibile ai controlli
        # ovvi: le due varianti hanno per costruzione gli STESSI parametri e
        # gli STESSI FLOPs, quindi nessuna verifica di coerenza su quelli
        # avrebbe potuto accorgersene.
        #
        # Derivando la chiave da `model_cfg` il problema non si ripresenta:
        # qualunque opzione nuova entra automaticamente nell'identita' del
        # run, senza che nessuno debba ricordarsene.
        cfg = _with_defaults(r["model_cfg"])
        key = (tuple(sorted((k, _hashable(v)) for k, v in cfg.items())),
               r["epochs"])
        buckets[key].append(r)

    out = {}
    for key, rs in buckets.items():
        accs = [r["acc"] for r in rs]
        tests = [r["test"] for r in rs if r["test"] is not None]
        out[key] = {
            "n_seeds": len(rs), "acc": statistics.mean(accs),
            "std": statistics.stdev(accs) if len(accs) > 1 else 0.0,
            "test": statistics.mean(tests) if tests else None,
            "test_std": statistics.stdev(tests) if len(tests) > 1 else 0.0,
            "n_test": len(tests),
            "gflops": rs[0]["gflops"], "params": rs[0]["params"],
            "tags": sorted(r["tag"] for r in rs),
            **{k: rs[0][k] for k in ("kind", "channels", "seq_mode",
                                     "num_tokens", "num_points", "repeats",
                                     "constraint", "mode", "epochs")},
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=str, default="results/accuracy_vs_flops.png")
    ap.add_argument("--table", type=str, default="results/summary_table.md")
    ap.add_argument("--dpi", type=int, default=180)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = load_runs(ROOT / "results")
    if not runs:
        print("[errore] nessun run trovato in results/")
        return 1
    g = group(runs)
    print(f"{len(runs)} run in {len(g)} configurazioni")

    NOSTRO, LORO = "#1f4e79", "#8a4b12"

    fig = plt.figure(figsize=(11, 7.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1], hspace=0.42, wspace=0.24)
    ax = fig.add_subplot(gs[0, :])

    # ---- A: accuratezza contro costo -------------------------------------
    for lab, _p, f, a in PAPER_TABLE2:
        ax.scatter(f, a, s=70, facecolors="none", edgecolors=LORO,
                   linewidths=1.4, zorder=3)
        ax.annotate(lab, (f, a), textcoords="offset points", xytext=(7, -3),
                    fontsize=7.5, color=LORO)

    flip = True
    for key, v in sorted(g.items(), key=lambda kv: kv[1]["gflops"]):
        if v["kind"] == "dense":
            lab = f"denso {v['seq_mode']}"
            if v["channels"] != "c96":          # due letture dei canali
                lab += f" ({v['channels']})"    # avevano la stessa etichetta
        else:
            lab = f"sparso N={v['num_tokens']} P={v['num_points']}"
            if v["repeats"] != 3:
                lab += f" L={v['repeats']}"
            if v["constraint"] != "none":
                lab += " (vincolato)"
            if v["mode"] != "learned":
                lab += f" (regioni {v['mode']})"
        if v["epochs"] != 100:
            lab += f" · {v['epochs']}ep"
        ax.errorbar(v["gflops"], v["acc"], yerr=v["std"] or None, fmt="o",
                    ms=7, color=NOSTRO, ecolor=NOSTRO, capsize=3, zorder=4)
        ax.annotate(lab, (v["gflops"], v["acc"]), textcoords="offset points",
                    xytext=(7, 4 if flip else -11), fontsize=7.5, color=NOSTRO)
        flip = not flip

    ax.set_xscale("log")
    ax.set_xlabel("costo di un forward (GFLOP, scala logaritmica)")
    ax.set_ylabel("accuratezza (%)")
    ax.set_title("Accuratezza contro costo — nostro (pieno) e dichiarato dal "
                 "paper (vuoto)", fontsize=10.5)
    ax.grid(alpha=0.25, linewidth=0.6)

    # ---- B e C: le due ablation ------------------------------------------
    for idx, (axis_key, paper_series, xlabel, title) in enumerate([
        ("num_tokens", PAPER_TOKENS, "numero di token latenti N",
         "Ablation su N (Tabella 3)"),
        ("num_points", PAPER_POINTS, "numero di campioni P",
         "Ablation su P (Tabella 3)"),
    ]):
        a2 = fig.add_subplot(gs[1, idx])
        px = [p[0] for p in paper_series]
        pa = [p[2] for p in paper_series]
        a2.plot(px, pa, "o--", color=LORO, mfc="none", ms=6, lw=1.2,
                label="paper (test, 100 ep)")

        # la serie nostra: solo i run che variano QUESTO asse tenendo
        # l'altro al valore di default, altrimenti si mescolano famiglie
        other = "num_points" if axis_key == "num_tokens" else "num_tokens"
        default_other = 36 if other == "num_points" else 4

        # UNA SERIE PER BUDGET DI EPOCHE, e non una sola con tutti i punti.
        # Le ablation girano a 50 epoche e i run di riferimento a 100: unirli
        # produrrebbe una curva in cui un gradino di 0,4 punti e' l'effetto
        # del budget e non del parametro ablato — cioe' esattamente
        # l'artefatto che la figura dovrebbe rendere impossibile.
        series: dict[int, list] = defaultdict(list)
        for v in g.values():
            if (v["kind"] == "sparse" and v[other] == default_other
                    and v["repeats"] == 3 and v["constraint"] == "none"
                    and v["mode"] == "learned"):
                series[v["epochs"]].append((v[axis_key], v["acc"], v["std"]))

        for style, (ep, pts) in zip(["o-", "s:", "^-."], sorted(series.items())):
            pts.sort()
            a2.errorbar([p[0] for p in pts], [p[1] for p in pts],
                        yerr=[p[2] or 0 for p in pts], fmt=style, color=NOSTRO,
                        ms=6, lw=1.4, capsize=3,
                        label=f"nostro (validation, {ep} ep)")
        a2.set_xlabel(xlabel)
        a2.set_ylabel("accuratezza (%)")
        a2.set_title(title, fontsize=9.5)
        a2.grid(alpha=0.25, linewidth=0.6)
        a2.legend(fontsize=7.5, loc="lower right")

    fig.suptitle("Audio Sparse-Transformer — riproduzione", fontsize=12.5, y=0.985)
    fig.text(0.5, 0.005,
             "I punti del paper sono su TEST e a 100 epoche; i nostri su "
             "VALIDATION, e le ablation a 50 epoche: MISURATO su N=4, un run "
             "da 50 epoche rende 0,80 punti meno di uno da 100, perche' "
             "one-cycle ricuoce l'intero schedule nella meta' del tempo. "
             "Ogni serie porta il proprio costo: i FLOPs dichiarati nella "
             "Tabella 3 non coincidono con i nostri.",
             ha="center", fontsize=7.2, color="#585e66", wrap=True)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"figura  : {out}")

    # ---- tabella aggregata -----------------------------------------------
    rows = sorted(g.values(), key=lambda v: (v["kind"], v["gflops"]))
    lines = ["| Configurazione | seed | validation | **test** | params | GFLOP |",
             "|---|---|---|---|---|---|"]
    for v in rows:
        if v["kind"] == "dense":
            name = f"denso `{v['seq_mode']}` ({v['channels']})"
        else:
            name = f"sparso N={v['num_tokens']} P={v['num_points']}"
            if v["repeats"] != 3:
                name += f" L_rep={v['repeats']}"
            if v["constraint"] != "none":
                name += " vincolato"
            if v["mode"] != "learned":
                name += f", regioni {v['mode']}"
        if v["epochs"] != 100:
            name += f", {v['epochs']} ep"
        acc = (f"{v['acc']:.2f} % ± {v['std']:.2f}" if v["n_seeds"] > 1
               else f"{v['acc']:.2f} %")
        if v["test"] is None:
            test = "—"
        elif v["n_test"] > 1:
            test = f"**{v['test']:.2f} % ± {v['test_std']:.2f}**"
        else:
            test = f"**{v['test']:.2f} %**"
        lines.append(f"| {name} | {v['n_seeds']} | {acc} | {test} | "
                     f"{v['params']:,} | {v['gflops']:.4f} |")
    table = "\n".join(lines) + "\n"
    (ROOT / args.table).write_text(table, encoding="utf-8")
    print(f"tabella : {ROOT / args.table}\n")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
