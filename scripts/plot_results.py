"""Il grafico centrale: accuratezza contro costo, nostro e del paper.

    python scripts/plot_results.py

Legge tutti i `summary.json` sotto `results/`, raggruppa i run per
configurazione (quindi i seed dello stesso modello finiscono insieme,
come media e deviazione standard) e produce tre pannelli:

  A  accuratezza contro FLOPs — il piano su cui il paper argomenta
  B  ablation sul numero di token N
  C  ablation sul numero di campioni P

IL CONFRONTO NON E' ALLA PARI, e la figura lo deve dire: i punti del paper e
i nostri non sono misurati nello stesso modo.

  SPLIT   i loro numeri sono su TEST, i nostri su VALIDATION.
  EPOCHE  le nostre ablation girano a 50 epoche invece di 100, e la
          penalita' MISURATA su N=4 e' 0.80 punti (95.57 -> 94.77): con
          one-cycle un run da 50 epoche non e' la prima meta' di uno da 100.
          L'ordinamento fra configurazioni si conserva, i livelli no.
  COSTO   i FLOPs della Tabella 3 non coincidono con i nostri e lo scarto
          cresce con N (da -11% a +78%), quindi sull'asse x ogni serie porta
          il proprio costo.

Per questo i punti del paper sono vuoti e i nostri pieni, e le tre
avvertenze compaiono nella figura e non solo qui.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.paper import POINTS as PAPER_POINTS
from src.paper import TABLE2 as PAPER_TABLE2
from src.paper import TOKENS as PAPER_TOKENS
from src.train import model_config_with_defaults

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
        # accuratezza su TEST, se quel run e' stato valutato. Il file lo scrive
        # `evaluate.py` una volta sola: la sua presenza e' essa stessa la
        # garanzia che il test set non sia stato riusato.
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
            # la config INTERA, non solo i campi delle etichette: e' cio' che
            # identifica un run
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

    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in runs:
        # La chiave si deriva dalla CONFIG INTERA, non da un elenco di campi
        # scritto a mano: con l'elenco, l'aggiunta di `region_mode` aveva fuso
        # i run a regioni congelate con quelli a regioni apprese in un unico
        # gruppo da sei seed. Difetto invisibile ai controlli ovvi, perche' le
        # due varianti hanno per costruzione gli stessi parametri e FLOPs.
        cfg = model_config_with_defaults(r["model_cfg"])
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
        pa = [p[3] for p in paper_series]
        a2.plot(px, pa, "o--", color=LORO, mfc="none", ms=6, lw=1.2,
                label="paper (test, 100 ep)")

        # la serie nostra: solo i run che variano QUESTO asse tenendo
        # l'altro al valore di default, altrimenti si mescolano famiglie
        other = "num_points" if axis_key == "num_tokens" else "num_tokens"
        default_other = 36 if other == "num_points" else 4

        # Una serie PER BUDGET DI EPOCHE: le ablation girano a 50 epoche e i
        # run di riferimento a 100, e unirli darebbe una curva in cui un
        # gradino e' l'effetto del budget e non del parametro ablato.
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
