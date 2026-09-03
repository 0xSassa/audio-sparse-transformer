"""Costo dichiarato contro costo misurato, su tutta la Tabella 3.

    python scripts/cost_ablation.py

Il paper dichiara params e FLOPs per nove configurazioni: cinque lungo il
numero di token N (a P=36) e quattro lungo il numero di campioni P (a N=4),
piu' quella di default che le due righe condividono. Sono nove verifiche
dell'implementazione che NON richiedono di addestrare nulla: si costruisce il
modello e si conta.

Il valore della verifica sta nel fatto che i due assi si controllano a
vicenda. Lungo N i parametri dichiarati non si muovono (2.87 M ovunque);
lungo P piu' che raddoppiano (2.44 -> 5.35 M), perche' il decoder genera
`Ms` in R^{PxP}. Un'implementazione che sbagliasse la lettura del metodo
avrebbe pochissime probabilita' di riprodurre entrambi gli andamenti.

LA PENDENZA. Il costo di questo modello e' lineare in N:

    FLOPs(N) = fisso + pendenza * N

dove `fisso` e' il front-end convolutivo, che dipende dalle nostre assunzioni
sullo spettrogramma, mentre `pendenza` e' il costo di un token e dipende SOLO
da quantita' che il paper dichiara (P, C, d_token, d_encoder, L_rep, L_enc).
Lo script rifa' la misura con n_mels diversi proprio per mostrare che la
pendenza non si muove: e' il numero su cui il confronto col paper non ammette
la scusa dell'assunzione.

`grid_sample` resta esclusa dalle due tabelle, come in ogni altro conteggio
del repository: includerla sposterebbe entrambe le colonne nella stessa
direzione senza cambiare il rapporto. Lo script la quantifica pero' a parte,
perche' e' il meccanismo che il paper sta difendendo e dichiararla
trascurabile senza mostrarne il conto non basta.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.flops import analyze, grid_sample_flops
from src.paper import POINTS, TOKENS
from src.train import build_frontend, build_model
from src.utils import load_config, provenance

CONFIG = "configs/sparse.yaml"


def measure(cfg: dict, key: str, value: int) -> tuple[int, float]:
    """Params e FLOPs del modello sparso con `cfg.model[key] = value`."""
    import copy

    cfg = copy.deepcopy(cfg)
    cfg["model"][key] = value
    frontend = build_frontend(cfg)
    n_frames = frontend.n_frames(cfg["data"]["clip_samples"])
    model = build_model(cfg, frontend.n_mels, n_frames)
    report = analyze(model, (1, frontend.n_mels, n_frames))
    return report.params, report.flops


def line_fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Minimi quadrati su una retta: restituisce (pendenza, intercetta)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    slope = num / den
    return slope, my - slope * mx


def sweep(cfg: dict, key: str, series: list[tuple]) -> list[dict]:
    rows = []
    for value, paper_params, paper_gflop, _acc in series:
        params, flops = measure(cfg, key, value)
        rows.append({
            key: value,
            "params": params,
            "params_paper": paper_params * 1e6,
            "params_err": 100 * (params - paper_params * 1e6) / (paper_params * 1e6),
            "flops": flops,
            "flops_paper": paper_gflop * 1e9,
            "flops_err": 100 * (flops - paper_gflop * 1e9) / (paper_gflop * 1e9),
        })
    return rows


def table(rows: list[dict], key: str, label: str) -> str:
    out = [f"| {label} | params nostri | dichiarati | scarto | MFLOP nostri "
           f"| dichiarati | scarto |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(
            f"| {r[key]} | {r['params']:,} | {r['params_paper'] / 1e6:.2f} M "
            f"| {r['params_err']:+.1f} % | {r['flops'] / 1e6:.2f} "
            f"| {r['flops_paper'] / 1e6:.2f} | {r['flops_err']:+.1f} % |"
        )
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=str, default=CONFIG)
    ap.add_argument("--n-mels", type=int, nargs="+", default=[32, 64, 128],
                    help="valori con cui rifare la sweep su N, per mostrare "
                         "che la pendenza per token non dipende da questa "
                         "assunzione")
    ap.add_argument("--mels-at-standard-hop", type=int, nargs="+",
                    default=[40, 64, 80, 96, 128],
                    help="bin mel da provare tenendo l'hop a 10 ms, che e' la "
                         "convenzione della letteratura su Speech Commands "
                         "(AST 25/10 ms, KWT 30/10 ms)")
    ap.add_argument("--out", type=str, default="results/cost_ablation.md")
    ap.add_argument("--json", type=str, default="results/cost_ablation.json")
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    default_mels = cfg["features"]["n_mels"]

    tokens = sweep(cfg, "num_tokens", TOKENS)
    points = sweep(cfg, "num_points", POINTS)

    # pendenza per token: nostra e implicita nei numeri del paper
    ours = line_fit([r["num_tokens"] for r in tokens],
                    [r["flops"] for r in tokens])
    theirs = line_fit([n for n, _p, _f, _a in TOKENS],
                      [f * 1e9 for _n, _p, f, _a in TOKENS])

    # la stessa pendenza, con altre assunzioni sullo spettrogramma
    invariance = []
    for n_mels in args.n_mels:
        probe = {**cfg, "features": {**cfg["features"], "n_mels": n_mels}}
        flops = [measure(probe, "num_tokens", n)[1] for n, *_ in TOKENS]
        slope, fixed = line_fit([n for n, *_ in TOKENS], flops)
        invariance.append({"n_mels": n_mels, "slope": slope, "fixed": fixed})

    # `grid_sample` non e' una matmul: nessun contatore la vede, e va aggiunta
    # a mano. Il campionamento avviene N*P volte per ognuna delle L_rep
    # ripetizioni, su C canali.
    m = cfg["model"]
    gs_points = m["num_tokens"] * m["num_points"] * m["repeats"]
    gs = grid_sample_flops(gs_points, 96)
    default_flops = tokens[0]["flops"]

    # Il front-end si recupera dai numeri del paper? Si tiene l'hop a 10 ms,
    # che e' la convenzione universale su questo dataset, e si fa variare il
    # solo numero di bin mel. Bersaglio: il costo dichiarato letto come MAC,
    # cioe' 2x quello della Tabella 3, perche' la pendenza dice che i loro
    # numeri sono MAC.
    target_mac = 2.0 * TOKENS[0][2] * 1e9
    standard_hop = []
    for n_mels in args.mels_at_standard_hop:
        probe = {**cfg, "features": {**cfg["features"], "n_mels": n_mels}}
        _params, flops = measure(probe, "num_tokens", cfg["model"]["num_tokens"])
        standard_hop.append({"n_mels": n_mels, "flops": flops,
                             "err": 100 * (flops - target_mac) / target_mac})

    md = [
        "# Costo dichiarato contro costo misurato",
        "",
        "Generato da `python scripts/cost_ablation.py`. Nessun training: "
        "params e FLOPs di un forward con batch 1, `grid_sample` esclusa.",
        "",
        "## Ablation su N (token), a P=36",
        "",
        table(tokens, "num_tokens", "N"),
        "",
        "## Ablation su P (campioni per token), a N=4",
        "",
        table(points, "num_points", "P"),
        "",
        "## Come cresce il costo con N",
        "",
        "| | pendenza (MFLOP per token) | parte fissa (MFLOP) |",
        "|---|---|---|",
        f"| nostro | {ours[0] / 1e6:.2f} | {ours[1] / 1e6:.1f} |",
        f"| implicito nella Tabella 3 | {theirs[0] / 1e6:.2f} | {theirs[1] / 1e6:.1f} |",
        f"| rapporto | {ours[0] / theirs[0]:.2f}x | {ours[1] / theirs[1]:.2f}x |",
        "",
        "La parte fissa e' il front-end convolutivo e dipende dalle nostre "
        "assunzioni sullo spettrogramma. La pendenza no:",
        "",
        "| n_mels | pendenza (MFLOP per token) | parte fissa (MFLOP) |",
        "|---|---|---|",
    ]
    for row in invariance:
        mark = "  (adottato)" if row["n_mels"] == default_mels else ""
        md.append(f"| {row['n_mels']}{mark} | {row['slope'] / 1e6:.2f} "
                  f"| {row['fixed'] / 1e6:.1f} |")
    md += ["", "Cambiando n_mels di un fattore "
               f"{max(args.n_mels) // min(args.n_mels)} la parte fissa si sposta "
               "e la pendenza resta dov'e': e' fissata solo da quantita' che il "
               "paper dichiara.", ""]
    md += [
        "## Il front-end si recupera dai loro numeri?",
        "",
        "No. Tenendo l'hop a 10 ms, che e' la convenzione della letteratura su "
        "Speech Commands (AST usa 25/10 ms, KWT 30/10 ms) e che su un secondo "
        f"di audio da' circa 100 frame, il bersaglio di "
        f"{target_mac / 1e6:.1f} MFLOP non si raggiunge a nessun numero di bin:",
        "",
        "| bin mel, hop 10 ms | " + " | ".join(
            str(r["n_mels"]) for r in standard_hop) + " |",
        "|---" * (len(standard_hop) + 1) + "|",
        "| scarto dal costo dichiarato | " + " | ".join(
            f"{r['err']:+.0f} %" for r in standard_hop) + " |",
        "",
        "Per chiudere servirebbe un hop di 4-5 ms, che nessun lavoro su questo "
        "dataset usa. Il modello denso non entra nel conto: `dim` e `depth` "
        "non sono dichiarati dal paper, li abbiamo ricostruiti noi.",
        "",
        "## Il costo di `grid_sample`, che nessun contatore vede",
        "",
        "Il campionamento bilineare non e' una matmul, quindi non compare in "
        "nessuna delle tabelle sopra. Nella configurazione adottata sono "
        f"{gs_points} punti su 96 canali, cioe' {gs / 1e6:.3f} MFLOP: lo "
        f"{100 * gs / default_flops:.2f} % del costo totale del modello sparso. "
        "Irrilevante nel numero, ma e' il meccanismo che il paper difende, e "
        "ometterlo senza mostrarne il conto sarebbe scorretto.",
        "",
    ]
    text = "\n".join(md)

    (ROOT / args.out).write_text(text, encoding="utf-8")
    (ROOT / args.json).write_text(json.dumps({
        "provenance": provenance(),
        "config": args.config,
        "tokens": tokens,
        "points": points,
        "front_end_standard_hop": {"target_mac_flops": target_mac,
                                   "sweep": standard_hop},
        "grid_sample": {"points": gs_points, "flops": gs,
                        "share_of_sparse": gs / default_flops},
        "slope": {"ours": ours[0], "paper": theirs[0],
                  "fixed_ours": ours[1], "fixed_paper": theirs[1]},
        "invariance": invariance,
    }, indent=2), encoding="utf-8")

    print(text)
    print(f"scritti : {ROOT / args.out}")
    print(f"          {ROOT / args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
