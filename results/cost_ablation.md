# Costo dichiarato contro costo misurato

Generato da `python scripts/cost_ablation.py`. Nessun training: params e FLOPs di un forward con batch 1, `grid_sample` esclusa.

## Ablation su N (token), a P=36

| N | params nostri | dichiarati | scarto | MFLOP nostri | dichiarati | scarto |
|---|---|---|---|---|---|---|
| 4 | 2,822,711 | 2.87 M | -1.6 % | 48.51 | 54.61 | -11.2 % |
| 9 | 2,823,051 | 2.87 M | -1.6 % | 90.13 | 73.66 | +22.4 % |
| 16 | 2,823,527 | 2.87 M | -1.6 % | 148.74 | 101.00 | +47.3 % |
| 25 | 2,824,139 | 2.87 M | -1.6 % | 224.69 | 135.00 | +66.4 % |
| 36 | 2,824,887 | 2.87 M | -1.6 % | 318.42 | 179.00 | +77.9 % |

## Ablation su P (campioni per token), a N=4

| P | params nostri | dichiarati | scarto | MFLOP nostri | dichiarati | scarto |
|---|---|---|---|---|---|---|
| 16 | 2,393,231 | 2.44 M | -1.9 % | 38.28 | 51.64 | -25.9 % |
| 36 | 2,822,711 | 2.87 M | -1.6 % | 48.51 | 54.61 | -11.2 % |
| 64 | 3,492,527 | 3.53 M | -1.1 % | 66.45 | 59.19 | +12.3 % |
| 128 | 5,323,823 | 5.35 M | -0.5 % | 123.27 | 71.54 | +72.3 % |

## Come cresce il costo con N

| | pendenza (MFLOP per token) | parte fissa (MFLOP) |
|---|---|---|
| nostro | 8.44 | 14.3 |
| implicito nella Tabella 3 | 3.88 | 38.8 |
| rapporto | 2.17x | 0.37x |

La parte fissa e' il front-end convolutivo e dipende dalle nostre assunzioni sullo spettrogramma. La pendenza no:

| n_mels | pendenza (MFLOP per token) | parte fissa (MFLOP) |
|---|---|---|
| 32 | 8.44 | 6.6 |
| 64  (adottato) | 8.44 | 14.3 |
| 128 | 8.44 | 29.6 |

Cambiando n_mels di un fattore 4 la parte fissa si sposta e la pendenza resta dov'e': e' fissata solo da quantita' che il paper dichiara.

## Il front-end si recupera dai loro numeri?

No. Tenendo l'hop a 10 ms, che e' la convenzione della letteratura su Speech Commands (AST usa 25/10 ms, KWT 30/10 ms) e che su un secondo di audio da' circa 100 frame, il bersaglio di 109.2 MFLOP non si raggiunge a nessun numero di bin:

| bin mel, hop 10 ms | 40 | 64 | 80 | 96 | 128 |
|---|---|---|---|---|---|
| scarto dal costo dichiarato | -61 % | -56 % | -52 % | -49 % | -42 % |

Per chiudere servirebbe un hop di 4-5 ms, che nessun lavoro su questo dataset usa. Il modello denso non entra nel conto: `dim` e `depth` non sono dichiarati dal paper, li abbiamo ricostruiti noi.

## Il costo di `grid_sample`, che nessun contatore vede

Il campionamento bilineare non e' una matmul, quindi non compare in nessuna delle tabelle sopra. Nella configurazione adottata sono 432 punti su 96 canali, cioe' 0.292 MFLOP: lo 0.60 % del costo totale del modello sparso. Irrilevante nel numero, ma e' il meccanismo che il paper difende, e ometterlo senza mostrarne il conto sarebbe scorretto.
