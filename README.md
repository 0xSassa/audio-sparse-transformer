# Audio Sparse-Transformer — riproduzione

Progetto d'esame per **B031278 Deep Learning** (Fall 2025, prof. Paolo Frasconi),
MSc in Artificial Intelligence, Università di Firenze.

Riproduzione semplificata di:

> H. Salami Kavaki, M. I. Mandel, *Audio Sparse-Transformer for Speech
> Classification*, ICASSP 2025, pp. 1–5. DOI 10.1109/ICASSP49660.2025.10890475

con riferimento a:

> A. Gazneli, G. Zimerman, T. Ridnik, G. Sharir, A. Noy, *End-to-End Audio
> Strikes Back*, arXiv:2204.11479, 2022.

Il metodo è la trasposizione all'audio di **SparseFormer** (Gao et al.,
arXiv:2304.03768); la parametrizzazione delle regioni viene da Faster R-CNN,
il decoding adattivo da AdaMixer, il front-end da Xiao et al. 2021.

Non esiste codice pubblico per Kavaki & Mandel: modello, training e misure
sono reimplementati dalle formule del paper. Dataset: Google Speech
Commands V2, 35 classi, split ufficiali speaker-disjoint.

---

## Risultati

Modello EMA. Il test set è stato toccato **una volta sola per modello**, a
esperimenti chiusi; media ± deviazione standard su 3 seed dove indicato.

| Modello | **Test** (3 seed) | Validation | Params | GFLOP |
|---|---|---|---|---|
| **Denso `flatten`** — baseline adottato | **97,03 % ± 0,20** | 96,97 ± 0,04 | 5 197 795 | 0,5604 |
| **Sparso N=4** — configurazione del paper | **95,52 % ± 0,31** | 95,71 ± 0,37 | 2 822 711 | 0,0485 |
| Sparso N=16 | — | 96,46 % | 2 823 527 | 0,1487 |
| Sparso, regioni congelate su griglia — 3 seed | — | 94,94 % ± 0,22 | 2 822 711 | 0,0485 |
| Sparso, `L_rep`=1 | — | 94,98 % | 2 010 591 | 0,0349 |
| Sparso, regioni vincolate al piano | — | 95,21 % | 2 822 711 | 0,0485 |
| Denso `pool_freq` — ricostruzione scartata | — | 94,09 % | 4 875 235 | 0,5275 |

La tabella completa di tutti i 19 run, ablation a 50 epoche comprese, è
`results/summary_table.md`; la figura è `results/accuracy_vs_flops.png`.
Entrambe si rigenerano con `python scripts/plot_results.py`.

L'obiettivo era **riprodurre l'esperimento e capirlo**, non far tornare le
cifre: uno scarto di qualche punto sui valori assoluti è atteso e nasce dalle
assunzioni che il paper lascia aperte (elencate sotto). Su quel piano
l'esperimento è riprodotto — il nostro denso è mezzo punto *sopra* il valore
che gli autori dichiarano per il loro.

### Il risultato è il confronto interno, e cambia segno

Il paper dà il modello sparso **0,21 punti sopra** il proprio
dense-transformer. Con la stessa pipeline, lo stesso front-end e lo stesso
budget di epoche, il nostro resta **1,52 punti sotto sul test set**
(*t* = 7,2; −1,26 su validation). È un cambio di *segno* su un confronto
controllato, che nessuna tolleranza sui livelli assorbe.

Tre risultati collaterali che il paper non riporta:

- **Il rumore da seed del modello sparso è dieci volte quello del denso**
  (0,36 contro 0,04 su validation), perché `grid_sample` non ha un backward
  deterministico su CUDA. Il margine di +0,21 dichiarato dagli autori,
  misurato su un solo seed, vale 0,6 deviazioni di quel rumore.
- **La saliency appresa vale 0,76 ± 0,25 punti**: congelando le regioni su una
  griglia regolare fissa, a parità *esatta* di parametri e FLOPs, il modello fa
  ancora 94,94 %. Quasi tutta la prestazione viene dall'architettura, non da
  *dove* il modello guarda.
- **Il guadagno in FLOPs non si traduce in latenza**, e cambia segno col batch.

### Latenza: il guadagno in FLOPs cambia segno col batch

```bash
python scripts/benchmark_models.py     # scrive results/benchmark.json
```

Il confronto è contro il denso `flatten`, cioè il baseline adottato.

| Modello | GFLOP | ms @ batch 1 | ms @ batch 64 |
|---|---|---|---|
| Denso `flatten` | 0,5604 | 2,97 | 11,77 |
| Sparso | 0,0485 | **4,79** | **6,48** |
| rapporto | 0,087× | 1,61× (più lento) | 0,55× (più veloce) |

Un fattore 11,6 sui FLOPs vale 0,62 a batch 1 e 1,82 a batch 64. Il regime
in cui il metodo perde è batch 1 — cioè il dispositivo edge che motiva il
lavoro: a batch 1 nessuna delle matrici satura la GPU e il tempo è
interamente costo di lancio dei kernel, che il modello sparso paga di più;
a batch 64 le matrici del denso diventano grandi abbastanza da far contare
l'aritmetica, e l'ordine si rovescia.

### Cosa non è stato fatto

L'ottimizzazione automatica degli iperparametri (Optuna / Hyperband) era in
programma e non è stata eseguita: tutti i valori vengono dal paper, dagli
ancestor o dalle assunzioni dichiarate sotto. Le ablation su *N* e la
penalità del budget a 50 epoche sono state misurate, quella su *P* no.

---

## Installazione

Richiede Python 3.12 e una GPU NVIDIA. Su Blackwell (sm_120, es. RTX 5050)
la build CUDA 12.8 **non è opzionale**: le wheel di default arrivano a sm_90
e falliscono a runtime con `no kernel image is available for execution on
the device`.

```bash
python -m venv .venv
.venv\Scripts\activate                 # Windows
python -m pip install -U pip setuptools wheel

# passo 1 — stack CUDA dall'indice PyTorch
pip install torch==2.9.1+cu128 torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu128

# passo 2 — il resto da PyPI
pip install -r requirements.txt

# verifica
python scripts/check_env.py
```

`check_env.py` deve stampare `compute capability : (12, 0)`, `kernel di prova : OK`
e trovare `soundfile`. Se non è così, fermarsi lì.

`requirements.txt` garantisce che l'installazione vada a buon fine;
`requirements.lock.txt` fissa le versioni **esatte** con cui sono stati
prodotti i risultati in `results/`.

---

## Riproduzione

### 1. Dati

```bash
python scripts/prepare_data.py --root data/raw --cache data/cache
```

Scarica Speech Commands V2 (~2,3 GB), verifica gli split ufficiali e
costruisce una cache memory-mapped `int16` (~3,4 GB su disco). I dati non
sono nel repository.

Lo script **si ferma con errore** se gli split non danno esattamente:

| Split | Campioni |
|---|---|
| train | 84 843 |
| validation | 9 981 |
| test | 11 005 |

Sono i conteggi dichiarati nel paper e coincidono con gli elenchi ufficiali
`validation_list.txt` / `testing_list.txt`, che sono **speaker-disjoint**.
Uno split casuale gonfia l'accuratezza di punti interi e rende i numeri
non confrontabili.

### 2. Training

Sulla RTX 5050 Laptop un'epoca costa **~80 s**: circa **2,2 ore** per un run
da 100 epoche, e i 19 run archiviati valgono ~35 ore di GPU.

I due run che portano il risultato principale:

```bash
python scripts/run_seeds.py --config configs/dense_flatten.yaml --seeds 0 1 2
python scripts/run_seeds.py --config configs/sparse.yaml        --seeds 0 1 2
```

`run_seeds.py` salta i seed già completati, aggrega media e deviazione in
`results/<prefix>_aggregate.json` e si può interrompere e riprendere.

Le altre righe della tabella, una per comando. `--set` **richiede** `--tag`:
senza, il run scriverebbe nella cartella del run di base e lo
sovrascriverebbe in silenzio.

```bash
# denso pool_freq — la ricostruzione scartata, nelle due letture dei canali
python -m src.train --config configs/dense.yaml --seed 0 --tag dense_c96_seed0
python -m src.train --config configs/dense.yaml --seed 0 --tag dense_seed0 \
    --set model.channel_reading=c196_proj96

# ablation: regioni congelate sulla griglia (3 seed)
for s in 0 1 2; do
  python -m src.train --config configs/sparse.yaml --seed $s \
      --tag sparse_grid_seed$s --set model.region_mode=grid
done

# ablation: una sola ripetizione dell'estrattore, e regioni vincolate al piano
python -m src.train --config configs/sparse.yaml --seed 0 \
    --tag sparse_rep1_seed0 --set model.repeats=1
python -m src.train --config configs/sparse.yaml --seed 0 \
    --tag sparse_clip_seed0 --set model.region_constraint=clip

# ablation su N a 50 epoche (il budget ridotto costa 0,80 punti: misurato)
for n in 4 9 16 25 36; do
  python -m src.train --config configs/sparse.yaml --seed 0 --epochs 50 \
      --tag sparse_N${n}_seed0 --set model.num_tokens=$n
done

# N=16 anche a 100 epoche, per separare l'effetto di N da quello del budget
python -m src.train --config configs/sparse.yaml --seed 0 \
    --tag sparse_N16_ep100_seed0 --set model.num_tokens=16
```

Altre opzioni: `--epochs N` (sovrascrive la config), `--resume` (riprende da
`last.pt`), `--stop-after N` (ferma dopo N epoche lasciando lo schedule
configurato per il totale, per spezzare un run lungo su più sessioni),
`--no-deterministic`.

Ogni run scrive in `results/<tag>/`: `config.yaml` e `resolved_config.json`
archiviati, `metrics.csv` per epoca, `best.pt` e `last.pt`, `summary.json`,
`validation_report.json` e i log TensorBoard in `tb/`. Nel repository sono
versionati **solo** i file di testo: pesi e log binari restano in locale.

### 3. Valutazione finale

```bash
python scripts/evaluate.py --run dense_flatten_seed0
```

Il test set si tocca **una volta sola**. Lo script scrive
`TEST_EVALUATED.json` nella cartella del run e si rifiuta di ripartire, a
meno di `--force` — che però registra l'accaduto. Il checkpoint si seleziona
sul validation set, durante il training, e si valuta sempre il **modello
EMA**, non i pesi correnti.

### 4. Figure

```bash
python scripts/plot_results.py                       # accuracy_vs_flops.png + summary_table.md
python scripts/benchmark_models.py                   # benchmark.json: costo e latenza
python scripts/plot_sampling.py --run sparse_seed0   # sampling_trace.png
```

---

## Mappa dei run archiviati

Ogni cartella sotto `results/` corrisponde a una riga di
`results/summary_table.md`. I nomi sono quelli con cui i run sono stati
eseguiti e non vengono cambiati a posteriori: `resolved_config.json` accanto
ai risultati riporta la configurazione realmente usata.

| Cartella | Configurazione | Epoche |
|---|---|---|
| `dense_flatten_seed{0,1,2}` | denso `flatten`, c96 — **baseline adottato** | 100 |
| `sparse_seed{0,1,2}` | sparso N=4 P=36 — **configurazione del paper** | 100 |
| `dense_c96_seed0` | denso `pool_freq`, c96 | 100 |
| `dense_seed0` | denso `pool_freq`, c196_proj96 (prima lettura dei canali) | 100 |
| `sparse_grid_seed{0,1,2}` | sparso, regioni congelate sulla griglia | 100 |
| `sparse_rep1_seed0` | sparso, `L_rep`=1 | 100 |
| `sparse_clip_seed0` | sparso, regioni vincolate al piano | 100 |
| `sparse_N16_ep100_seed0` | sparso N=16 | 100 |
| `sparse_N{4,9,16,25,36}_seed0` | ablation su N | 50 |

Solo `dense_flatten_seed{0,1,2}` e `sparse_seed{0,1,2}` hanno un
`test_report.json`: sono gli unici sei run per cui il test set è stato
toccato.

---

## Assunzioni dichiarate

Il paper non specifica quanto segue. Ogni voce è una scelta nostra, ed è
marcata `[ASSUNZIONE]` anche in `configs/base.yaml`.

| Punto | Scelta | Impatto |
|---|---|---|
| **Parametri dello spettrogramma** — mai dichiarati: mel o lineare, `n_fft`, hop, numero di bin | log-mel, 25 ms / 10 ms, 64 bin (→ 101 frame) | **il più alto di tutti**: da trattare come iperparametro |
| Contraddizione interna nel paper (§3.2): feature «96 dimensional» ma «196 kernels» 7×7 stride 2 | c96; il 196 è un refuso, lo dimostrano i FLOPs del modello sparso | medio |
| Numero di layer della early convolution | uno solo: il plurale della §3.2 è stato testato e confutato (2 conv → +164 % di FLOPs) | medio |
| Iperparametri del dense-transformer — di cui il paper dà solo params e FLOPs | ricostruiti cercando fra 432 combinazioni; due varianti (`pool_freq`, `flatten`) che sbagliano in direzioni opposte | alto sul baseline |
| Numero di epoche | 100 | medio (a 50 epoche si perdono 0,80 punti: misurato) |
| Formula esatta di *PhaseMix* (EAT la descrive in una riga) | ampiezze mescolate, fasi interpolate sui vettori unitari | basso/medio |
| Parametro `alpha` del Beta per il mixing | 0,2 | basso |
| Codifica posizionale del denso — dichiarata assente sullo sparso, taciuta sul denso | assente, per simmetria | basso, non testata |
| Convenzione FLOPs vs MAC del paper | FLOPs, dedotta dai numeri del paper stesso | **alto sul claim principale** |

### Sulla misura dei FLOPs

`src/flops.py` conta con `torch.utils.flop_counter`, nativo in torch, e
riporta **FLOPs** — non MAC, che sono la metà. Il paper non dichiara la
convenzione: è stata dedotta dai suoi stessi numeri, e il contatore è stato
scritto *prima* dei modelli, perché uno scritto dopo tende a confermare il
risultato atteso.

In fase 2 girava in parallelo anche `fvcore` come secondo contatore
indipendente, proprio per decidere fra le due convenzioni; i due
concordavano sul fattore 2 esatto. Deciso il punto, il secondo contatore non
discriminava più nulla ed è stato rimosso insieme alla sua dipendenza.

`grid_sample` (l'interpolazione bilineare del campionamento sparso) non è
contata dal contatore: va aggiunta a mano con `grid_sample_flops()`. Vale lo
0,3 % del totale — irrilevante nel numero, ma è il meccanismo che il paper
sta difendendo, e ometterlo sarebbe scorretto.

---

## Target di riproduzione

Valori riportati da Kavaki & Mandel, Tabella 2, su Speech Commands V2:

| Modello | Params | FLOPs | Accuracy | Nota |
|---|---|---|---|---|
| Sparse-transformer (N=4, P=36) | 2,87 M | 0,055 G | **96,88 %** | riprodotto: 95,52 % |
| Dense-transformer | 4,80 M | 0,645 G | 96,67 % | riprodotto: 97,03 % |
| EAT-S | 5,30 M | 0,373 G | 98,15 % | *citato*, non rimisurato dagli autori |
| HTS-AT | 28,80 M | 4,066 G | 98,00 % | citato |
| SimPF (MobileNetV2) | 3,40 M | 0,244 G | 95,60 % | citato |

I due modelli riprodotti tornano entro l'1,6 % sui parametri e il 12 % sui
FLOPs dichiarati; è l'*ordine fra i due* che non si riproduce.

---

## Struttura

```
configs/                 YAML, uno per esperimento; base.yaml e' ereditato
  base.yaml              dati, feature, augmentation, ottimizzatore, logging
  dense.yaml             baseline denso, variante pool_freq
  dense_flatten.yaml     baseline denso, variante flatten — quello adottato
  sparse.yaml            il modello del paper
src/
  data/
    speech_commands.py   download, split ufficiali, cache memmap
    dataset.py           Dataset e DataLoader
    features.py          log-mel (GPU)
    augment.py           mixing e phasemix (GPU, a livello di batch)
  models/
    frontend.py          early convolution (stem del paper)
    transformer.py       encoder reimplementato
    dense.py             baseline denso
    sparse.py            estrattore sparso: regioni, campionamento, decoding
    sparse_model.py      modello sparso completo
  flops.py               contatore di FLOPs e misura di latenza
  utils.py               seed, config, override, EMA, provenienza
  train.py               training loop
  metrics.py             metriche per classe e confusioni
  tracking.py            CSV per epoca e TensorBoard
scripts/
  check_env.py           verifica ambiente
  prepare_data.py        scarica i dati e costruisce la cache
  evaluate.py            valutazione sul test set, con guardia
  run_seeds.py           piu' seed, con media e deviazione standard
  benchmark_models.py    costo e latenza dei modelli a piu' batch
  plot_results.py        la figura principale e summary_table.md
  plot_sampling.py       la figura del campionamento sparso
tests/
  test_pipeline.py       forme, invarianti, contatore FLOPs (non serve il dataset)
results/                 metriche, configurazioni e report di ogni run
```

## Test

54 test, in circa 50 s e **senza il dataset scaricato** — verificano forme,
invarianti delle augmentation, equivalenza dell'encoder con
`nn.MultiheadAttention`, contatore di FLOPs, EMA, riproducibilità dei seed,
geometria delle regioni e il raggruppamento dei run:

```bash
pytest -q
```

---

## Codice di terze parti

Nessun codice copiato. Il repository ufficiale di EAT
(`github.com/Alibaba-MIIL/AudioClassfication`) è stato **consultato** solo per
sciogliere ambiguità sulle augmentation e sui parametri dell'ottimizzatore;
i punti in cui ha deciso una scelta sono annotati nel codice e nei config.
Non esiste codice pubblico per Kavaki & Mandel: il metodo è reimplementato
dalle formule del paper e da SparseFormer.

## Dati

Non inclusi nel repository. Google Speech Commands V2:
`http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz`
