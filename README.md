# Audio Sparse-Transformer — riproduzione

Progetto d'esame per **B031278 Deep Learning** (Fall 2025, prof. Paolo Frasconi),
MSc in Artificial Intelligence, Università di Firenze.

Riproduzione di:

> H. Salami Kavaki, M. I. Mandel, *Audio Sparse-Transformer for Speech
> Classification*, ICASSP 2025, pp. 1–5. DOI 10.1109/ICASSP49660.2025.10890475

con riferimento a:

> A. Gazneli, G. Zimerman, T. Ridnik, G. Sharir, A. Noy, *End-to-End Audio
> Strikes Back*, arXiv:2204.11479, 2022.

Il metodo è la trasposizione all'audio di **SparseFormer** (Gao et al.,
arXiv:2304.03768); la parametrizzazione delle regioni viene da Faster R-CNN,
il decoding adattivo da AdaMixer, il front-end da Xiao et al. 2021.

---

## Stato

| Fase | Contenuto | Stato |
|---|---|---|
| 0 | Lettura paper + ancestor, setup | completata (restano form risorse e ricevimento) |
| 1 | Prerequisiti teorici | in corso, in parallelo |
| 2 | Dati, split ufficiali, contatore FLOPs | **completata** |
| 3 | Baseline: early conv + dense transformer | **completata (seed 0): `flatten` 97,01 %, +0,34 sul dichiarato** |
| 4 | Sparse feature extractor | **completata (seed 0): 95,57 %, −1,44 dal denso** |
| 5 | Ablation, seed multipli, HPO | da fare |
| 6 | Presentazione | da fare |

### Risultati finora (validation, modello EMA, seed 0)

| Modello | **Test** (3 seed) | Validation | Params | GFLOP |
|---|---|---|---|---|
| **Denso `flatten`** — baseline | **97,03 % ± 0,20** | 96,97 ± 0,04 | 5 197 795 | 0,5604 |
| **Sparso N=4** — configurazione del paper | **95,52 % ± 0,31** | 95,71 ± 0,37 | 2 822 711 | 0,0485 |
| Sparso N=16 | — | 96,46 % | 2 823 527 | 0,1487 |
| Sparso, regioni congelate su griglia — 3 seed | — | 94,94 % ± 0,22 | 2 822 711 | 0,0485 |
| Sparso, `L_rep`=1 | — | 94,98 % | 2 010 591 | 0,0349 |
| Sparso, regioni vincolate al piano | — | 95,21 % | 2 822 711 | 0,0485 |
| Denso `pool_freq` — ricostruzione scartata | — | 94,09 % | 4 875 235 | 0,5275 |

L'obiettivo è **riprodurre l'esperimento e capirlo**, non far tornare le cifre:
uno scarto di qualche punto sui valori assoluti è atteso e nasce dalle
assunzioni che il paper lascia aperte. Su quel piano l'esperimento è
riprodotto — il nostro denso è mezzo punto *sopra* il valore dichiarato.

**Il risultato è il confronto interno.** Il paper dà il modello sparso 0,21
punti *sopra* il proprio dense-transformer; con la stessa pipeline il nostro
resta **1,52 punti sotto sul test set** (*t* = 7,2; −1,26 su validation). È un
cambio di *segno* su un confronto controllato, che nessuna tolleranza sui
livelli assorbe. Il test set è stato toccato una volta sola per modello, a
esperimenti chiusi.

Tre risultati collaterali che il paper non riporta:

- **Il rumore da seed del modello sparso è dieci volte quello del denso**
  (0,36 contro 0,04), perché `grid_sample` non ha un backward deterministico
  su CUDA. Il margine di +0,21 dichiarato dagli autori, misurato su un solo
  seed, vale 0,6 deviazioni di quel rumore.
- **La saliency appresa vale 0,76 ± 0,25 punti**: congelando le regioni su una
  griglia regolare fissa, a parità esatta di parametri e FLOPs, il modello fa
  ancora 94,94 %. Quasi tutta la prestazione viene dall'architettura, non da
  *dove* il modello guarda.
- **Il guadagno in FLOPs non si traduce in latenza**, e cambia segno col batch
  (vedi sotto).

### Latenza: il guadagno in FLOPs cambia segno col batch

```
python scripts/benchmark_models.py
```

| Modello | GFLOP | ms @ batch 1 | ms @ batch 64 |
|---|---|---|---|
| Denso `c96` | 0,5275 | 2,95 | 11,08 |
| Sparso | 0,0485 | **4,65** | **6,49** |
| rapporto | 0,092× | 1,58× (più lento) | 0,59× (più veloce) |

Un fattore 10,9 sui FLOPs vale 0,63 a batch 1 e 1,71 a batch 64. Il regime
in cui il metodo perde è batch 1 — cioè il dispositivo edge che motiva il
lavoro: a batch 1 nessuna delle matrici satura la GPU e il tempo è
interamente costo di lancio dei kernel, che il modello sparso paga di più;
a batch 64 le matrici del denso diventano grandi abbastanza da far contare
l'aritmetica, e l'ordine si rovescia.

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

---

## Riproduzione

### 1. Dati

```bash
python scripts/prepare_data.py --root data/raw --cache data/cache
```

Scarica Speech Commands V2 (~2.3 GB), verifica gli split ufficiali e
costruisce una cache memory-mapped `int16` (~3.4 GB su disco).

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

```bash
python -m src.train --config configs/dense.yaml --seed 0
```

Opzioni utili: `--epochs N` (sovrascrive la config), `--tag nome` (cartella del
run), `--resume` (riprende da `last.pt`), `--deterministic`.

Ogni run scrive in `results/<tag>/`: `config.yaml` e `resolved_config.json`
archiviati, `metrics.csv` per epoca, `best.pt` e `last.pt`, `summary.json`,
e i log TensorBoard in `tb/`.

Tempi misurati sulla RTX 5050 Laptop: **~80 s per epoca**, quindi circa
**2,2 ore** per un run da 100 epoche.

### 3. Valutazione finale

```bash
python scripts/evaluate.py --run dense_seed0
```

Il test set si tocca **una volta sola**. Lo script scrive
`TEST_EVALUATED.json` nella cartella del run e si rifiuta di ripartire, a
meno di `--force` — che però registra l'accaduto. Il checkpoint si seleziona
sul validation set, durante il training.

Si valuta sempre il **modello EMA**, non i pesi correnti.

---

## Target di riproduzione

Valori riportati da Kavaki & Mandel, Tabella 2, su Speech Commands V2:

| Modello | Params | FLOPs | Accuracy | Nota |
|---|---|---|---|---|
| Sparse-transformer (N=4, P=36) | 2,87 M | 0,055 G | **96,88 %** | obiettivo Fase 4 |
| Dense-transformer | 4,80 M | 0,645 G | 96,67 % | obiettivo Fase 3 |
| EAT-S | 5,30 M | 0,373 G | 98,15 % | *citato*, non rimisurato dagli autori |
| HTS-AT | 28,80 M | 4,066 G | 98,00 % | citato |
| SimPF (MobileNetV2) | 3,40 M | 0,244 G | 95,60 % | citato |

I nostri risultati andranno **accanto** a questi, con media e deviazione
standard su ≥ 3 seed.

---

## Assunzioni dichiarate

Il paper non specifica quanto segue. Ogni voce è una scelta nostra, ed è
marcata `[ASSUNZIONE]` anche in `configs/base.yaml`.

| Punto | Scelta | Impatto |
|---|---|---|
| **Parametri dello spettrogramma** — mai dichiarati: mel o lineare, `n_fft`, hop, numero di bin | log-mel, 25 ms / 10 ms, 64 bin (→ 101 frame) | **il più alto di tutti**: da trattare come iperparametro |
| Contraddizione interna nel paper (§3.2): feature «96 dimensional» ma «196 kernels» 7×7 stride 2 | da testare in entrambe le letture | medio |
| Numero di layer della early convolution | da fissare seguendo Xiao et al. 2021 | medio |
| Numero di epoche | 100, da verificare sulle curve | medio |
| Formula esatta di *PhaseMix* (EAT la descrive in una riga) | ampiezze mescolate, fasi interpolate sui vettori unitari | basso/medio — da confrontare col codice EAT |
| Parametro `alpha` del Beta per il mixing | 0,2 | basso |
| Convenzione FLOPs vs MAC del paper | da calibrare misurando EAT-S con entrambi i contatori | **alto sul claim principale** |

### Sulla misura dei FLOPs

`src/flops.py` usa **due contatori indipendenti**: `torch.utils.flop_counter`
(nativo) e `fvcore`. I due differiscono di ~2× perché fvcore conta MAC e li
chiama "flops". Il paper non dichiara la convenzione: va dedotta misurando
EAT-S e vedendo quale valore si avvicina a 0,373 G.

`grid_sample` (l'interpolazione bilineare del campionamento sparso) non è
contata da nessuno dei due: va aggiunta a mano con `grid_sample_flops()`.

---

## Struttura

```
configs/          YAML, uno per esperimento riportato
src/
  data/
    speech_commands.py   download, split ufficiali, cache memmap
    dataset.py           Dataset e DataLoader
    features.py          log-mel (GPU)
    augment.py           mixing e phasemix (GPU, a livello di batch)
  flops.py               doppio contatore + latenza
  utils.py               seed, config, EMA
  models/
    frontend.py          early convolution (stem del paper)
    transformer.py       encoder reimplementato
    dense.py             baseline denso
    sparse.py            estrattore sparso: regioni, campionamento, decoding
    sparse_model.py      modello sparso completo
  train.py               training loop
  metrics.py             metriche per classe e confusioni
  tracking.py            CSV, TensorBoard, W&B
scripts/
  check_env.py           verifica ambiente
  prepare_data.py        scarica i dati e costruisce la cache
  evaluate.py            valutazione sul test set, con guardia
  run_seeds.py           piu' seed, con media e deviazione standard
  benchmark_models.py    costo e latenza dei modelli a piu' batch
  plot_sampling.py       la figura del campionamento sparso
tests/
  test_pipeline.py       forme, invarianti, contatore FLOPs (non serve il dataset)
results/                 metriche, configurazioni e report di ogni run
```

## Test

Girano senza il dataset scaricato — verificano forme, invarianti delle
augmentation, equivalenza dell'encoder con `nn.MultiheadAttention`, contatore
di FLOPs, EMA, riproducibilità dei seed e geometria delle regioni:

```bash
pytest -q
```

---

## Codice di terze parti

Nessun codice copiato. Il repository ufficiale di EAT
(`github.com/Alibaba-MIIL/AudioClassfication`) è stato **consultato** solo per
sciogliere ambiguità sulle augmentation; ogni uso sarà annotato nel punto
esatto del codice. Non esiste codice pubblico per Kavaki & Mandel: il metodo
è reimplementato dalle formule del paper e da SparseFormer.

## Dati

Non inclusi nel repository. Google Speech Commands V2:
`http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz`
